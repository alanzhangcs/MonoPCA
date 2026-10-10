"""
MonoPCA: Rethinking Geometric Depth in Monocular 3D Object Detection
"""
import math
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import box_ops
from utils.misc import accuracy, get_world_size, is_dist_avail_and_initialized, inverse_sigmoid
from lib.losses.focal_loss import sigmoid_focal_loss
from lib.helpers.backbone_helper import build_resnet, build_vfm
from .matcher import build_matcher
from .depthaware_transformer import build_depthaware_transformer
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from .region_seg_head import RegionSegHead
from .attribute_net import AttributeNet
from .connector import Connector
from .mob import MoB
from .pcdepth import pc_depth

num_heading_bin = 12


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MonoPCA(nn.Module):
    """
    MonoPCA model for monocular 3D object detection.

    - MoB (Mixture of Backbone): fuse ResNet and EVA-02 multi-scale features
    - focal-length conditioning injected into feature maps and queries
    - region segmentation head producing region-enhanced features
    - depth predictor whose depth embedding is fed to the transformer encoder as an extra level
    - transformer decoder with iterative box refinement predicting 2D boxes + 3D attributes
    - PCDepth: geometry-based depth prior refined by a predicted offset
    """
    def __init__(self, mob, depthaware_transformer, depth_predictor, num_classes, num_queries,
                 aux_loss=True, group_num=11, target_focal_list=[700], focal_embedding_path=None):
        """ Initializes the model.
        Parameters:
            mob: MoB (Mixture of Backbone) module. See mob.py
            depthaware_transformer: transformer. See depthaware_transformer.py
            depth_predictor: depth predictor. See depth_predictor/depth_predictor.py
            num_classes: number of object classes
            num_queries: number of object queries per group
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            group_num: number of query groups used for one-to-many assignment during training
            target_focal_list: virtual focal lengths the images are resampled to
            focal_embedding_path: precomputed focal-length embeddings (.npy)
        """
        super().__init__()

        self.mob = mob
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        hidden_dim = depthaware_transformer.d_model
        self.hidden_dim = hidden_dim

        self.region_head = RegionSegHead(d_model=hidden_dim)
        self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim * 2)

        # prediction heads
        class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_embed.bias.data = torch.ones(num_classes) * bias_value
        bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        nn.init.constant_(bbox_embed.layers[-1].bias.data[2:], -2.0)

        # one head per decoder layer
        num_pred = depthaware_transformer.decoder.num_layers
        self.class_embed = _get_clones(class_embed, num_pred)
        self.bbox_embed = _get_clones(bbox_embed, num_pred)
        self.dim_embed_3d = _get_clones(MLP(hidden_dim, hidden_dim, 3, 2), num_pred)
        self.angle_embed = _get_clones(MLP(hidden_dim, hidden_dim, 24, 2), num_pred)
        self.depth_embed = _get_clones(MLP(hidden_dim, hidden_dim, 2, 2), num_pred)  # depth offset and log-variance
        self.sizeNet = _get_clones(AttributeNet(hidden_dim, hidden_dim, hidden_dim, 2), num_pred)
        self.angleNet = _get_clones(AttributeNet(hidden_dim, hidden_dim, hidden_dim, 2), num_pred)
        self.depthNet = _get_clones(AttributeNet(hidden_dim, hidden_dim, hidden_dim, 2), num_pred)
        # iterative bounding box refinement
        self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed

        # focal-length embeddings
        self.focal2idx = {round(val, 4): idx for idx, val in enumerate(target_focal_list)}
        focal_length_embeddings = np.load(focal_embedding_path, allow_pickle=True).item()
        focal_embedding = np.concatenate(
            [np.asarray(focal_length_embeddings[str(focal)], dtype=np.float32) for focal in target_focal_list], axis=0)
        self.focal_embedding = nn.Embedding.from_pretrained(torch.tensor(focal_embedding), freeze=True)
        self.connector = Connector(focal_embedding.shape[-1], hidden_dim, hidden_dim, 2, act="gelu")

    def forward(self, images, calibs, target_focal, img_sizes):
        srcs, masks, pos = self.mob(images)

        # Focal-length conditioning: map each sample's focal to a learned embedding,
        # then project it to `hidden_dim` for feature/query adaptation.
        batch_indices = torch.tensor([self.focal2idx[round(float(val.item()), 4)] for val in target_focal], device=images.device)
        batch_embeddings = self.connector(self.focal_embedding(batch_indices))

        # Feature-level adaptation: add the per-sample embedding to every pyramid level.
        for i in range(len(srcs)):
            srcs[i] += batch_embeddings[:, :, None, None]

        srcs, region_probs, seg_embed = self.region_head(srcs)

        # depth_embed is concatenated to the visual features as an extra encoder level
        pred_depth_map_logits, depth_embed = self.depth_predictor(srcs, masks[1], seg_embed[1] + pos[1])

        if self.training:
            query_embeds = self.query_embed.weight
        else:
            # in inference, only use one query group
            query_embeds = self.query_embed.weight[:self.num_queries]

        hs, init_reference, inter_references = self.depthaware_transformer(
            srcs, masks, pos, query_embeds, depth_embed)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # 3d center + 2d box
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_classes.append(self.class_embed[lvl](hs[lvl]))

            # 1) 3D size, with the focal embedding injected (query-level adaptation)
            size_feat = self.sizeNet[lvl](hs[lvl] + batch_embeddings.unsqueeze(1))
            size_feat = size_feat + hs[lvl]
            size3d_chain = self.dim_embed_3d[lvl](size_feat)  # [B, Q, 3]
            outputs_3d_dims.append(size3d_chain)

            # 2) angle
            angle_feat = self.angleNet[lvl](size_feat)
            angle_feat = angle_feat + size_feat
            angle_chain = self.angle_embed[lvl](angle_feat)  # [B, Q, 24]
            outputs_angles.append(angle_chain)
            ry = class2angle(angle_chain[:, :, :num_heading_bin], angle_chain[:, :, num_heading_bin:])

            # 3) depth: PCDepth prior from 2D box height, 3D size, yaw and calibration, plus a predicted offset
            depth_feat = self.depthNet[lvl](angle_feat + batch_embeddings.unsqueeze(1))
            depth_feat = depth_feat + angle_feat
            reg = self.depth_embed[lvl](depth_feat)  # [B, Q, 2] -> (residual offset, log-variance)
            depth_pc = pc_depth(size3d_chain, outputs_coord, ry, img_sizes, calibs)
            outputs_depths.append(torch.cat([reg[:, :, 0:1] + depth_pc.unsqueeze(-1),
                                             reg[:, :, 1:2]], -1))

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits
        out['pred_region_prob'] = region_probs

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth):
        return [{'pred_logits': a, 'pred_boxes': b,
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]


def class2angle(cls_logits, residual):
    """
    Decode heading bins into yaw in [-pi, pi].

    Args:
        cls_logits: [B, Q, num_heading_bin] bin logits
        residual:   [B, Q, num_heading_bin] per-bin residuals
    Returns:
        [B, Q] angles in radians
    """
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    cls_indices = torch.argmax(cls_logits, dim=-1, keepdim=True)  # [B, Q, 1]
    selected_residual = torch.gather(residual, -1, cls_indices).squeeze(-1)
    angle = cls_indices.squeeze(-1).float() * angle_per_class + selected_residual
    return torch.where(angle > np.pi, angle - 2 * np.pi, angle)


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoPCA.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, group_num=11):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        self.group_num = group_num

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
   
        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1] 
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance  
        
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes
        
        return losses  
    
    def loss_dims(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12, device=heading_target_cls.device).scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        depth_map_logits = outputs["pred_depth_map_logits"]

        b, l, h, w = depth_map_logits.shape

        num_gt_per_img = [len(t["boxes"]) for t in targets]
        gt_boxes2d = torch.cat([t["boxes"] for t in targets], dim=0) * torch.tensor([w, h, w, h], device=depth_map_logits.device)
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t["depth"] for t in targets], dim=0).squeeze(dim=1)

        losses = dict()

        losses["loss_depth_map"] = self.ddn_loss(depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def loss_region(self, outputs, targets, indices, num_boxes):
        region_probs = outputs['pred_region_prob']
        gt_region = torch.cat([t['obj_region'].unsqueeze(0) for t in targets], dim=0)

        loss = 0
        losses = dict()
        for region_prob in region_probs:
            gt_region_resized = F.interpolate(gt_region.unsqueeze(1).float(), size=region_prob.shape[2:], mode='bilinear', align_corners=True)
            # Compute intersection and union
            intersection = (region_prob * gt_region_resized).sum()
            total = region_prob.sum() + gt_region_resized.sum()
            # Compute Dice Coefficient
            dice_coef = (2. * intersection + 1) / (total + 1)
            # Compute Dice Loss
            dice_loss = 1 - dice_coef
            loss += dice_loss

        losses['loss_region'] = loss

        return losses
    

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            'region': self.loss_region,
        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss in self.losses:
                    if loss == "depth_map" or loss == "region":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # MoB: Mixture of Backbone (ResNet + EVA-02)
    resnet_backbone = build_resnet(cfg['resnet_backbone'])
    vfm_backbone = build_vfm(cfg['vfm_backbone'])
    mob = MoB(resnet_backbone, vfm_backbone, hidden_dim=cfg['hidden_dim'], num_channels=cfg['num_channels'],
              num_feature_levels=cfg['num_feature_levels'])

    depthaware_transformer = build_depthaware_transformer(cfg)
    depth_predictor = DepthPredictor(cfg)

    model = MonoPCA(
        mob,
        depthaware_transformer,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        group_num=cfg['group_num'],
        target_focal_list=cfg['target_focal_list'],
        focal_embedding_path=cfg['focal_embedding_path'])

    matcher = build_matcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    weight_dict['loss_region'] = cfg['region_loss_coef']

    # auxiliary decoder-layer losses share the same weights
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'depth_map', 'region']

    criterion = SetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        losses=losses,
        group_num=cfg['group_num'])

    device = torch.device(cfg['device'])
    criterion.to(device)

    return model, criterion
