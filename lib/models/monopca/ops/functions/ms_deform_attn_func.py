# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import MultiScaleDeformableAttention as MSDA
import torch
import torch.nn.functional as F

class MSDeformAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # <-- under AMP, cast half/bfloat16 inputs to fp32
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step

        # these index tensors must be long
        if value_spatial_shapes.dtype != torch.long:
            value_spatial_shapes = value_spatial_shapes.long()
        if value_level_start_index.dtype != torch.long:
            value_level_start_index = value_level_start_index.long()

        # value / sampling / attention are already fp32 here under AMP
        out = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, ctx.im2col_step
        )

        # note: custom_fwd casts `out` back to the outer autocast dtype (fp16/bf16) on exit
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index,
                              sampling_locations, attention_weights)
        return out

    @staticmethod
    @custom_bwd  # <-- let AMP handle dtype casts in backward
    def backward(ctx, grad_output):
        (value, value_spatial_shapes, value_level_start_index,
         sampling_locations, attention_weights) = ctx.saved_tensors

        # the custom kernel only accepts fp32, so keep gradients fp32
        if grad_output.dtype != torch.float32:
            grad_output = grad_output.float()

        grad_value, grad_sampling_loc, grad_attn_weight = MSDA.ms_deform_attn_backward(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, grad_output, ctx.im2col_step
        )

        # gradients for the 6 forward inputs
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None



def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
