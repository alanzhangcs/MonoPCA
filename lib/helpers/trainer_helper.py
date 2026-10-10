import os
import tqdm

import torch
import numpy as np
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc
from utils.misc import is_main_process
import torch.distributed as dist


class Trainer(object):
    def __init__(self, cfg, model, optimizer, train_loader, test_loader, lr_scheduler, warmup_lr_scheduler, logger, loss, model_name, use_amp=False):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join("./" + cfg["save_path"], model_name)
        self.tester = None
        self.eval_epoch = cfg.get("eval_epoch", 0)
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16 if self.use_amp else None
        
        


        # Gradient clipping configuration
        self.clip_grad_norm = cfg.get("clip_grad_norm", 1.0)  # Default max norm of 1.0

        # GradScaler is only needed for float16 (bfloat16 has float32 exponent range, no overflow)
        self.scaler = torch.cuda.amp.GradScaler() if (self.use_amp and self.amp_dtype is torch.float16) else None

        # loading pretrain/resume model
        if cfg.get("pretrain_model"):
            assert os.path.exists(cfg["pretrain_model"])
            load_checkpoint(model=self.model, optimizer=None, filename=cfg["pretrain_model"], map_location=self.device, logger=self.logger)

        if cfg.get("resume_model", None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(model=self.model.to(self.device), optimizer=self.optimizer, filename=resume_model_path, map_location=self.device, logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))

    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg["max_epoch"]), dynamic_ncols=True, leave=True, desc="epochs") if is_main_process() else None
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg["max_epoch"]):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # set the DistributedSampler epoch so shuffling differs per epoch
            if isinstance(self.train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if is_main_process() and (self.epoch % self.cfg["save_frequency"]) == 0 and self.epoch > self.eval_epoch:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg["save_all"]:
                    ckpt_name = os.path.join(self.output_dir, "checkpoint_epoch_%d" % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, "checkpoint")

                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch), ckpt_name)

            if self.tester is not None and self.epoch > self.eval_epoch:
                if is_main_process():
                    self.logger.info("Test Epoch {}".format(self.epoch))

                results = self.tester.inference()

                if dist.is_initialized():
                    all_results = [None for _ in range(dist.get_world_size())]
                    dist.all_gather_object(all_results, results)

                    if is_main_process():
                        merged_results = {}
                        for part in all_results:
                            merged_results.update(part)
                        self.logger.info("==> Saving ...")

                        self.tester.save_results(merged_results)
                        cur_result = self.tester.evaluate()
                else:
                    self.logger.info("==> Saving ...")
                    self.tester.save_results(results)
                    cur_result = self.tester.evaluate()

                if is_main_process() and cur_result > best_result:
                    best_result = cur_result
                    best_epoch = self.epoch
                    ckpt_name = os.path.join(self.output_dir, "checkpoint_best")
                    save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch), ckpt_name)
                    ckpt_name_with_epoch = os.path.join(self.output_dir, "checkpoint_best_epoch_%d" % self.epoch)
                    save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch, best_result, best_epoch), ckpt_name_with_epoch)

                if is_main_process():
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

            if progress_bar is not None:
                progress_bar.update()

        if is_main_process():
            self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()

        if is_main_process():
            print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch + 1 == self.cfg["max_epoch"]), desc="iters") if is_main_process() else None

        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):

            inputs = inputs.to(self.device, dtype=torch.float32)
            calibs = calibs.to(self.device, dtype=torch.float32)

            # Move targets to device with appropriate dtypes
            for key in targets.keys():
                if key == "mask_2d":
                    # Keep mask_2d as boolean for indexing
                    targets[key] = targets[key].to(self.device, dtype=torch.bool)
                elif key == "labels" or key == "indices" or key == "heading_bin":
                    # Keep integer types as long
                    targets[key] = targets[key].to(self.device, dtype=torch.long)
                elif key == "img_size":
                    # Keep img_size as long for indexing
                    targets[key] = targets[key].to(self.device, dtype=torch.long)
                else:
                    # All other numeric data as float32
                    targets[key] = targets[key].to(self.device, dtype=torch.float32)

            img_sizes = targets["img_size"]

            target_focals = targets["target_focals"].to(self.device, dtype=torch.float32)

            targets = self.prepare_targets(targets, inputs.shape[0])

            # ====== forward ======
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs = self.model(inputs, calibs, target_focals, img_sizes)
                    mask_dict = None
                    detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)

                    weight_dict = self.detr_loss.weight_dict
                    detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
                    detr_losses = sum(detr_losses_dict_weighted)

                if self.amp_dtype is torch.float16:
                    # ====== backward with scaler ======
                    self.scaler.scale(detr_losses).backward()

                    # ====== gradient clipping with AMP ======
                    if self.clip_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    detr_losses.backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
            else:
                outputs = self.model(inputs, calibs, target_focals, img_sizes)
                mask_dict = None
                detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)

                weight_dict = self.detr_loss.weight_dict
                detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] for k in detr_losses_dict.keys() if k in weight_dict]
                detr_losses = sum(detr_losses_dict_weighted)

                # ====== backward ======
                detr_losses.backward()

                # ====== gradient clipping ======
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                self.optimizer.step()

            # reduce losses across ranks for logging only
            detr_losses_dict_reduced = misc.reduce_dict(detr_losses_dict, average=True)
            detr_losses_dict_log = {}
            detr_losses_log = 0
            for k in detr_losses_dict_reduced.keys():
                if k in weight_dict:
                    detr_losses_dict_log[k] = (detr_losses_dict_reduced[k] * weight_dict[k]).item()
                    detr_losses_log += detr_losses_dict_log[k]
            detr_losses_dict_log["loss_detr"] = detr_losses_log

            # print loss info
            flags = [True] * 5
            if is_main_process() and batch_idx % 30 == 0:
                print("----", batch_idx, "----")
                print("%s: %.2f, " % ("loss_detr", detr_losses_dict_log["loss_detr"]))
                for key, val in detr_losses_dict_log.items():
                    if key == "loss_detr":
                        continue
                    if key[-1].isdigit() and int(key[-1]) < len(flags):
                        if flags[int(key[-1])]:
                            print("")
                            flags[int(key[-1])] = False
                    print("%s: %.2f, " % (key, val), end="")
                print("")
                print("")

            if progress_bar is not None:
                progress_bar.update()

        if progress_bar is not None:
            progress_bar.close()

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets["mask_2d"]

        key_list = ["labels", "boxes", "calibs", "depth", "size_3d", "heading_bin", "heading_res", "ry_bin", "ry_res", "ry_sin_cos", "boxes_3d", "offsets", "real_h_2d"]
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
                if key == "depth_map":
                    target_dict[key] = val[bz]
                if key == "obj_region":
                    target_dict[key] = val[bz]
                if key == "target_focals":
                    target_dict[key] = val[bz]
                if key == "image_mask":
                    target_dict[key] = val[bz]
                if key == "img_patch14":
                    target_dict[key] = val[bz]
            targets_list.append(target_dict)
        return targets_list
