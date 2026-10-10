import os
import tqdm
import shutil

import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections

import torch.distributed as dist
from utils.misc import is_main_process
class Tester(object):
    def __init__(
        self,
        cfg,
        model,
        dataloader,
        logger,
        train_cfg=None,
        model_name="monopca",
    ):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs  # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join("./" + train_cfg["save_path"], model_name)
        self.save_dir = os.path.join("./" + train_cfg["save_path"], model_name)
        self.dataset_type = cfg.get("type", "KITTI")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name
        self.eval_dataset = cfg["eval_dataset"] if "eval_dataset" in cfg.keys() else cfg["type"]
        self.checkpoint_path = cfg["checkpoint_path"] if "checkpoint_path" in cfg.keys() else None
        self.eval_waymo = cfg.get("eval_waymo", False)

    def test(self):
        assert self.cfg["mode"] in ["single", "all"]

        # test a single checkpoint
        if not is_main_process():
            self.logger = None  # only the main process writes logs

        checkpoint_path = self.checkpoint_path if self.checkpoint_path is not None else os.path.join(self.output_dir, "checkpoint_best.pth")

        assert os.path.exists(checkpoint_path)
        load_checkpoint(
            model=self.model,
            optimizer=None,
            filename=checkpoint_path,
            map_location=self.device,
            logger=self.logger,
        )
        self.model.to(self.device)
        results = self.inference()

        if dist.is_initialized():
            print(f"[Rank {dist.get_rank()}] Starting all_gather_object ...")
            all_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_results, results)
            print(f"[Rank {dist.get_rank()}] all_gather_object completed")
            if is_main_process():
                merged_results = {}
                for part in all_results:
                    merged_results.update(part)
                self.logger.info("==> Saving ...")

                self.save_results(merged_results)
                self.evaluate()
        else:
            self.logger.info("==> Saving ...")
            self.save_results(results)
            self.evaluate()

    def inference(self):

        torch.set_grad_enabled(False)
        self.model.eval()
        results = {}
        rank_str = f"[Rank {torch.distributed.get_rank()}]" if torch.distributed.is_initialized() else ""

        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc=f"{rank_str} Inference") if is_main_process() else None
        
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            inputs = inputs.to(self.device, dtype=torch.float32)
            calibs = calibs.to(self.device, dtype=torch.float32)
            
            # Move info data to device with appropriate dtypes
            img_sizes = info["img_size"].to(self.device, dtype=torch.long)
            target_focals = targets["target_focals"].to(self.device, dtype=torch.float32)
            with torch.no_grad():

                outputs = self.model(inputs, calibs, target_focals, img_sizes)
            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg["topk"])
            dets = dets.detach().cpu().numpy()

            # get calibs
            if info.get("dataset_name"):
                dataset_names = info["dataset_name"]
                calibs = [self.dataloader.dataset.get_calib(dataset_names[i], info["img_id"][i]) for i in range(len(info["img_id"]))]
                info.pop("dataset_name")
            else:
                calibs = [self.dataloader.dataset.get_calib(index) for index in info["img_id"]]

            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get("threshold", 0.2),
                large_object_id=self.cfg.get("large_object_id", []),
            )

            results.update(dets)
            if progress_bar is not None:
                progress_bar.update()

        if progress_bar is not None:
            progress_bar.close()

        return results  # each rank returns its own share of the results

    def save_results(self, results):
        output_dir = os.path.join(self.save_dir, "outputs", "data")
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            output_path = os.path.join(output_dir, "{:06d}.txt".format(img_id))
            f = open(output_path, "w")
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write("{} 0.0 0".format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(" {:.2f}".format(results[img_id][i][j]))
                f.write("\n")
            f.close()

    def evaluate(self):
        results_dir = os.path.join(self.save_dir, "outputs", "data")
        assert os.path.exists(results_dir)


        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result
