# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


class PoseValidator(DetectionValidator):
    """Validator for keypoint-only pose model with dual classification branches.

    This validator handles pose estimation without bounding boxes, using keypoints
    for both localization and IoU computation. It supports dual classification
    branches for number (1-8) and color (R, B, N, P) prediction.

    Attributes:
        sigma (np.ndarray): Sigma values for OKS calculation.
        kpt_shape (list[int]): Shape of keypoints [nkpt, ndim].
        nc_color (int): Number of color classes (armor detection custom).
        color_names (list[str]): Names of color classes (armor detection custom).
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize PoseValidator for keypoint-only pose estimation (armor detection custom)."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.nc_color = 4
        self.color_names = ["B", "R", "N", "P"]
        self.nc_size = 2
        self.size_names = ["s", "b"]
        self.args.task = "pose"
        self.metrics = PoseMetrics()
        # Color classification stats
        self.color_correct = 0
        self.color_total = 0

    #对输入批次的数据进行预处理
    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch by converting keypoints and color data to float."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].float() #关键点转成浮点数
        if "color" in batch:
            batch["color"] = batch["color"].to(self.device)
        return batch
    
    def get_desc(self) -> str:
        """Return description of evaluation metrics for keypoint-only mode."""
        return ("%22s" + "%11s" * 6) % (
            "Class",
            "Images",
            "Instances",
            "P",
            "R",
            "mAP50",
            "mAP50-95",
        )
    
    
    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for pose validation."""
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

        # 初始化颜色分类和数字类别数
        # 从模型的头部获取nc_num（它存储的是nc_num，而不是总的nc）
        pose_head = None
        inner_model = model.model if hasattr(model, "model") else model
        # Get the actual nn.Sequential layers
        model_layers = inner_model.model if hasattr(inner_model, "model") else inner_model
        for layer in reversed(list(model_layers)):
            if hasattr(layer, "nc_num"):
                pose_head = layer
                break

        # If found, use the pose head's nc_num. Otherwise, infer from data
        if pose_head:
            self.nc_num = pose_head.nc_num
        else:
            # Fallback: try to get from data if available
            self.nc_num = self.data.get("nc", 8)  # default to 8 if not found

        self.nc_color = len(self.data.get("color_names", ["B", "R", "N", "P"]))
        self.color_names = self.data.get("color_names", ["B", "R", "N", "P"])
        self.color_correct = 0
        self.color_total = 0

        self.nc_size = len(self.data.get("size_names", ["s", "b"]))
        self.size_names = self.data.get("size_names", ["s", "b"])

        # Override names from data config (not model) to use correct armor class names
        if "names" in self.data:
            self.names = self.data["names"]
            self.nc = len(self.names)

    #对模型输出进行后处理，生成可用的关键点和分类预测
    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Postprocess predictions to extract keypoints and classifications.

        The model outputs: (combined, (feats, (cls_num, cls_color, cls_size, kpt)))
        where combined = (bs, nc_num + nc_color + nc_size + nk, na) with sigmoid'd class scores
        and decoded keypoints.
        """
        # Handle validation mode output format
        if isinstance(preds, (tuple, list)):
            preds = preds[0]  # Get combined tensor

        bs, channels, na = preds.shape
        nc_num = self.nc_num  # Use nc_num (number of number classes), not self.nc (total classes)
        nc_color = self.nc_color
        nc_size = self.nc_size
        nkpt, ndim = self.kpt_shape
        nk = nkpt * ndim

        results = []
        for b_idx in range(bs):


            # print(f"    num_conf max={num_conf.max().item():.3f}, color_indices={color_indices.tolist()}", flush=True)
            pred_data = preds[b_idx]  # (nc_num + nc_color + nc_size + nk, na)

            # 分离预测
            num_scores = pred_data[:nc_num]  # 数字类别概率
            color_scores = pred_data[nc_num:nc_num + nc_color]  # (nc_color, na)
            size_scores = pred_data[nc_num + nc_color:nc_num + nc_color + nc_size]  # (nc_size, na)
            kpt_data = pred_data[nc_num + nc_color + nc_size:]  # (nk, na) - decoded keypoints

            # print(f"[DEBUG postprocess] sample {b_idx}: num_scores shape={num_scores.shape}, "f"color_scores shape={color_scores.shape}, kpt_data shape={kpt_data.shape}", flush=True)
            # Get confidence from number classification
            num_conf, cls_indices = num_scores.max(dim=0)  # (na,)
            color_indices = color_scores.argmax(dim=0)  # (na,)
            size_indices = size_scores.argmax(dim=0)  # (na,)
            # print(f"num_conf: min={num_conf.min():.6f}, max={num_conf.max():.6f}, mean={num_conf.mean():.6f}")

            #决定哪些进入 NMS,影响最终输出
            conf_thres = 0.5
            max_dets = 50 # 最大检测数print(f"num_conf: min={num_conf.min():.6f}, max={num_conf.max():.6f}, mean={num_conf.mean():.6f}")

            # 按置信度过滤
            valid_mask = num_conf > conf_thres
            # valid_mask = torch.ones_like(num_conf, dtype=torch.bool)

            if valid_mask.sum() == 0:
                results.append({
                    "cls": torch.zeros((0,), dtype=torch.long, device=preds.device),
                    "conf": torch.zeros((0,), device=preds.device),
                    "keypoints": torch.zeros((0, nkpt, ndim), device=preds.device),
                    "color_cls": torch.zeros((0,), dtype=torch.long, device=preds.device),
                    "size_cls": torch.zeros((0,), dtype=torch.long, device=preds.device),
                })
                continue


            # Get valid predictions
            valid_conf = num_conf[valid_mask]
            valid_cls = cls_indices[valid_mask]
            valid_color = color_indices[valid_mask]
            valid_size = size_indices[valid_mask]
            valid_kpts = kpt_data[:, valid_mask]  # (nk, num_valid)

            # Reshape keypoints to (num_valid, nkpt, ndim)
            num_valid = valid_kpts.shape[1]
            kpts = valid_kpts.view(nkpt, ndim, num_valid).permute(2, 0, 1)  # (num_valid, nkpt, ndim)

            # 按置信度排序，跳过 NMS（mAP 计算会处理重复检测）
            # sorted_indices = valid_conf.argsort(descending=True)[:max_dets]

            iou_thres = getattr(self.args, 'iou', 0.5)
            nms_keep = self._kpt_nms(kpts, valid_conf, iou_thres=iou_thres)
            sorted_indices = valid_conf[nms_keep].argsort(descending=True)[:max_dets]
            final_indices = nms_keep[sorted_indices]

            results.append({
                "cls": valid_cls[final_indices],
                "conf": valid_conf[final_indices],
                "keypoints": kpts[final_indices],
                "color_cls": valid_color[final_indices],
                "size_cls": valid_size[final_indices],
            })

        return results

    def _kpt_nms(self, kpts: torch.Tensor, scores: torch.Tensor, iou_thres: float = 0.5) -> torch.Tensor:
        """基于关键点边界框的 NMS。

        Args:
            kpts: (N, nkpt, 2) 关键点坐标
            scores: (N,) 置信度分数
            iou_thres: IoU 阈值


        Returns:
            keep: 保留的索引
        """
        if kpts.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=kpts.device)

        # 从关键点计算边界框 (x1, y1, x2, y2)
        x_coords = kpts[..., 0]  # (N, nkpt)
        y_coords = kpts[..., 1]  # (N, nkpt)

        x1 = x_coords.min(dim=1)[0]  # (N,)
        y1 = y_coords.min(dim=1)[0]
        x2 = x_coords.max(dim=1)[0]
        y2 = y_coords.max(dim=1)[0]

        boxes = torch.stack([x1, y1, x2, y2], dim=1)  # (N, 4)

        # 使用 torchvision 的 NMS 或手动实现
        try:
            from torchvision.ops import nms
            keep = nms(boxes, scores, iou_thres)
        except ImportError:
            # 手动实现 NMS
            keep = self._manual_nms(boxes, scores, iou_thres)

        return keep

    def _manual_nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
        """手动实现的 NMS。"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # 按分数降序排列
        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)

            if order.numel() == 1:
                break

            # 计算 IoU
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])

            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            # 保留 IoU 小于阈值的
            inds = torch.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _add_bbox_from_kpts(
        self,
        data: dict[str, torch.Tensor],
        is_gt: bool = False
    ) -> dict[str, torch.Tensor]:
        """从关键点生成边界框，供 confusion_matrix 使用。

        Args:
            data: 包含 'keypoints' 的字典
            is_gt: 是否是 ground truth 数据

        Returns:
            添加了 'bboxes' 字段的字典
        """
        result = {k: v for k, v in data.items()}
        kpts = data["keypoints"]

        if kpts.shape[0] == 0:
            result["bboxes"] = torch.zeros((0, 4), device=kpts.device)
            return result

        # 从关键点计算边界框 (x1, y1, x2, y2) 格式
        x_coords = kpts[..., 0]  # (N, nkpt)
        y_coords = kpts[..., 1]  # (N, nkpt)

        x1 = x_coords.min(dim=1)[0]  # (N,)
        y1 = y_coords.min(dim=1)[0]
        x2 = x_coords.max(dim=1)[0]
        y2 = y_coords.max(dim=1)[0]

        result["bboxes"] = torch.stack([x1, y1, x2, y2], dim=1)  # (N, 4)
        return result


    #准备单张图像的批次数据
    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare batch with keypoints scaled to image dimensions.

        For keypoint-only mode, we don't need bbox processing from parent class.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        h, w = imgsz

        # Process keypoints - scale to image dimensions
        kpts = batch["keypoints"][idx]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h

        # Process color labels
        if "color" in batch:
            color = batch["color"][idx]
            if color.dim() == 2:
                color = color.squeeze(-1)
        else:
            color = torch.zeros(kpts.shape[0], device=self.device, dtype=torch.long)

        # print(f"[DEBUG _prepare_batch] batch_idx={si}, imgsz={imgsz}, ori_shape={ori_shape}, ratio_pad={ratio_pad}", flush=True)
        # print(f"  cls={cls.tolist()}, keypoints shape={kpts.shape}, color shape={color.shape}", flush=True)


        return {
            "cls": cls,
            "keypoints": kpts,
            "color": color,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }
    #处理预测结果
    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepare predictions for evaluation."""
        if self.args.single_cls:
            pred["cls"] = pred["cls"] * 0
        return pred
    #更新关键点和颜色分类指标
    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update metrics using keypoint IoU matching."""
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            # if predn["keypoints"].numel() > 0:
            #     print(f"[DEBUG update_metrics] batch {si}: predn keypoints min={predn['keypoints'].min().item():.2f}, "
            #         f"max={predn['keypoints'].max().item():.2f}, conf min={predn['conf'].min().item():.3f}, "
            #         f"max={predn['conf'].max().item():.3f}", flush=True)
            # else:
            #     print(f"[DEBUG update_metrics] batch {si}: No predicted keypoints", flush=True)

            # # 打印 GT 关键点范围
            # if pbatch["keypoints"].numel() > 0:
            #     print(f"[DEBUG update_metrics] batch {si}: GT keypoints min={pbatch['keypoints'].min().item():.2f}, "
            #         f"max={pbatch['keypoints'].max().item():.2f}", flush=True)
            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0

            # Compute keypoint-based metrics
            stats = self._process_batch(predn, pbatch)

            self.metrics.update_stats(
                {
                    **stats,
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )

            if self.args.plots:                                           
                predn_with_bbox = self._add_bbox_from_kpts(predn)         
                pbatch_with_bbox = self._add_bbox_from_kpts(pbatch, is_gt=True)                                                                     
                self.confusion_matrix.process_batch(predn_with_bbox, pbatch_with_bbox, conf=self.args.conf)    

            if no_pred:
                continue

            # Save results if needed
            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )
    #处理单张图的关键点匹配和颜色准确率
    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Process batch using keypoint IoU for matching."""
        gt_cls = batch["cls"]

        if gt_cls.shape[0] == 0 or preds["cls"].shape[0] == 0:
            tp_p = np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)
        else:
            # Compute area from keypoints (no bbox available)
            gt_kpts = batch["keypoints"]
            x_coords = gt_kpts[..., 0]
            y_coords = gt_kpts[..., 1]
            widths = x_coords.max(dim=1)[0] - x_coords.min(dim=1)[0]
            heights = y_coords.max(dim=1)[0] - y_coords.min(dim=1)[0]
            area = widths * heights * 0.53  # Scale factor from COCO

            # Compute keypoint IoU
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()

        # For keypoint-only mode, use same metrics for both tp and tp_p
        tp = {"tp": tp_p, "tp_p": tp_p}

        # Color classification accuracy
        if gt_cls.shape[0] > 0 and preds["cls"].shape[0] > 0 and "color" in batch:
            gt_color = batch["color"]
            pred_color = preds["color_cls"]

            # Match using keypoint IoU
            gt_kpts = batch["keypoints"]
            x_coords = gt_kpts[..., 0]
            y_coords = gt_kpts[..., 1]
            area = (x_coords.max(dim=1)[0] - x_coords.min(dim=1)[0]) * \
                   (y_coords.max(dim=1)[0] - y_coords.min(dim=1)[0])
            iou_kpt = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)

            # For each prediction, find best matching GT
            for pi in range(len(preds["cls"])):
                if iou_kpt.shape[0] > 0:
                    iou_col = iou_kpt[:, pi] if iou_kpt.dim() == 2 else iou_kpt
                    best_iou = iou_col.max()

                    if best_iou > 0.5:
                        best_gt_idx = iou_col.argmax()
                        self.color_total += 1
                        if pred_color[pi] == gt_color[best_gt_idx]:
                            self.color_correct += 1

        return tp
    #完成指标计算
    def finalize_metrics(self, *args, **kwargs):
        """Finalize metrics including color accuracy."""
        super().finalize_metrics(*args, **kwargs)
        self.color_accuracy = self.color_correct / max(self.color_total, 1)

    def print_results(self):
        """Print results for keypoint-only pose model with color accuracy."""
        # For keypoint-only mode, use pose metrics only (skip box metrics)
        pose_results = self.metrics.pose.mean_results()  # P, R, mAP50, mAP50-95 for pose
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *pose_results))

        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"No labels found in {self.args.task} set, cannot compute metrics without labels")

        # Print color accuracy separately
        if self.color_total > 0:
            LOGGER.info(f"Color Classification: {self.color_correct}/{self.color_total} = {self.color_accuracy:.4f}")
    #获取指标统计字典
    def get_stats(self) -> dict[str, Any]:
        """Get statistics including color accuracy."""
        stats = super().get_stats()
        stats["color_accuracy"] = self.color_correct / max(self.color_total, 1)
        return stats
    #可视化验证集标签
    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation samples with keypoints and color labels."""
        from ultralytics.utils.plotting import plot_images

        # Create a copy without bboxes to avoid drawing both bbox labels and keypoint labels
        plot_batch = {k: v for k, v in batch.items() if k != "bboxes"}

        plot_images(
            labels=plot_batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            color_names=self.color_names,
            size_names=self.size_names,
            on_plot=self.on_plot,
        )
    #可视化模型预测
    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
    ) -> None:
        """Plot predicted keypoints with color labels."""
        from ultralytics.utils.plotting import plot_images

        if not preds or len(preds) == 0:
            LOGGER.warning(f"Val batch {ni}: No predictions to plot")
            return

        # Check if any predictions have content
        total_preds = sum(len(p["cls"]) for p in preds)
        # print(f"[DEBUG plot_predictions] batch {ni}: total predictions={total_preds}", flush=True)
        if total_preds == 0:
            LOGGER.warning(f"Val batch {ni}: All predictions empty")
            return

        # Add batch index to predictions
        # Note: keypoints from kpts_decode are already in absolute pixel coordinates,
        # plot_images will handle them correctly (it checks if max <= 1.01 for normalization)
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i
            # Keypoints are already in absolute pixel coordinates from kpts_decode,
            # no need to scale them here
            # print(f"  pred {i}: cls={pred['cls'].tolist()}, conf={pred['conf'].tolist()}, keypoints shape={pred['keypoints'].shape}", flush=True)

        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}

        # Map color_cls to color for plotting
        if "color_cls" in batched_preds:
            batched_preds["color"] = batched_preds["color_cls"]

        # Map size_cls to size for plotting
        if "size_cls" in batched_preds:
            batched_preds["size"] = batched_preds["size_cls"]

        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            color_names=self.color_names,
            size_names=self.size_names,
            on_plot=self.on_plot,
            conf_thres=0.5,  #决定在图上画哪些框
        )
    #将关键点预测从网络输入尺寸缩放回原图尺寸
    # def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
    #     """Scale predictions to original image size."""
    #     return {
    #         "keypoints": ops.scale_coords( #将关键点从网络输入尺寸缩放回原图尺寸
    #             pbatch["imgsz"],
    #             predn["keypoints"].clone(),
    #             pbatch["ori_shape"],
    #             ratio_pad=pbatch["ratio_pad"],
    #         ),
    #         "cls": predn["cls"],
    #         "conf": predn["conf"],
    #         "color_cls": predn.get("color_cls", torch.zeros_like(predn["cls"])),
    #     }

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scale predictions to original image size."""

        # 调用原函数逻辑进行缩放
        kpts_scaled = ops.scale_coords(
            pbatch["imgsz"],
            predn["keypoints"].clone(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        )

        # print(f"[DEBUG scale_preds] Processing image: {pbatch['im_file']}", flush=True)
        # if kpts_scaled.numel() > 0:
        #     print(f"  keypoints scaled min={kpts_scaled.min().item():.2f}, "
        #         f"max={kpts_scaled.max().item():.2f}, first kpt={kpts_scaled[0, 0, :].tolist()}", flush=True)
        # else:
        #     print("  No keypoints to scale", flush=True)


        # 保持原返回结构不变
        return {
            "keypoints": kpts_scaled,
            "cls": predn["cls"],
            "conf": predn["conf"],
            "color_cls": predn.get("color_cls", torch.zeros_like(predn["cls"])),
        }


    #将预测关键点保存为 TXT 文件
    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save predictions to text file (keypoints only, no bbox)."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)
    #将关键点预测转换为 COCO JSON 格式
    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Convert predictions to COCO JSON format (keypoints only)."""
        keypoints = predn["keypoints"]
        conf = predn["conf"]
        cls_pred = predn["cls"]

        stem = Path(pbatch["im_file"]).stem
        image_id = int(stem) if stem.isnumeric() else stem

        for i, kpts in enumerate(keypoints.tolist()):
            entry = {
                "image_id": image_id,
                "keypoints": [val for pt in kpts for val in pt],
                "score": float(conf[i].item() if isinstance(conf[i], torch.Tensor) else conf[i]),
                "category_id": int(cls_pred[i].item() if isinstance(cls_pred[i], torch.Tensor) else cls_pred[i]),
            }
            if hasattr(self, "jdict"):
                self.jdict.append(entry)
    #调用 COCO API 对 JSON 文件进行评估
    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate using COCO JSON format (keypoints only)."""
        anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"
        pred_json = self.save_dir / "predictions.json"
        return super().coco_evaluate(stats, pred_json, anno_json, ["keypoints"], suffix=["Pose"])
