# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import torch
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class PosePredictor(DetectionPredictor):
     def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
         super().__init__(cfg, overrides, _callbacks)
         self.args.task = "pose"

     def postprocess(self, preds, img, orig_imgs, **kwargs):
         """Post-process predictions for keypoint-only pose model."""
         from ultralytics.engine.results import Results

         # Handle Pose model output: (combined, (feats, (cls_num, cls_color, kpt)))
         if isinstance(preds, (tuple, list)):
             preds = preds[0]  # Get combined tensor

         if not isinstance(orig_imgs, list):
             orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

         results = []
         for b_idx, (pred_data, orig_img, img_path) in enumerate(zip(
             preds, orig_imgs, self.batch[0]
         )):
             # pred_data shape: (nc_num + nc_color + nk, na)
             nc_num = self.model.model[-1].nc_num
             nc_color = self.model.model[-1].nc_color
             nkpt, ndim = self.model.model[-1].kpt_shape
             nk = nkpt * ndim

             # 分离预测
             num_scores = pred_data[:nc_num]  # (nc_num, na)
             color_scores = pred_data[nc_num:nc_num + nc_color]  # (nc_color, na)
             kpt_data = pred_data[nc_num + nc_color:]  # (nk, na)

             # 获取置信度
             num_conf, cls_indices = num_scores.max(dim=0)  # (na,)
             color_indices = color_scores.argmax(dim=0)  # (na,)

             # 置信度过滤
             valid_mask = num_conf > self.args.conf

             if valid_mask.sum() == 0:
                 results.append(Results(
                     orig_img, path=img_path, names=self.model.names,
                     keypoints=torch.zeros((0, nkpt, ndim),device=preds.device)
                 ))
                 continue

             valid_conf = num_conf[valid_mask]
             valid_cls = cls_indices[valid_mask]
             valid_color = color_indices[valid_mask]
             valid_kpts = kpt_data[:, valid_mask]  # (nk, num_valid)

             # Reshape keypoints: (nk, num_valid) -> (num_valid, nkpt, ndim)
             num_valid = valid_kpts.shape[1]
             kpts = valid_kpts.view(nkpt, ndim, num_valid).permute(2, 0, 1)

             kpts_for_nms = kpts.clone()
             x_min = kpts_for_nms[..., 0].min(dim=1).values
             y_min = kpts_for_nms[..., 1].min(dim=1).values
             x_max = kpts_for_nms[..., 0].max(dim=1).values
             y_max = kpts_for_nms[..., 1].max(dim=1).values
             boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

            # 修复的NMS：实际执行NMS过滤
             from torchvision.ops import nms as torchvision_nms
             keep = torchvision_nms(boxes.float(), valid_conf.float(), self.args.iou)
             keep = keep[:self.args.max_det]

            # 缩放关键点到原图尺寸
             final_kpts = ops.scale_coords(img.shape[2:], kpts[keep], orig_img.shape)
             valid_conf = valid_conf[keep]
             valid_cls = valid_cls[keep]
             valid_color = valid_color[keep]

            
             img_h, img_w = orig_img.shape[:2]
             valid_in_range = (
                (final_kpts[..., 0] >= 0) & (final_kpts[..., 0] < img_w) &
                (final_kpts[..., 1] >= 0) & (final_kpts[..., 1] < img_h)
             ).all(dim=1)  # (num_pred,) - 所有关键点都在范围内

             final_kpts = final_kpts[valid_in_range]
             valid_conf = valid_conf[valid_in_range]
             valid_cls = valid_cls[valid_in_range]
             valid_color = valid_color[valid_in_range]

            
             if final_kpts.shape[0] > 0:
                kpt_x = final_kpts[..., 0]  # (num_pred, 4)
                kpt_y = final_kpts[..., 1]  # (num_pred, 4)

                x_min = kpt_x.min(dim=1).values
                x_max = kpt_x.max(dim=1).values
                y_min = kpt_y.min(dim=1).values
                y_max = kpt_y.max(dim=1).values

                width = x_max - x_min
                height = y_max - y_min

                min_area = 50.0  # 10x10
                valid_area = (width * height) >= min_area

               
                aspect_ratio = width / (height + 1e-6)
                aspect_ratio = torch.max(aspect_ratio, 1.0 / (aspect_ratio + 1e-6))
                valid_aspect = aspect_ratio <= 5.0

                valid_geometry = valid_area & valid_aspect
                final_kpts = final_kpts[valid_geometry]
                valid_conf = valid_conf[valid_geometry]
                valid_cls = valid_cls[valid_geometry]
                valid_color = valid_color[valid_geometry]


             results.append(Results(
                orig_img, path=img_path, names=self.model.names,
                keypoints=final_kpts
             ))

         return results
