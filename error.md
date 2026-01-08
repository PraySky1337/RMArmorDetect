      Epoch    GPU_mem  pose_loss  kobj_loss   cls_loss color_loss  size_loss  Instances       Size
: 0% ──────────── 0/88  1.1s
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/rry/.config/Ultralytics/DDP/_temp_sxgoh9kg140047087455488.py", line 13, in <module>
[rank0]:     results = trainer.train()
[rank0]:   File "/home/rry/RMArmorDetect/ultralytics/engine/trainer.py", line 243, in train
[rank0]:     self._do_train()
[rank0]:   File "/home/rry/RMArmorDetect/ultralytics/engine/trainer.py", line 427, in _do_train
[rank0]:     loss, self.loss_items = self.model(batch)
[rank0]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/parallel/distributed.py", line 1643, in forward
[rank0]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank0]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/parallel/distributed.py", line 1459, in _run_ddp_forward
[rank0]:     return self.module(*inputs, **kwargs)  # type: ignore[index]
[rank0]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:   File "/home/rry/RMArmorDetect/ultralytics/nn/tasks.py", line 136, in forward
[rank0]:     return self.loss(x, *args, **kwargs)
[rank0]:   File "/home/rry/RMArmorDetect/ultralytics/nn/tasks.py", line 328, in loss
[rank0]:     return self.criterion(preds, batch)
[rank0]:   File "/home/rry/RMArmorDetect/armor_detect/losses/armor_pose_loss.py", line 238, in __call__
[rank0]:     num_class_targets[b, valid_anchor_indices, cls_ids] = oks_soft
[rank0]: RuntimeError: Index put requires the source and destination dtypes match, got Half for the destination and Float for the source.
[rank1]: Traceback (most recent call last):
[rank1]:   File "/home/rry/.config/Ultralytics/DDP/_temp_sxgoh9kg140047087455488.py", line 13, in <module>
[rank1]:     results = trainer.train()
[rank1]:   File "/home/rry/RMArmorDetect/ultralytics/engine/trainer.py", line 243, in train
[rank1]:     self._do_train()
[rank1]:   File "/home/rry/RMArmorDetect/ultralytics/engine/trainer.py", line 427, in _do_train
[rank1]:     loss, self.loss_items = self.model(batch)
[rank1]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/parallel/distributed.py", line 1643, in forward
[rank1]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank1]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/parallel/distributed.py", line 1459, in _run_ddp_forward
[rank1]:     return self.module(*inputs, **kwargs)  # type: ignore[index]
[rank1]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank1]:     return self._call_impl(*args, **kwargs)
[rank1]:   File "/home/rry/anaconda3/envs/yolo11/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank1]:     return forward_call(*args, **kwargs)
[rank1]:   File "/home/rry/RMArmorDetect/ultralytics/nn/tasks.py", line 136, in forward
[rank1]:     return self.loss(x, *args, **kwargs)
[rank1]:   File "/home/rry/RMArmorDetect/ultralytics/nn/tasks.py", line 328, in loss
[rank1]:     return self.criterion(preds, batch)
[rank1]:   File "/home/rry/RMArmorDetect/armor_detect/losses/armor_pose_loss.py", line 238, in __call__
[rank1]:     num_class_targets[b, valid_anchor_indices, cls_ids] = oks_soft
[rank1]: RuntimeError: Index put requires the source and destination dtypes match, got Half for the destination and Float for the source. 

