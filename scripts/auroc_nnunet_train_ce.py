# nnU-Net v2 training with a cross-entropy-only trainer of 250 epochs (nnUNetTrainerCELoss_250epochs): dice loss
# saturates the softmax (nearly every case gets max p ~ 1), so patient scores need CE-calibrated probabilities, as
# in the PI-CAI nnU-Net baseline. Includes the /dev/shm workaround of /data/sota/nnunet/bin/train.py.
import multiprocessing.reduction as red

import numpy as np
import torch
import torch.multiprocessing  # noqa: F401


def _rebuild(buf, dtype, shape):
    return torch.from_numpy(np.frombuffer(buf, dtype=dtype).reshape(shape).copy())


def _reduce(t):
    a = t.detach().cpu().contiguous().numpy()
    return _rebuild, (a.tobytes(), a.dtype.str, a.shape)


for cls in (torch.Tensor, torch.nn.parameter.Parameter):
    red.ForkingPickler.register(cls, _reduce)

import nnunetv2.run.run_training as rt  # noqa: E402
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerCELoss import nnUNetTrainerCELoss  # noqa: E402


class nnUNetTrainerCELoss_250epochs(nnUNetTrainerCELoss):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


_find = rt.recursive_find_trainer_class_by_name
rt.recursive_find_trainer_class_by_name = lambda name: nnUNetTrainerCELoss_250epochs if name == "nnUNetTrainerCELoss_250epochs" else _find(name)

if __name__ == "__main__":
    rt.run_training_entry()
