import torch
import sys
sys.path.append("src")
from eeg_dataset_utils import EEG_dataset

def test_eeg_dataset_Mar25(path="../Mar25/"):
    eeg_dataset = EEG_dataset(path)
    sample = eeg_dataset.__getitem__(0)
    assert len(sample) == 3
    assert type(sample[0]) == torch.Tensor
    assert type(sample[1]) == torch.Tensor
    assert type(sample[2]) == torch.Tensor
    assert sample[0].shape == torch.Size([600, 22])
    assert sample[1].shape == torch.Size([600, 22])
    assert sample[2].shape == torch.Size([1])