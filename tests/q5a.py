from otter.test_files import test_case

OK_FORMAT = False

name = "q5a"
points = 3

@test_case(points=None, hidden=False)
def test_q5a_len(SpectrogramDataset):
    import os, tempfile, wave, struct
    td = tempfile.mkdtemp()
    path = os.path.join(td, '1-3-0-0.wav')
    with wave.open(path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b'\x00\x00' * 1600)
    ds = SpectrogramDataset(td, transforms=None, return_targets=True)
    assert len(ds) == 1
    item = ds[0]
    assert item is not None
    spec, target = item
    assert int(target) == 3, f'1-3-0-0.wav should have class id 3, got {target}'
    assert spec.ndim == 3 and spec.shape[0] == 3, f'spectrogram should have shape (3, H, W), got {tuple(spec.shape)}'
    ds_unlabeled = SpectrogramDataset(td, transforms=None, return_targets=False)
    unlabeled = ds_unlabeled[0]
    assert unlabeled is not None
    import torch
    assert isinstance(unlabeled, torch.Tensor), f'When return_targets is False, __getitem__ should return only the spectrogram tensor, not a (spectrogram, label) pair. Got {type(unlabeled)}.'

