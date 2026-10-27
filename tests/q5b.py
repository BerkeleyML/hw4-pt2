from otter.test_files import test_case

OK_FORMAT = False

name = "q5b"
points = 2

@test_case(points=None, hidden=False)
def test_q5b_loaders(spectrogram_train_dataloader, spectrogram_val_dataloader, resize_transform):
    assert spectrogram_train_dataloader.batch_size == 32
    assert spectrogram_val_dataloader.batch_size == 32
    size = getattr(resize_transform, 'size', None)
    assert size in ([224, 224], (224, 224), 224) or list(size) == [224, 224]

