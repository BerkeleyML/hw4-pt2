from otter.test_files import test_case

OK_FORMAT = False

name = "q4e"
points = 2

@test_case(points=None, hidden=False)
def test_q4e_split(train_set, val_set, train_dataloader, val_dataloader):
    from torch.utils.data import RandomSampler, SequentialSampler
    assert len(train_set) == 800, f'Expected 800 training samples (80% of 1000), but got {len(train_set)}.'
    assert len(val_set) == 200, f'Expected 200 validation samples (20% of 1000), but got {len(val_set)}.'
    assert train_dataloader.batch_size == 32, f'Expected train_dataloader.batch_size == 32, but got {train_dataloader.batch_size}.'
    assert val_dataloader.batch_size == 32, f'Expected val_dataloader.batch_size == 32, but got {val_dataloader.batch_size}.'
    assert isinstance(train_dataloader.sampler, RandomSampler), 'Expected the training DataLoader to use shuffle=True.'
    assert isinstance(val_dataloader.sampler, SequentialSampler), 'Expected the validation DataLoader to use shuffle=False.'

