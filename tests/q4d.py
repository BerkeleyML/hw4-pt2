from otter.test_files import test_case

OK_FORMAT = False

name = "q4d"
points = 2

@test_case(points=None, hidden=False)
def test_q4d_len(DNADataset):

    class Tok:

        def __call__(self, text, return_tensors='pt', max_length=512, truncation=True, padding=None):
            import torch
            n = len(text.split())
            if truncation:
                n = min(n, max_length)
            if padding == 'max_length':
                n = max_length
            return {'input_ids': torch.ones(1, n), 'attention_mask': torch.ones(1, n)}
    short = 'AAAAAA'
    long = ' '.join(['AAAAAA'] * 600)
    ds = DNADataset([short, long], targets=[0, 1], tokenizer=Tok(), return_targets=True)
    assert len(ds) == 2, f'Expected a dataset with 2 samples, but len(ds) returned {len(ds)}.'
    for item in (ds[0], ds[1]):
        required_keys = {'input_ids', 'attention_mask', 'targets'}
        missing_keys = required_keys - set(item)
        assert not missing_keys, f'DNADataset.__getitem__ is missing required key(s): {sorted(missing_keys)}.'
        assert tuple(item['input_ids'].shape) == (512,), f"Expected input_ids shape (512,). Tokenize with max_length=512 and padding='max_length', then drop the batch dimension. Got {tuple(item['input_ids'].shape)}."
        assert tuple(item['attention_mask'].shape) == (512,), f"Expected attention_mask shape (512,) after squeezing the tokenizer batch dimension, but got {tuple(item['attention_mask'].shape)}. The mask should be 1D like input_ids, not a square matrix."

