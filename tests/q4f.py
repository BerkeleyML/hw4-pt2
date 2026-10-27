from otter.test_files import test_case

OK_FORMAT = False

name = "q4f"
points = 3

@test_case(points=None, hidden=False)
def test_q4f_smoke(train_dna_classifier, DNAClassifier):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    class Tiny(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

        def forward(self, input_ids, attention_mask):
            return self.linear(input_ids.float())

    class DS(torch.utils.data.Dataset):

        def __len__(self):
            return 2

        def __getitem__(self, i):
            return {'input_ids': torch.ones(4, dtype=torch.long), 'attention_mask': torch.ones(4, dtype=torch.long), 'targets': torch.tensor(i % 3, dtype=torch.long)}
    loader = DataLoader(DS(), batch_size=2)
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    old_grading = train_dna_classifier.__globals__.get('IS_GRADING_ENV', False)
    train_dna_classifier.__globals__['IS_GRADING_ENV'] = False
    try:
        histories = train_dna_classifier(model, optimizer, criterion, 'cpu', 1, loader, loader)
    finally:
        train_dna_classifier.__globals__['IS_GRADING_ENV'] = old_grading
    assert isinstance(histories, tuple) and len(histories) == 4, 'Expected four metric histories: train loss, val loss, train accuracy, and val accuracy.'
    assert all((len(history) == 1 for history in histories)), 'Expected each metric history to contain one value after one epoch.'

