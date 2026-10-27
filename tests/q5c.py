from otter.test_files import test_case

OK_FORMAT = False

name = "q5c"
points = 3

@test_case(points=None, hidden=False)
def test_q5c_smoke(train_image_classifier):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    x = torch.randn(4, 3)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    old_grading = train_image_classifier.__globals__.get('IS_GRADING_ENV', False)
    train_image_classifier.__globals__['IS_GRADING_ENV'] = False
    try:
        histories = train_image_classifier(model, optimizer, criterion, 'cpu', 1, loader, loader)
    finally:
        train_image_classifier.__globals__['IS_GRADING_ENV'] = old_grading
    assert isinstance(histories, tuple) and len(histories) == 4, 'Expected four metric histories: train loss, val loss, train accuracy, and val accuracy.'
    assert all((len(history) == 1 for history in histories)), 'Expected each metric history to contain one value after one epoch.'

