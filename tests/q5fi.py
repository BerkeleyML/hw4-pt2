from otter.test_files import test_case

OK_FORMAT = False

name = "q5fi"
points = 1

@test_case(points=None, hidden=False)
def test_q5fi_setup(frozen_backbone, frozen_bb_train_losses, frozen_bb_val_losses, frozen_bb_train_accuracies, frozen_bb_val_accuracies):
    trainable = [n for n, p in frozen_backbone.named_parameters() if p.requires_grad]
    frozen = [n for n, p in frozen_backbone.named_parameters() if not p.requires_grad]
    assert any(('classifier' in n for n in trainable)), 'the classifier head should be trainable'
    extra = [n for n in trainable if 'classifier' not in n]
    assert not extra, f'only the classifier head should be trainable, but these parameters are too: {extra[:5]}'
    assert any(('classifier' not in n for n in frozen)), 'the backbone should be frozen'
    histories = [frozen_bb_train_losses, frozen_bb_val_losses, frozen_bb_train_accuracies, frozen_bb_val_accuracies]
    assert all((isinstance(history, list) and len(history) == 5 for history in histories))

