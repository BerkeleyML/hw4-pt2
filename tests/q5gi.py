from otter.test_files import test_case

OK_FORMAT = False

name = "q5gi"
points = 1

@test_case(points=None, hidden=False)
def test_q5gi_setup(unfrozen_convnext, unfrozen_train_losses, unfrozen_val_losses, unfrozen_train_accuracies, unfrozen_val_accuracies):
    assert all((p.requires_grad for p in unfrozen_convnext.parameters()))
    histories = [unfrozen_train_losses, unfrozen_val_losses, unfrozen_train_accuracies, unfrozen_val_accuracies]
    assert all((isinstance(history, list) and len(history) == 5 for history in histories))

