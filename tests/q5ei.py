from otter.test_files import test_case

OK_FORMAT = False

name = "q5ei"
points = 1

@test_case(points=None, hidden=False)
def test_q5ei_setup(from_scratch_train_losses, from_scratch_val_losses, from_scratch_train_accuracies, from_scratch_val_accuracies):
    histories = [from_scratch_train_losses, from_scratch_val_losses, from_scratch_train_accuracies, from_scratch_val_accuracies]
    assert all((isinstance(history, list) and len(history) == 5 for history in histories))

