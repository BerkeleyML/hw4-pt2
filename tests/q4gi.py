from otter.test_files import test_case

OK_FORMAT = False

name = "q4gi"
points = 1

@test_case(points=None, hidden=False)
def test_q4gi_setup(dna_train_losses, dna_val_losses, dna_train_accuracies, dna_val_accuracies):
    histories = [dna_train_losses, dna_val_losses, dna_train_accuracies, dna_val_accuracies]
    assert all((isinstance(history, list) and len(history) == 5 for history in histories))

