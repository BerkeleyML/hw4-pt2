from otter.test_files import test_case

OK_FORMAT = False

name = "q4c"
points = 2

@test_case(points=None, hidden=False)
def test_q4c_head(DNAClassifier):
    import torch.nn as nn
    m = DNAClassifier(num_classes=3)
    assert isinstance(m.classifier, nn.Linear)
    assert m.classifier.in_features == 768
    assert m.classifier.out_features == 3

