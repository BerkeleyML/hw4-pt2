from otter.test_files import test_case

OK_FORMAT = False

name = "q5d"
points = 1

@test_case(points=None, hidden=False)
def test_q5d_replace(replace_final_convnext_linear_layer):
    import torch.nn as nn
    from torchvision.models import convnext_base
    model = convnext_base(weights=None)
    model = replace_final_convnext_linear_layer(model, num_classes=10)
    assert model.classifier[-1].out_features == 10

