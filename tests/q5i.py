from otter.test_files import test_case

OK_FORMAT = False

name = "q5i"
points = 2

@test_case(points=None, hidden=False)
def test_q5i_prediction(predicted_class, true_class, class_id_to_sound):
    assert predicted_class in class_id_to_sound.values()
    assert true_class in class_id_to_sound.values()

