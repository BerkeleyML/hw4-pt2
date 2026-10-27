from otter.test_files import test_case

OK_FORMAT = False

name = "q4aii"
points = 1

@test_case(points=None, hidden=False)
def test_q4aii_strings(chimpanzee_dna, dog_dna, human_dna):
    assert isinstance(chimpanzee_dna, str) and len(chimpanzee_dna) >= 50
    assert isinstance(dog_dna, str) and len(dog_dna) >= 50
    assert isinstance(human_dna, str) and len(human_dna) >= 50

