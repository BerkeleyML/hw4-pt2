from otter.test_files import test_case

OK_FORMAT = False

name = "q4ai"
points = 1

@test_case(points=None, hidden=False)
def test_q4ai_balanced(combined_df):
    assert len(combined_df) == 3 * 656, f'Length of the combined dataframe is not {3 * 656}'
    counts = combined_df['target'].value_counts()
    assert set(counts.index) == {0, 1, 2}, f'target should contain exactly {{0, 1, 2}}, got {sorted(counts.index)}'
    assert (counts == 656).all(), f'Each target class should have 656 rows, got {counts.to_dict()}'
    assert 'gene_family' in combined_df.columns, 'rename the `class` column to `gene_family`'
    species = set(combined_df['species'])
    assert species == {'chimpanzee', 'dog', 'human'}, f"species should be the strings {{'chimpanzee', 'dog', 'human'}} after you add that column back, got {species}"

