from otter.test_files import test_case

OK_FORMAT = False

name = "q4b"
points = 1

@test_case(points=None, hidden=False)
def test_q4b_kmers(sequence_to_kmer, combined_df):
    got = sequence_to_kmer('ATGCGT', k=3)
    assert got == 'ATG TGC GCG CGT'
    assert 'kmers' in combined_df.columns, "combined_df is missing the 'kmers' column"
    row = combined_df.iloc[0]
    assert row['kmers'] == sequence_to_kmer(row['sequence'], k=6), "combined_df['kmers'] should be sequence_to_kmer(sequence, k=6)"

