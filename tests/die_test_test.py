import numpy as np
from isotools import Gene


def test_die_test_ignores_isoforms_unexpressed_in_both_groups():
    # regression test for #29: a gene's isoform list is the union of isoforms
    # seen across *all* samples in the Transcriptome, not just the two groups
    # being compared. An isoform only covered in a sample outside both groups
    # ends up as an all-zero row here, which made chi2_contingency raise
    # ValueError("expected frequencies has a zero element") since a zero row
    # makes the expected-frequency table singular. min_cov (checked on group
    # totals) does not catch this, since the other isoforms still carry
    # enough coverage.
    coverage = np.array(
        [
            [706, 3, 0, 2, 0, 0, 0, 0],  # group 0 samples
            [1218, 0, 0, 0, 1, 0, 0, 0],  # group 1 samples
        ]
    )
    gene = Gene(
        100,
        500,
        {"chr": "chr1", "strand": "+", "ID": "GENE1", "coverage": coverage},
        None,
    )
    pval, deltaPI, transcript_ids = gene.die_test([[0], [1]], min_cov=1)
    assert np.isfinite(pval)
    assert np.isfinite(deltaPI)
    # the zero-coverage isoforms (original indices 2, 5, 6, 7) must not appear
    assert set(transcript_ids).issubset({0, 1, 3, 4})
