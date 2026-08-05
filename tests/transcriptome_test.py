import logging
from intervaltree import IntervalTree
from isotools import Gene, Transcriptome


def test_make_index_warns_on_ambiguous_name(caplog):
    # regression test for #27: only gene *id* collisions were checked in
    # make_index(); gene *name* collisions (common in real annotation --
    # paralogs, duplicated symbols) were completely silent, with the
    # second gene silently overwriting the first in the name->gene index.
    gene1 = Gene(
        0, 100, {"chr": "chr1", "strand": "+", "ID": "GENE1", "name": "DUP"}, None
    )
    gene2 = Gene(
        200, 300, {"chr": "chr1", "strand": "+", "ID": "GENE2", "name": "DUP"}, None
    )

    with caplog.at_level(logging.WARNING, logger="isotools"):
        Transcriptome(
            data={"chr1": IntervalTree([gene1, gene2])},
            infos={"reference_file": "test"},
        )

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "DUP" in m and "ambiguous" in m for m in messages
    ), "expected a warning about the ambiguous gene name"

    # id collisions must still be reported, and must not be confused with
    # name collisions (they are tracked separately)
    gene3 = Gene(
        400, 500, {"chr": "chr1", "strand": "+", "ID": "GENE1", "name": "OTHER"}, None
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="isotools"):
        Transcriptome(
            data={"chr1": IntervalTree([gene1, gene3])},
            infos={"reference_file": "test"},
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "GENE1" in m and "ambiguous" in m for m in messages
    ), "expected a warning about the ambiguous gene id"
