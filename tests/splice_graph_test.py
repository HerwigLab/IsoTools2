import numpy as np
import pytest
from isotools.transcriptome import Transcriptome
from isotools.splice_graph import SegmentGraph
from isotools._utils import (
    _find_splice_sites,
    _get_overlap,
    _get_exonic_region,
    pairwise,
)

# @pytest.mark.dependency(depends=['test_import_bam'])


def test_find_splice_bubbles_numpy_coordinates():
    # regression test for #49: exon coordinates derived from real alignments
    # are numpy int64, not plain python int. Comparing two numpy.int64 values
    # gives numpy.bool_, which was previously used directly as a tuple/list
    # index in find_splice_bubbles -- Python only accepts bool/int there, so
    # this crashed with "TypeError: tuple indices must be integers or slices,
    # not numpy.bool" for every real gene with an exon-skipping-type event.
    exons = [
        [
            (np.int64(0), np.int64(10)),
            (np.int64(20), np.int64(30)),
            (np.int64(40), np.int64(50)),
        ],
        [(np.int64(0), np.int64(10)), (np.int64(40), np.int64(50))],
    ]
    sg = SegmentGraph(exons, "+")
    bubbles = list(sg.find_splice_bubbles(types=["ES"]))
    assert isinstance(bubbles, list)


def test_import_find_splice_site():
    isoseq = Transcriptome.load("tests/data/example_1_isotools.pkl")
    for gene, _, transcript in isoseq.iter_transcripts(query="not NOVEL_GENE"):
        sj = [(exon1[1], exon2[0]) for exon1, exon2 in pairwise(transcript["exons"])]
        c1 = gene.ref_segment_graph.find_splice_sites(sj)
        c2 = _find_splice_sites(sj, gene.ref_transcripts)
        assert all(
            c1 == c2
        ), "isotools._transcriptome_io._find_splice_sites and Segment_Graph.find_splice_sites yield different results"


@pytest.mark.dependency()
def test_exon_regions():
    isoseq = Transcriptome.load("tests/data/example_1_isotools.pkl")
    for gene in isoseq.iter_genes(query="not NOVEL_GENE"):
        c1 = gene.ref_segment_graph.get_exonic_region()
        c2 = _get_exonic_region(gene.ref_transcripts)
        assert len(c1) == len(
            c2
        ), "isotools._transcriptome_io._get_exonic_region and Segment_Graph.get_exonic_region yield different length"
        assert all(
            reg1[0] == reg2[0] and reg1[1] == reg2[1] for reg1, reg2 in zip(c1, c2)
        ), "isotools._transcriptome_io._get_exonic_region and Segment_Graph.get_exonic_region yield different regions"
    assert True


@pytest.mark.dependency(depends=["test_exon_regions"])
def test_import_exonic_overlap():
    isoseq = Transcriptome.load("tests/data/example_1_isotools.pkl")
    for gene, _, transcript in isoseq.iter_transcripts(query="not NOVEL_GENE"):
        c1 = gene.ref_segment_graph.get_overlap(transcript["exons"])[0]
        c2 = _get_overlap(transcript["exons"], gene.ref_transcripts)
        assert (
            c1 == c2
        ), "isotools._transcriptome_io._get_overlap and Segment_Graph.get_overlap yield different results"
    assert True
