import pyhmmer
from isotools import Transcriptome
from isotools.domains import add_domains_to_table, get_hmmer_sequences


def _example_transcriptome():
    isoseq = Transcriptome.from_reference("tests/data/example.gff.gz")
    for sa in ("CTL", "VPA"):
        isoseq.add_sample_from_bam(
            f"tests/data/example_1_{sa}.bam",
            sample_name=sa,
            group=sa,
            platform="SequelII",
        )
    return isoseq


def test_get_hmmer_sequences_returns_digital_sequence_block():
    # regression test for #53: get_hmmer_sequences returned a plain list, but
    # pyhmmer's Pipeline.search_hmm has required a DigitalSequenceBlock since
    # pyhmmer 0.7.0, so any add_hmmer_domains() call crashed with a TypeError.
    isoseq = _example_transcriptome()
    alphabet = pyhmmer.easel.Alphabet.amino()
    sequences, seq_ids = get_hmmer_sequences(
        isoseq, "tests/data/example.fa", alphabet, query=True, ref_query=True
    )
    assert isinstance(sequences, pyhmmer.easel.DigitalSequenceBlock)
    assert len(sequences) == len(seq_ids)

    pipeline = pyhmmer.plan7.Pipeline(alphabet)
    builder = pyhmmer.plan7.Builder(alphabet)
    background = pyhmmer.plan7.Background(alphabet)
    hmm, _, _ = builder.build(sequences[0], background)
    hits = pipeline.search_hmm(hmm, sequences)  # would raise TypeError before the fix
    assert len(hits) >= 1


def test_get_hmmer_sequences_query_true_includes_all_transcripts():
    # regression test for #53: query=True / ref_query=True (the documented
    # "include all transcripts" value, and the default for both parameters)
    # was passed straight through as a filter expression, crashing with
    # AssertionError: expression should be a string.
    isoseq = _example_transcriptome()
    alphabet = pyhmmer.easel.Alphabet.amino()
    sequences, seq_ids = get_hmmer_sequences(
        isoseq, "tests/data/example.fa", alphabet, query=True, ref_query=True
    )
    assert len(sequences) > 0


def test_anno_domains():
    isoseq = Transcriptome.load("tests/data/example_1_isotools.pkl")
    isoseq.add_annotation_domains(
        "tests/data/example_anno_domains.csv", category="domains", progress_bar=False
    )
    gene = isoseq["FN1"]
    ref_tr = next(
        transcript
        for transcript in gene.ref_transcripts
        if transcript["transcript_name"] == "FN1-207"
    )
    dom = ref_tr["domain"]["annotation"]
    assert (
        len(dom) == 25
    ), "expected 25 annotation domains in FN1-207, but found {len(dom)}"

    diffexpr = isoseq.altsplice_test(groups=isoseq.groups()).sort_values("pvalue")
    diffexpr = add_domains_to_table(diffexpr, isoseq, "annotation", insert_after="nmdB")
