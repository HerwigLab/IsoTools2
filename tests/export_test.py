from pysam import FastaFile
from isotools.transcriptome import Transcriptome


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


def test_write_fasta_add_coord(tmp_path):
    isoseq = _example_transcriptome()
    with FastaFile("tests/data/example.fa") as genome_fh:
        for gene in isoseq:
            gene.add_orfs(genome_fh, reference=True)
            gene.add_orfs(genome_fh, reference=False)

    gene = isoseq["FN1"]
    ref_transcript = gene.ref_transcripts[0]
    assert (
        "CDS" in ref_transcript
    ), "expected FN1 reference transcript 0 to have an annotated CDS"
    tr_start, tr_end = ref_transcript["exons"][0][0], ref_transcript["exons"][-1][1]
    cds_start, cds_end = ref_transcript["CDS"]

    tr_fn = tmp_path / "tr.fa"
    prot_fn = tmp_path / "prot.fa"
    isoseq.write_fasta(
        "tests/data/example.fa",
        str(tr_fn),
        reference=True,
        protein=False,
        add_coord=True,
        gois=["FN1"],
    )
    isoseq.write_fasta(
        "tests/data/example.fa",
        str(prot_fn),
        reference=True,
        protein=True,
        add_coord=True,
        gois=["FN1"],
    )

    tr_header = tr_fn.read_text().splitlines()[0]
    prot_header = prot_fn.read_text().splitlines()[0]

    assert (
        tr_header
        == f">{gene.id}_0 {gene.chrom}:{tr_start}-{tr_end}:{gene.strand} gene={gene.name}"
    )
    assert (
        prot_header
        == f">{gene.id}_0 {gene.chrom}:{cds_start}-{cds_end}:{gene.strand} gene={gene.name}"
    )


def test_write_fasta_no_coord_by_default(tmp_path):
    # regression test: add_coord defaults to False, so the header must be
    # unchanged from before add_coord was introduced.
    isoseq = _example_transcriptome()
    fn = tmp_path / "tr.fa"
    isoseq.write_fasta("tests/data/example.fa", str(fn), reference=True, gois=["FN1"])
    header = fn.read_text().splitlines()[0]
    gene = isoseq["FN1"]
    assert header == f">{gene.id}_0 gene={gene.name}"
