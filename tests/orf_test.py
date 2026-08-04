from isotools.transcriptome import Transcriptome
from pysam import FastaFile


def test_add_orfs_lowercase_fasta():
    # regression test for #26: genome FASTA files commonly soft-mask repeats
    # with lowercase bases. The start/stop codons here are uppercase (so
    # find_orfs locates the ORF regardless), but the surrounding sequence
    # used for the kozak score lookup is lowercase -- this crashed with an
    # uncaught KeyError before add_orfs uppercased the sequence.
    isoseq = Transcriptome.from_reference("tests/data/lowercase_orf_example.gff3")
    gene = isoseq["GENE1"]
    with FastaFile("tests/data/lowercase_orf_example.fa") as genome_fh:
        gene.add_orfs(genome_fh, reference=True, minlen=0, max_5utr_len=12)

    orf = gene.ref_transcripts[0].get("ORF")
    assert (
        orf is not None
    ), "expected an ORF to be found despite lowercase genome sequence"
    genome_start, genome_end, orf_info = orf
    assert (genome_start, genome_end) == (12, 33), "unexpected ORF coordinates"
    assert orf_info["start_codon"] == "ATG"
    assert orf_info["stop_codon"] == "TAA"
    assert isinstance(
        orf_info["kozak"], float
    ), "kozak score should compute without error"
