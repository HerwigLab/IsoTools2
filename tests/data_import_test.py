import pytest
from pysam import FastaFile
from isotools.transcriptome import Transcriptome
from isotools._utils import splice_identical
import logging

logger = logging.getLogger("isotools")
logger.setLevel(logging.INFO)


@pytest.mark.dependency()
def test_import_gff():
    isoseq = Transcriptome.from_reference("tests/data/example.gff.gz")
    assert len(isoseq) == 65, "we expect 65 genes"
    isoseq.save_reference("tests/data/example_ref_isotools.pkl")
    assert True


def test_import_gff_chromosome_alias():
    # regression test for #36: RefSeq-style GFF3 files use an accession as
    # seqid (e.g. NC_000001.11) with the plain chromosome name (e.g. "1")
    # given via a "chromosome" attribute on a "region" feature line. Genes
    # were previously silently dropped when filtering against the plain
    # name, since the tabix-based alias resolution was lost when the
    # reader was rewritten to fix #28.
    transcriptome = Transcriptome.from_reference(
        "tests/data/refseq_style_chrom_alias.gff3", chromosomes={"1"}
    )
    assert len(transcriptome) == 1, "gene should be found via chromosome alias"
    gene = next(iter(transcriptome))
    assert gene.id == "GENE1"
    assert gene.chrom == "1", "gene should be indexed under the aliased chromosome name"


def test_add_sample_from_csv_missing_gene_info():
    # regression test for #25: a coverage csv row referencing a transcript_id
    # not found in the transcripts file previously broke gene_id/chr column
    # construction for *every* row, not just the unmatched one (an all-or-
    # nothing list comprehension inside a try/except). PB.999.1 below is not
    # in the gtf; the two real transcripts must still import successfully.
    isoseq = Transcriptome.from_reference("tests/data/example.gff.gz")
    id_map = isoseq.add_sample_from_csv(
        "tests/data/no_gene_lines_example_coverage.csv",
        transcripts_file="tests/data/no_gene_lines_example.gtf",
        transcript_id_col="transcript_id",
        reconstruct_genes=False,
        infer_genes=True,
        sep=",",
    )
    assert isoseq.n_transcripts == 2, "the two known transcripts should still import"
    assert set(id_map) == {"PB.7", "PB.1"}


def test_read_gff_progress_bar_plain_and_gzip():
    # regression test for #37: the progress bar's byte-position tracking
    # must work for both plain and gzip files. It previously crashed for
    # plain files only ("OSError: telling position disabled by next() call"),
    # since the gzip branch bypasses the disabled TextIOWrapper.tell().
    Transcriptome.from_reference(
        "tests/data/infer_genes_example.gff3", progress_bar=True, infer_genes=True
    )
    Transcriptome.from_reference("tests/data/example.gff.gz", progress_bar=True)


def test_import_gff_infer_genes():
    # infer_genes_example.gff3 has no "gene" lines, only "transcript"/"exon",
    # with two transcripts (GENE1.1, GENE1.2) sharing Parent=GENE1 -- the
    # gene must be inferred from the transcript lines, not the exon lines
    # (whose Parent is the transcript id, not the gene id)
    isoseq = Transcriptome.from_reference(
        "tests/data/infer_genes_example.gff3", infer_genes=True
    )
    assert len(isoseq) == 1, "expected exactly one inferred gene"
    gene = next(iter(isoseq))
    assert gene.id == "GENE1"
    assert gene.name == "TESTGENE"
    assert gene.chrom == "chr2_part"
    assert gene.strand == "+"
    assert gene.n_ref_transcripts == 2, "expected both transcripts on the gene"
    assert (gene.start, gene.end) == (
        99,
        600,
    ), "gene span should cover both transcripts"


@pytest.mark.dependency(depends=["test_import_gff"])
def test_import_bam():
    isoseq = Transcriptome.from_reference("tests/data/example_ref_isotools.pkl")
    assert isoseq.n_transcripts == 0, "there should not be any transcripts"
    for sample in ("CTL", "VPA"):
        isoseq.add_sample_from_bam(
            f"tests/data/example_1_{sample}.bam",
            sample_name=sample,
            group=sample,
            platform="SequelII",
        )
    # assert isoseq.n_transcripts == 185, 'we expect 185 transcripts'
    isoseq.add_qc_metrics("tests/data/example.fa")
    isoseq.add_orf_prediction("tests/data/example.fa")
    isoseq.save("tests/data/example_1_isotools.pkl")


@pytest.mark.dependency(depends=["test_import_bam"])
def test_cpm():
    # regression test for the tpm -> cpm rename: values are counts per
    # million (count / total_reads * 1e6), not length-normalized TPM --
    # correct for full-length long reads, where read count already is a
    # direct proxy for molecule count.
    isoseq = Transcriptome.load("tests/data/example_1_isotools.pkl")
    tab = isoseq.transcript_table(coverage=True, cpm=True)
    stab = isoseq.sample_table.set_index("name")
    for sample in isoseq.samples:
        cov_col, cpm_col = f"{sample}_coverage", f"{sample}_cpm"
        assert cov_col in tab.columns and cpm_col in tab.columns
        total = stab.loc[sample, "nonchimeric_reads"]
        expected = tab[cov_col] / total * 1e6
        assert (
            tab[cpm_col] - expected
        ).abs().max() < 1e-6, "cpm should be count/total_reads*1e6"

    gene = next(iter(isoseq.iter_genes(query="EXPRESSED")))
    gene_cpm = gene.cpm()
    assert gene_cpm.shape == (len(isoseq.samples), gene.n_transcripts)


@pytest.mark.dependency(depends=["test_import_bam"])
def test_fsm():
    isoseq = Transcriptome.load("tests/data/example_1_isotools.pkl")
    count = 0
    for gene, _, transcript in isoseq.iter_transcripts(query="FSM"):
        assert transcript["annotation"][0] == 0
        count += 1
        for ref_id in transcript["annotation"][1]["FSM"]:
            assert splice_identical(
                transcript["exons"], gene.ref_transcripts[ref_id]["exons"]
            )
    assert count == 22, "expected 22 FSM transcripts"


@pytest.mark.dependency(depends=["test_import_bam"])
def test_import_csv_reconstruct():  # reconstruct gene structure from scratch
    isoseq = Transcriptome.load("tests/data/example_1_isotools.pkl")
    cov_tab = isoseq.transcript_table(coverage=True)
    cov_tab.to_csv("tests/data/example_1_cov.csv")
    isoseq.write_gtf("tests/data/example_1.gtf")
    isoseq_csv = Transcriptome.from_reference("tests/data/example_ref_isotools.pkl")
    isoseq_csv._add_novel_gene(
        "nix", 10, 20, "-", {"exons": [10, 20]}
    )  # additional gene should not confuse/break the function
    id_map = isoseq_csv.add_sample_from_csv(
        "tests/data/example_1_cov.csv",
        "tests/data/example_1.gtf",
        reconstruct_genes=True,
        sample_properties=isoseq.sample_table,
        sep=",",
    )
    remapped_genes = {
        gid: gid2 for gid2, id_dict in id_map.items() for gid in id_dict.values()
    }
    logger.info("remapped %s transcripts", sum(len(d) for d in id_map))
    assert set(isoseq.samples) == set(
        isoseq_csv.samples
    ), "discrepant samples after csv import"
    stab1, stab2 = isoseq.sample_table.set_index(
        "name"
    ), isoseq_csv.sample_table.set_index("name")
    for sample in isoseq.samples:
        assert stab1.loc[sample, "group"] == stab2.loc[sample, "group"], (
            "wrong group after csv import for sample %s" % sample
        )
        assert (
            stab1.loc[sample, "nonchimeric_reads"]
            == stab2.loc[sample, "nonchimeric_reads"]
        ), ("wrong number of reads after csv import for sample %s" % sample)

    discrepancy = False
    for gene in isoseq.iter_genes(query="EXPRESSED"):
        if (gene.is_annotated and gene.id in remapped_genes) or (
            gene.id not in isoseq_csv and gene.id not in remapped_genes
        ):
            logger.error("gene missing/renamed after csv import: %s" % str(gene))
            discrepancy = True
    for gene_csv in isoseq_csv.iter_genes(query="EXPRESSED"):
        if not gene_csv.is_annotated and gene_csv.id in remapped_genes:
            gene_id = remapped_genes[gene_csv.id]
        else:
            gene_id = gene_csv.id
        gene = isoseq[gene_id]
        if len(gene.transcripts) != len(gene_csv.transcripts):
            logger.error(
                "number of transcripts for %s changed after csv import: %s != %s",
                gene.id,
                len(gene.transcripts),
                len(gene_csv.transcripts),
            )
            discrepancy = True
    assert not discrepancy, "discrepancy found after csv import"


@pytest.mark.dependency(depends=["test_import_bam"])
def test_import_csv():  # use gene structure from gtf
    isoseq = Transcriptome.load("tests/data/example_1_isotools.pkl")
    cov_tab = isoseq.transcript_table(coverage=True)
    cov_tab.to_csv("tests/data/example_1_cov.csv")
    isoseq.write_gtf("tests/data/example_1.gtf")
    isoseq_csv = Transcriptome.from_reference("tests/data/example_ref_isotools.pkl")
    isoseq_csv._add_novel_gene(
        "nix", 10, 20, "-", {"exons": [10, 20]}
    )  # make it a little harder
    id_map = isoseq_csv.add_sample_from_csv(
        "tests/data/example_1_cov.csv",
        "tests/data/example_1.gtf",
        reconstruct_genes=True,
        sample_properties=isoseq.sample_table,
        sep=",",
    )
    remapped_genes = {gid: k for k, v in id_map.items() for gid in v.values()}
    logger.info("remapped %s genes", len(id_map))
    assert set(isoseq.samples) == set(
        isoseq_csv.samples
    ), "discrepant samples after csv import"
    stab1, stab2 = isoseq.sample_table.set_index(
        "name"
    ), isoseq_csv.sample_table.set_index("name")
    for sample in isoseq.samples:
        assert stab1.loc[sample, "group"] == stab2.loc[sample, "group"], (
            "wrong group after csv import for sample %s" % sample
        )
        assert (
            stab1.loc[sample, "nonchimeric_reads"]
            == stab2.loc[sample, "nonchimeric_reads"]
        ), ("wrong number of reads after csv import for sample %s" % sample)
    discrepancy = False
    for gene in isoseq.iter_genes(query="EXPRESSED"):
        if (gene.is_annotated and gene.id in remapped_genes) or (
            gene.id not in isoseq_csv and gene.id not in remapped_genes
        ):
            logger.error("gene missing/renamed after csv import: %s" % str(gene))
            discrepancy = True
    for gene_csv in isoseq_csv.iter_genes(query="EXPRESSED"):
        if not gene_csv.is_annotated and gene_csv.id in remapped_genes:
            gene_id = remapped_genes[gene_csv.id]
        else:
            gene_id = gene_csv.id
        gene = isoseq[gene_id]
        if len(gene.transcripts) != len(gene_csv.transcripts):
            logger.error(
                "number of transcripts for %s changed after csv import: %s != %s",
                gene.id,
                len(gene.transcripts),
                len(gene_csv.transcripts),
            )
            discrepancy = True
    assert not discrepancy, "discrepancy found after csv import"


@pytest.mark.dependency(depends=["test_import_gff"])
def test_orf():
    total, same = {"+": 0, "-": 0}, {"+": 0, "-": 0}
    isoseq = Transcriptome.from_reference("tests/data/example_ref_isotools.pkl")
    with FastaFile("tests/data/example.fa") as genome_fh:
        for gene in isoseq:
            gene.add_orfs(genome_fh=genome_fh, reference=True)
            for transcript in gene.ref_transcripts:
                if (
                    transcript["transcript_type"] == "protein_coding"
                    and "CDS" in transcript
                ):
                    total[gene.strand] += 1
                    if transcript["CDS"] == transcript["ORF"][:2]:
                        same[gene.strand] += 1
    assert (
        same["+"] / total["+"] > 0.9
    ), "at least 90% protein coding transcripts CDS on + should match longest ORF."
    assert (
        same["-"] / total["-"] > 0.9
    ), "at least 90% protein coding transcripts CDS on - should match longest ORF."
