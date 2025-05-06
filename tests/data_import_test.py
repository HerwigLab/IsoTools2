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
