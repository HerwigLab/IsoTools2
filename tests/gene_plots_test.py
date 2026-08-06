import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from isotools import Gene, SQANTI_PALETTE  # noqa: E402


def test_gene_track_colorbysqanti_uses_shared_palette():
    # regression test: SQANTI_PALETTE was extracted from a local dict inside
    # gene_track into a shared, importable module-level constant. Confirm
    # gene_track still colors transcripts exactly as before the extraction.
    transcripts = [
        {"exons": [(10, 30)], "transcript_name": "tr1", "annotation": (0, {})}
    ]
    gene = Gene(
        10,
        30,
        {"chr": "chr1", "strand": "+", "ID": "g1", "transcripts": transcripts},
        None,
    )
    fig, ax = plt.subplots()
    gene.gene_track(ax=ax, reference=False, colorbySqanti=True, label_transcripts=False)
    line_color = ax.get_lines()[0].get_color()
    assert to_rgba(line_color) == to_rgba(SQANTI_PALETTE[0]["color"])
    plt.close(fig)
