# Testing conventions

Two different test styles exist in this directory:

1. **Self-contained tests (preferred for new tests).** Build the minimal input
   directly -- either in memory via a `conftest.py` fixture (e.g.
   `example_gene`, used by `altsplice_test.py`/`coordination_test.py`), or as
   a small, purpose-built, git-tracked fixture file under `tests/data/` when
   file I/O is inherently part of what's being tested (e.g.
   `lowercase_orf_example.fa`/`.gff3`, `infer_genes_example.gff3`). No shared
   mutable state, no dependency on other tests having run first, no need to
   clean anything between runs.

2. **The pipeline chain** (`data_import_test.py`, `splice_graph_test.py`,
   `diffsplice_test.py`, `domain_test.py`). A sequence of tests that must run
   *in order*, each writing a pickle (`tests/data/example_1_isotools.pkl`,
   `example_ref_isotools.pkl`) that later tests load and assume is fresh.
   Ordering is nominally enforced by `pytest.mark.dependency`, but that's a
   plugin, not a core dependency -- a plain `pytest` run without it installed
   silently ignores the markers. The generated files are gitignored, so stale
   state from a previous run or a different branch/checkout can silently
   produce misleading pass/fail results; run `git clean -fdx tests/data/`
   before trusting a result from this chain if anything seems off.
   This does exercise the real end-to-end BAM-import -> reconstruction ->
   stats workflow realistically, so it has real value and isn't being
   removed -- just don't extend it for new, narrowly-scoped tests.

**When adding a test for a specific bug or feature, use style 1.** Reserve
the pipeline chain for genuine end-to-end integration concerns.

## File organization

One test file per feature area, named after what it covers -- not
necessarily 1:1 with a source file, but close (`domain_test.py` for
`domains.py`, `splice_graph_test.py` for `splice_graph.py`, `orf_test.py`
for ORF-prediction logic in `gene.py`, `data_import_test.py` for import
functions in `_transcriptome_io.py`, `utils_test.py` for the general
primitives in `_utils.py`, etc.). Before creating a new file, check whether
an existing one already covers that area.
