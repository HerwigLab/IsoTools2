from pysam import FastaFile
from tqdm import tqdm
import logging
import re
from ._utils import _filter_function, DEFAULT_KOZAK_PWM
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .transcriptome import Transcriptome
    from .gene import Gene

logger = logging.getLogger("isotools")
BOOL_OP = {"and", "or", "not", "is"}
DEFAULT_GENE_FILTER = {
    "NOVEL_GENE": "not reference",
    "EXPRESSED": "transcripts",
    "CHIMERIC": "chimeric",
}

DEFAULT_REF_TRANSCRIPT_FILTER = {
    "REF_UNSPLICED": "len(exons)==1",
    "REF_MULTIEXON": "len(exons)>1",
    "REF_INTERNAL_PRIMING": "downstream_A_content>.5",
}

DEFAULT_TRANSCRIPT_FILTER = {
    # 'CLIPPED_ALIGNMENT':'clipping',
    "INTERNAL_PRIMING": "len(exons)==1 and downstream_A_content and downstream_A_content>.5",  # more than 50% a
    "RTTS": "noncanonical_splicing is not None and novel_splice_sites is not None and \
        any(2*i in novel_splice_sites and 2*i+1 in novel_splice_sites for i,_ in noncanonical_splicing)",
    "NONCANONICAL_SPLICING": "noncanonical_splicing",
    "NOVEL_TRANSCRIPT": "annotation[0]>0",
    "FRAGMENT": 'fragments and any("novel exonic " in a or "fragment" in a for a in annotation[1])',
    "UNSPLICED": "len(exons)==1",
    "MULTIEXON": "len(exons)>1",
    "SUBSTANTIAL": "gene.coverage.sum() * .01 < gene.coverage[:,trid].sum()",
    "HIGH_COVER": "gene.coverage.sum(0)[trid] >= 7",
    "PERMISSIVE": "gene.coverage.sum(0)[trid] >= 2 and (FSM or not (RTTS or INTERNAL_PRIMING or FRAGMENT))",
    "BALANCED": "gene.coverage.sum(0)[trid] >= 2 and (FSM or (HIGH_COVER and not (RTTS or FRAGMENT or INTERNAL_PRIMING)))",
    "STRICT": "gene.coverage.sum(0)[trid] >= 7 and SUBSTANTIAL and (FSM or not (RTTS or FRAGMENT or INTERNAL_PRIMING))",
    "CAGE_SUPPORT": 'sqanti_classification is not None and sqanti_classification["within_CAGE_peak"]',
    "TSS_RATIO": 'sqanti_classification is not None and sqanti_classification["ratio_TSS"] > 1.5',
    "POLYA_MOTIF": 'sqanti_classification is not None and sqanti_classification["polyA_motif_found"]',
    "POLYA_SITE": 'sqanti_classification is not None and sqanti_classification["within_polyA_site"]',
}

SPLICE_CATEGORY = ["FSM", "ISM", "NIC", "NNC", "NOVEL"]


ANNOTATION_VOCABULARY = [
    "antisense",
    "intergenic",
    "genic genomic",
    "novel exonic PAS",
    "novel intronic PAS",
    "readthrough fusion",
    "novel exon",
    "novel 3' splice site",
    "intron retention",
    "novel 5' splice site",
    "exon skipping",
    "novel combination",
    "novel intronic TSS",
    "novel exonic TSS",
    "mono-exon",
    "novel junction",
    "5' fragment",
    "3' fragment",
    "intronic",
]


# filtering functions for the transcriptome class


def add_orf_prediction(
    self: "Transcriptome",
    genome_fn,
    progress_bar=True,
    filter_transcripts=None,
    filter_ref_transcripts=None,
    min_len=300,
    max_5utr_len=500,
    min_kozak=None,
    prefer_annotated_init=True,
    kozak_matrix=DEFAULT_KOZAK_PWM,
    fickett_score=True,
    hexamer_file=None,
):
    """Performs ORF prediction on the transcripts.

    For each transcript the first valid open reading frame is determined, and metrics to assess the coding potential
    (UTR and CDS lengths, Kozak score, Fickett score, hexamer score and NMD prediction). The hexamer score depends on hexamer frequency table,
    see CPAT python module for prebuild tables and instructions.

    :param genome_fn: Path to the genome in fastA format.
    :param min_len: Minimum length of the ORF, Does not apply to annotated initiation sites.
    :param min_kozak: Minimal score for translation initiation site. Does not apply to annotated initiation sites.
    :param max_5utr_len: Maximal length of the 5'UTR region. Does not apply to annotated initiation sites.
    :param prefer_annotated_init: If True, the initiation sites of annotated CDS are preferred.
    :param kozak_matrix: PWM (log odds ratios) for the kozak sequence similarity score.
    :param fickett_score: If set to True, the Fickett TESTCODE score is computed for the ORF.
    :param hexamer_file: Filename of the hexamer table, for the ORF hexamer scores. If set not None, the hexamer score is not computed.
    """
    if filter_transcripts is None:
        filter_transcripts = {}
    if filter_ref_transcripts is None:
        filter_ref_transcripts = {}

    if hexamer_file is None:
        coding = None
        noncoding = None
    else:
        coding = {}
        noncoding = {}
        for line in open(hexamer_file):
            line = line.strip()
            fields = line.split()
            if fields[0] == "hexamer":
                continue
            coding[fields[0]] = float(fields[1])
            noncoding[fields[0]] = float(fields[2])

    with FastaFile(genome_fn) as genome_fh:
        missing_chr = set(self.chromosomes) - set(genome_fh.references)
        if missing_chr:
            missing_genes = sum(len(self.data[mc]) for mc in missing_chr)
            logger.warning(
                "%s contigs are not contained in genome, affecting %s genes. \
                ORFs cannot be computed for these contigs: %s",
                str(len(missing_chr)),
                str(missing_genes),
                str(missing_chr),
            )

        for gene in self.iter_genes(progress_bar=progress_bar):
            if gene.chrom in genome_fh.references:
                if filter_transcripts is not None:
                    gene.add_orfs(
                        genome_fh,
                        reference=False,
                        prefer_annotated_init=prefer_annotated_init,
                        minlen=min_len,
                        min_kozak=min_kozak,
                        max_5utr_len=max_5utr_len,
                        tr_filter=filter_transcripts,
                        kozak_matrix=kozak_matrix,
                        get_fickett=fickett_score,
                        coding_hexamers=coding,
                        noncoding_hexamers=noncoding,
                    )
                if filter_ref_transcripts is not None:
                    gene.add_orfs(
                        genome_fh,
                        reference=True,
                        prefer_annotated_init=prefer_annotated_init,
                        minlen=min_len,
                        min_kozak=min_kozak,
                        max_5utr_len=max_5utr_len,
                        tr_filter=filter_ref_transcripts,
                        get_fickett=fickett_score,
                        kozak_matrix=kozak_matrix,
                        coding_hexamers=coding,
                        noncoding_hexamers=noncoding,
                    )


def add_qc_metrics(
    self: "Transcriptome",
    genome_fn: str,
    progress_bar=True,
    downstream_a_len=30,
    direct_repeat_wd=15,
    direct_repeat_wobble=2,
    direct_repeat_mm=2,
    unify_ends=True,
    correct_tss=True,
):
    """
    Retrieves QC metrics for the transcripts.

    Calling this function populates transcript["biases"] information, which can be used do create filters.
    In particular, the direct repeat length, the downstream adenosine content and information about non-canonical splice sites are fetched.
    In addition, genes are scanned for transcripts that are fully contained in other transcripts.

    :param genome_fn: Path to the genome in fastA format.
    :param downstream_a_len: The number of bases downstream the transcript where the adenosine fraction is determined.
    :param direct_repeat_wd: The number of bases around the splice sites scanned for direct repeats.
    :param direct_repeat_wobble: Number of bases the splice site sequences are shifted.
    :param direct_repeat_mm: Maximum number of missmatches in a direct repeat.
    :param unify_ends: If set, the TSS and PAS are unified using peak calling.
    :param correct_tss: If set TSS are corrected with respect to the reference annotation. Only used if unify_ends is set.
    """

    with FastaFile(genome_fn) as genome_fh:
        missing_chr = set(self.chromosomes) - set(genome_fh.references)
        if missing_chr:
            missing_genes = sum(len(self.data[mc]) for mc in missing_chr)
            logger.warning(
                "%s contigs are not contained in genome, affecting %s genes. \
                Some metrics cannot be computed: %s",
                str(len(missing_chr)),
                str(missing_genes),
                str(missing_chr),
            )

        for gene in self.iter_genes(progress_bar=progress_bar):
            if unify_ends:
                # remove segment graph (if unify TSS/PAS option selected)
                gene.data["segment_graph"] = None
                # "unify" TSS/PAS (if unify TSS/PAS option selected)
                gene._unify_ends(correct_tss=correct_tss)
            # compute segment graph (if not present)
            _ = gene.segment_graph
            gene.add_fragments()
            if gene.chrom in genome_fh.references:
                gene.add_direct_repeat_len(
                    genome_fh,
                    delta=direct_repeat_wd,
                    max_mm=direct_repeat_mm,
                    wobble=direct_repeat_wobble,
                )
                gene.add_noncanonical_splicing(genome_fh)
                gene.add_threeprime_a_content(genome_fh, length=downstream_a_len)

    self.infos["biases"] = True  # flag to check that the function was called


def remove_filter(self, tag):
    """Removes definition of filter tag.

    :param tag: Specify the tag of the filter definition to remove."""
    old = [f.pop(tag, None) for f in self.filter.values()]
    if not any(old):
        logger.error("filter tag %s not found", tag)


def add_filter(self, tag, expression, context="transcript", update=False):
    """Defines a new filter for gene, transcripts and reference transcripts.

    The provided expressions is evaluated during filtering in the provided context, when specified in a query string of a function that supports filtering.
    Importantly, filtering does not modify the original data; rather, it is only applied when specifying the query string.
    For examples, see the default filter definitions isotools.DEFAULT_GENE_FILTER,
    isotools.DEFAULT_TRANSCRIPT_FILTER and isotools.DEFAULT_REF_TRANSCRIPT_FILTER.

    :param tag: Unique tag identifer for this filter. Must be a single word.
    :param expression: Expression to be evaluated on gene, transcript, or reference transcript. Can use existing filters
        from the same context.
    :param context: The context for the filter expression, either "gene", "transcript" or "reference".
    :param update: If set, the already present definition of the provided tag gets overwritten.
    """

    assert context in [
        "gene",
        "transcript",
        "reference",
    ], "filter context must be 'gene', 'transcript' or 'reference'"
    assert tag == re.findall(r"\b\w+\b", tag)[0], '"tag" must be a single word'
    if not update:
        assert (
            tag not in self.filter[context]
        ), f"Filter tag {tag} is already present: `{self.filter[context][tag]}`. Set update=True to re-define."
    if context == "gene":
        attributes = {k for gene in self for k in gene.data.keys() if k.isidentifier()}
    else:
        attributes = {"gene", "trid"}
        if context == "transcript":
            attributes.update(
                {
                    k
                    for gene in self
                    for transcript in gene.transcripts
                    for k in transcript.keys()
                    if k.isidentifier()
                }
            )
        elif context == "reference":
            attributes.update(
                {
                    k
                    for gene in self
                    if gene.is_annotated
                    for transcript in gene.ref_transcripts
                    for k in transcript.keys()
                    if k.isidentifier()
                }
            )

    # test whether the expression can be evaluated
    try:
        _, f_args = _filter_function(expression, self.filter[context])
        # _=f() # this would fail for many default expressions - can be avoided by checking if used attributes are None - but not ideal
        # Could be extended by dummy gene/transcript argument
    except BaseException:
        logger.error("expression cannot be evaluated:\n%s", expression)
        raise
    unknown_attr = [attr for attr in f_args if attr not in attributes]
    if unknown_attr:
        logger.warning(
            f"Some attributes not present in {context} context, please make sure there is no typo: {','.join(unknown_attr)}\n\
                         \rThis can happen for correct filters when there are no or only a few transcripts loaded into the model."
        )
    if update:  # avoid the same tag in different context
        for old_context, filter_dict in self.filter.items():
            if filter_dict.pop(tag, None) is not None:
                logger.info(
                    "replaced existing filter rule %s in %s context", tag, old_context
                )
    self.filter[context][tag] = expression


def iter_genes(
    self: "Transcriptome",
    region=None,
    query=None,
    min_coverage=None,
    max_coverage=None,
    gois=None,
    progress_bar=False,
):
    """Iterates over the genes of a region, optionally applying filters.

    :param region: The region to be considered. Either a string "chr:start-end", or a tuple (chr, start, end). Start and end is optional.
        If omitted, the complete genome is searched.
    :param query: If provided, query string is evaluated on all the genes for filtering. Note that gene tags should be used in the query.
    :param min_coverage: The minimum coverage threshold. Genes with less reads in total are ignored.
    :param max_coverage: The maximum coverage threshold. Genes with more reads in total are ignored.
    :param gois: If provided, only a collection of genes of interest are considered, either gene ids or gene names.
        By default, all the genes are considered.
    :param progress_bar: If set True, the progress bar is shown."""

    if query:
        query_fun, used_tags = _filter_function(query)
        # used_tags={tag for tag in re.findall(r'\b\w+\b', query) if tag not in BOOL_OP}
        all_filter = list(self.filter["gene"])
        msg = "did not find the following filter rules: {}\nvalid rules are: {}"
        assert all(f in all_filter for f in used_tags), msg.format(
            ", ".join(f for f in used_tags if f not in all_filter),
            ", ".join(all_filter),
        )
        filter_fun = {
            tag: _filter_function(tag, self.filter["gene"])[0] for tag in used_tags
        }

        # test the filter expression with dummy tags
        try:
            query_fun(**{tag: True for tag in used_tags})
        except Exception as e:
            logger.error("Error in query string: \n%s", query)
            raise e

    if region is None:
        if gois is None:
            genes = self
        else:
            genes = {self[goi] for goi in gois}
    else:
        if isinstance(region, str):
            if region in self.data:
                genes = self.data[region]  # provide chromosome
                start = None
            else:  # parse region string (chr:start-end)
                try:
                    chrom, pos = region.split(":")
                    start, end = [int(i) for i in pos.split("-")]
                except Exception as e:
                    raise ValueError(
                        'incorrect region {} - specify as string "chr" or "chr:start-end" or tuple ("chr",start,end)'.format(
                            region
                        )
                    ) from e
        elif isinstance(region, tuple):
            chrom, start, end = region
        if start is not None:
            if chrom in self.data:
                genes = self.data[chrom][int(start) : int(end)]
            else:
                raise ValueError("specified chromosome {} not found".format(chrom))
        if gois is not None:
            genes = [gene for gene in genes if gene.id in gois or gene.name in gois]

    # often some genes take much longer than others - smoothing 0 means avg
    for gene in tqdm(genes, disable=not progress_bar, unit="genes", smoothing=0):
        if min_coverage is not None and gene.coverage.sum() < min_coverage:
            continue
        if max_coverage is not None and gene.coverage.sum() > max_coverage:
            continue
        if query is None or query_fun(
            **{tag: fun(**gene.data) for tag, fun in filter_fun.items()}
        ):
            yield gene


def iter_transcripts(
    self: "Transcriptome",
    region=None,
    query=None,
    min_coverage=None,
    max_coverage=None,
    genewise=False,
    gois=None,
    progress_bar=False,
):
    """Iterates over the transcripts of a region, optionally applying filters.

    By default, each iteration returns a 3 Tuple with the gene object, the transcript number and the transcript dictionary.

    :param region: The region to be considered. Either a string "chr:start-end", or a tuple (chr,start,end). Start and end is optional.
        If omitted, the complete genome is searched.
    :param query: If provided, query string is evaluated on all the transcripts for filtering. Note that transcript tags should be used in the query.
    :param min_coverage: The minimum coverage threshold. Transcripts with less reads in total are ignored.
    :param max_coverage: The maximum coverage threshold. Transcripts with more reads in total are ignored.
    :param genewise: In each iteration, return the gene and all transcript numbers and transcript dicts for the gene as tuples.
    :param gois: If provided, only transcripts from the list of genes of interest are considered. Provide as a list of gene ids or gene names.
        By default, all the genes are considered.
    :param progress_bar: Print a progress bar."""

    if query:
        # used_tags={tag for tag in re.findall(r'\b\w+\b', query) if tag not in BOOL_OP}
        all_filter = list(self.filter["transcript"]) + list(self.filter["gene"])
        query_fun, used_tags = _filter_function(query)
        msg = "did not find the following filter rules: {}\nvalid rules are: {}"
        assert all(f in all_filter for f in used_tags), msg.format(
            ", ".join(f for f in used_tags if f not in all_filter),
            ", ".join(all_filter),
        )
        transcript_filter_fun = {
            tag: _filter_function(tag, self.filter["transcript"])[0]
            for tag in used_tags
            if tag in self.filter["transcript"]
        }
        gene_filter_fun = {
            tag: _filter_function(tag, self.filter["gene"])[0]
            for tag in used_tags
            if tag in self.filter["gene"]
        }

        # test the filter expression with dummy tags
        try:
            _ = query_fun(**{tag: True for tag in used_tags})
        except BaseException:
            logger.error("Error in query string: \n%s", query)
            raise
    else:
        transcript_filter_fun = query_fun = None
        gene_filter_fun = {}

    if genewise:
        for gene in self.iter_genes(
            region=region, gois=gois, progress_bar=progress_bar
        ):
            gene_filter_eval = {
                tag: fun(**gene.data) for tag, fun in gene_filter_fun.items()
            }
            filter_result = tuple(
                _filter_transcripts(
                    gene,
                    gene.transcripts,
                    query_fun,
                    transcript_filter_fun,
                    gene_filter_eval,
                    min_coverage,
                    max_coverage,
                )
            )
            if filter_result:
                i_tuple, transcript_tuple = zip(*filter_result)
                yield gene, i_tuple, transcript_tuple
    else:
        for gene in self.iter_genes(
            region=region, gois=gois, progress_bar=progress_bar
        ):
            gene_filter_eval = {
                tag: fun(**gene.data) for tag, fun in gene_filter_fun.items()
            }
            for i, transcript in _filter_transcripts(
                gene,
                gene.transcripts,
                query_fun,
                transcript_filter_fun,
                gene_filter_eval,
                min_coverage,
                max_coverage,
            ):
                yield gene, i, transcript


def iter_ref_transcripts(
    self: "Transcriptome",
    region=None,
    query=None,
    genewise=False,
    gois=None,
    progress_bar=False,
):
    """Iterates over the referemce transcripts of a region, optionally applying filters.

    :param region: The region to be considered. Either a string "chr:start-end", or a tuple (chr,start,end). Start and end is optional.
        If omitted, the complete genome is searched.
    :param genewise: In each iteration, return the gene and all transcript numbers and transcript dicts for the gene as tuples.
    :param query: The query is evaluated in the transcript context, so only transcript filter tags can be used.
    :param genewise: In each iteration, return the gene and all transcript numbers and transcript dicts for the gene as tuples.
    :param gois: If provided, only transcripts from the list of genes of interest are considered. Provide as a list of gene ids or gene names.
        By default, all the genes are considered.
    :param progress_bar: Print a progress bar."""

    if query:
        # used_tags={tag for tag in re.findall(r'\b\w+\b', query) if tag not in BOOL_OP}
        all_filter = list(self.filter["reference"]) + list(self.filter["gene"])
        query_fun, used_tags = _filter_function(query)
        msg = "did not find the following filter rules: {}\nvalid rules are: {}"
        ref_filter_fun = {
            tag: _filter_function(tag, self.filter["reference"])[0]
            for tag in used_tags
            if tag in self.filter["reference"]
        }
        gene_filter_fun = {
            tag: _filter_function(tag, self.filter["gene"])[0]
            for tag in used_tags
            if tag in self.filter["gene"]
        }
        assert all(f in all_filter for f in used_tags), msg.format(
            ", ".join(f for f in used_tags if f not in all_filter),
            ", ".join(all_filter),
        )
        try:  # test the filter expression with dummy tags
            _ = query_fun(**{tag: True for tag in used_tags})
        except BaseException:
            logger.error("Error in query string: \n{query}")
            raise
    else:
        ref_filter_fun = query_fun = None
        gene_filter_fun = {}
    if genewise:
        for gene in self.iter_genes(
            region=region, gois=gois, progress_bar=progress_bar
        ):
            gene_filter_eval = {
                tag: fun(**gene.data) for tag, fun in gene_filter_fun.items()
            }
            filter_result = tuple(
                _filter_transcripts(
                    gene,
                    gene.ref_transcripts,
                    query_fun,
                    ref_filter_fun,
                    gene_filter_eval,
                )
            )
            if filter_result:
                i_tuple, transcript_tuple = zip(*filter_result)
                yield gene, i_tuple, transcript_tuple
    else:
        for gene in self.iter_genes(
            region=region, gois=gois, progress_bar=progress_bar
        ):
            if gene.is_annotated:
                gene_filter_eval = {
                    tag: fun(**gene.data) for tag, fun in gene_filter_fun.items()
                }
                for i, transcript in _filter_transcripts(
                    gene,
                    gene.ref_transcripts,
                    query_fun,
                    ref_filter_fun,
                    gene_filter_eval,
                ):
                    yield gene, i, transcript


def _eval_filter_fun(fun, name, **args):
    """Decorator for the filter functions, which are lambdas and thus cannot have normal decorators.
    On exceptions the provided parameters are reported. This is helpful for debugging.
    """
    try:
        return fun(**args)
    except Exception as e:
        logger.error(
            "error when evaluating filter %s with arguments %s: %s",
            name,
            str(args),
            str(e),
        )
        raise  # either stop evaluation
        # return False   #or continue


def _filter_transcripts(
    gene: "Gene",
    transcripts,
    query_fun,
    filter_fun,
    g_filter_eval,
    mincoverage=None,
    maxcoverage=None,
):
    """Iterator over the transcripts of the gene.

    Transcrips are specified by lists of flags submitted to the parameters.

    :param query_fun: function to be evaluated on tags
    :param filter_fun: tags to be evalutated on transcripts"""
    for i, transcript in enumerate(transcripts):
        if mincoverage and gene.coverage[:, i].sum() < mincoverage:
            continue
        if maxcoverage and gene.coverage[:, i].sum() > maxcoverage:
            continue
        filter_transcript = transcript.copy()
        query_result = query_fun is None or query_fun(
            **g_filter_eval,
            **{
                tag: _eval_filter_fun(f, tag, gene=gene, trid=i, **filter_transcript)
                for tag, f in filter_fun.items()
            },
        )
        if query_result:
            yield i, transcript
