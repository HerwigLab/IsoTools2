from pysam import AlignmentFile
import numpy as np
import pandas as pd
import itertools
import re
from tqdm import tqdm
import builtins
import logging
from scipy.stats import chi2_contingency, fisher_exact
import math
from typing import Literal, TypeAlias, TYPE_CHECKING
from intervaltree import IntervalTree


if TYPE_CHECKING:
    from isotools.transcriptome import Transcriptome
    from .splice_graph import SegmentGraph

ASEType: TypeAlias = Literal["ES", "3AS", "5AS", "IR", "ME", "TSS", "PAS"]
ASEvent: TypeAlias = tuple[list[int], list[int], int, int, ASEType]
"""
In order:
- transcripts supporting the primary event (the longer path for the basic event types)
- transcripts supporting the alternative event (the shorter path for the basic event types)
- node A id
- node B id
- event type
"""

# from Kozak et al, NAR, 1987
kozak = np.array(
    [
        [23, 35, 23, 19],
        [26, 35, 21, 18],
        [25, 35, 22, 18],
        [23, 26, 33, 18],
        [19, 39, 23, 19],
        [23, 37, 20, 20],
        [17, 19, 44, 20],
        [18, 39, 23, 20],
        [25, 53, 15, 7],
        [61, 2, 36, 1],
        [27, 49, 13, 11],
        [15, 55, 21, 9],
        [23, 16, 46, 15],
    ]
)
bg = kozak.sum(0) / kozak.sum()
kozak.sum(1)  # check they sum up to 100%
kozak_weights = np.log2(kozak / 100 / bg)
kozak_weights = np.c_[kozak_weights, np.zeros(kozak_weights.shape[0])]
kozak_pos = list(range(-12, 0)) + [3]
DEFAULT_KOZAK_PWM = pd.DataFrame(kozak_weights.T, columns=kozak_pos, index=[*"ACGTN"])
logger = logging.getLogger("isotools")

cigar = "MIDNSHP=XB"
cigar_lup = {c: i for i, c in enumerate(cigar)}

compl = {"A": "T", "T": "A", "C": "G", "G": "C"}


def rc(seq):
    """reverse complement of seq
    :param seq: sequence
    :return: reverse complement of seq"""
    return "".join(reversed([compl[c] if c in compl else "N" for c in seq]))


def get_error_rate(bam_fn, n=1000):
    qual = 0
    total_len = 0
    with AlignmentFile(bam_fn, "rb", check_sq=False) as align:
        if n is None:
            stats = align.get_index_statistics()
            n = sum([s.mapped for s in stats])
        with tqdm(total=n, unit=" reads") as pbar:
            for i, read in enumerate(align):
                total_len += len(read.query_qualities)
                qual += sum([10 ** (-q / 10) for q in read.query_qualities])
                pbar.update(1)
                if i + 1 >= n:
                    break
    return (qual / total_len) * 100


def basequal_hist(
    bam_fn, qual_bins=10 ** (np.linspace(-7, 0, 30)), len_bins=None, n=10000
):
    """calculates base quality statistics for a bam file:

    :param bam_fn: path to bam file
    :param qual_bins: list of quality thresholds for binning
    :param len_bins: list of read length thresholds for binning
    :param n: number of reads to use for statistics
    :return: pandas Series or DataFrame with base quality statistics"""

    n_len_bins = 1 if len_bins is None else len(len_bins) + 1
    qual = np.zeros((len(qual_bins) + 1, n_len_bins), dtype=int)
    len_i = 0
    i = 0
    with AlignmentFile(bam_fn, "rb") as align:
        if n is None:
            stats = align.get_index_statistics()
            n = sum([s.mapped for s in stats])
        with tqdm(total=n, unit=" reads") as pbar:
            for read in align:
                if read.query_qualities is None:
                    pbar.update(1)
                    continue
                readl = len(read.query_qualities)
                if len_bins is not None:
                    len_i = next(
                        (i for i, th in enumerate(len_bins) if readl < th),
                        len(len_bins),
                    )
                error_rate = (
                    sum([10 ** (-q / 10) for q in read.query_qualities]) / readl * 100
                )
                q_i = next(
                    (i for i, th in enumerate(qual_bins) if error_rate < th),
                    len(qual_bins),
                )
                qual[q_i, len_i] += 1
                pbar.update(1)
                i += 1
                if i >= n:
                    break
    idx = [f"<{th:.2E} %" for th in qual_bins] + [f">={qual_bins[-1]:.2E} %"]
    if len_bins is None:
        return pd.Series(qual[:, 0], index=idx)
    col = [f"<{th/1000:.1f} kb" for th in len_bins] + [f">={len_bins[-1]/1000:.1f} kb"]
    return pd.DataFrame(qual, index=idx, columns=col)


def pairwise(iterable):  # e.g. usefull for enumerating introns
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def cigar_string2tuples(cigarstring):
    """converts cigar string to tuples ((operator_id, length), ...)
    :param cigarstring: cigar string
    :return: tuple of tuples"""

    res = re.findall(f"(\\d+)([{cigar}]+)", cigarstring)
    return tuple((cigar_lup[c], int(n)) for n, c in res)


def junctions_from_cigar(cigartuples, offset):
    "returns the exon positions"
    exons = list([[offset, offset]])
    for cigar in cigartuples:
        # N -> Splice junction
        if cigar[0] == 3:
            pos = exons[-1][1] + cigar[1]
            if exons[-1][0] == exons[-1][1]:
                # delete zero length exons
                # (may occur if insertion within intron, e.g. 10M100N10I100N10M)
                del exons[-1]
            exons.append([pos, pos])
        # MD=X -> move forward on reference
        elif cigar[0] in (0, 2, 7, 8):
            exons[-1][1] += cigar[1]
    # delete 0 length exons at the end
    if exons[-1][0] == exons[-1][1]:
        del exons[-1]
    return exons


def is_same_gene(tr1, tr2, spj_iou_th=0, reg_iou_th=0.5):
    "Checks whether tr1 and tr2 are the same gene by calculating intersection over union of the intersects"
    # current default definition of "same gene": at least one shared splice site
    # or more than 50% exonic overlap
    spj_i, reg_i = get_intersects(tr1, tr2)
    total_spj = (len(tr1) + len(tr2) - 2) * 2
    spj_iou = spj_i / (total_spj - spj_i) if total_spj > 0 else 0
    if spj_iou > spj_iou_th:
        return True
    total_len = sum([e[1] - e[0] for e in tr2 + tr1])
    reg_iou = reg_i / (total_len - reg_i)
    if reg_iou > reg_iou_th:
        return True
    return False


def splice_identical(exon_list1, exon_list2, strictness=math.inf):
    """
    Check whether two transcripts are identical in terms of splice sites.
    :param exon_list1: transcript 1 as a list of tuples for each exon
    :param exon_list2: transcript 2 as a list of tuples for each exon
    :param strictness: Number of bp that are allowed to differ for transcription start and end sites to be still considered identical.
    """
    # all splice sites are equal
    # different number of exons
    if len(exon_list1) != len(exon_list2):
        return False
    # single exon genes
    if len(exon_list1) == 1 and has_overlap(exon_list1[0], exon_list2[0]):
        return True
    # Check start of first and end of last exon
    if (
        abs(exon_list1[0][0] - exon_list2[0][0]) > strictness
        or abs(exon_list1[-1][1] - exon_list2[-1][1]) > strictness
    ):
        return False
    # check end of first and and start of last exon
    if exon_list1[0][1] != exon_list2[0][1] or exon_list1[-1][0] != exon_list2[-1][0]:
        return False
    # check other exons
    for exon1, exon2 in zip(exon_list1[1:-1], exon_list2[1:-1]):
        if exon1[0] != exon2[0] or exon1[1] != exon2[1]:
            return False
    return True


def kozak_score(sequence, pos, pwm=DEFAULT_KOZAK_PWM):
    return sum(
        pwm.loc[sequence[pos + i], i]
        for i in pwm.columns
        if pos + i >= 0 and pos + i < len(sequence)
    )


def find_orfs(
    sequence, start_codons=["ATG"], stop_codons=["TAA", "TAG", "TGA"], ref_cds={}
):
    """Find all open reading frames on the forward strand of the sequence.
    Return a 7-tuple with start and stop position, reading frame (0,1 or 2), start and stop codon sequence,
    number of upstream start codons, and the ids of the reference transcripts with matching CDS initialization.
    """
    orf = []
    starts = [[], [], []]
    stops = [[], [], []]
    for init, ref_ids in ref_cds.items():
        starts[init % 3].append((init, sequence[init: (init + 3)], ref_ids))
    for match in re.finditer("|".join(start_codons), sequence):
        if match.start() not in ref_cds:
            starts[match.start() % 3].append(
                (match.start(), match.group(), None)
            )  # position and codon
    for match in re.finditer("|".join(stop_codons), sequence):
        stops[match.start() % 3].append((match.end(), match.group()))
    for frame in range(3):
        stop, stop_codon = (0, None)
        for start, start_codon, ref_ids in starts[frame]:
            if (
                start < stop and ref_ids is None
            ):  # inframe start within the previous ORF
                continue
            try:
                stop, stop_codon = next(s for s in sorted(stops[frame]) if s[0] > start)
            except StopIteration:  # no stop codon - still report as it might be an uAUG
                stop, stop_codon = start, None
            orf.append([start, stop, frame, start_codon, stop_codon, 0, ref_ids])
    uORFs = 0
    for orf_i in sorted(orf):  # sort by start position
        orf_i[5] = uORFs
        uORFs += 1

    return sorted(orf)

    # return max(orf, key=lambda x: (bool(x[6]), x[1]-x[0] if x[1] else -1))  # prefer annotated CDS init


def has_overlap(r1, r2):
    "check the overlap of two intervals"
    # assuming start < end
    return r1[1] > r2[0] and r2[1] > r1[0]


def get_overlap(r1, r2):
    "check the overlap of two intervals"
    # assuming start < end
    return max(0, min(r1[1], r2[1]) - max(r1[0], r2[0]))


def get_intersects(tr1, tr2):
    "get the number of intersecting splice sites and intersecting bases of two transcripts"
    tr1_enum = enumerate(tr1)
    tr2_enum = enumerate(tr2)
    sjintersect = 0
    intersect = 0
    try:
        i, tr1_exon = next(tr1_enum)
        j, tr2_exon = next(tr2_enum)
        while True:
            if has_overlap(tr1_exon, tr2_exon):
                if tr1_exon[0] == tr2_exon[0] and i > 0 and j > 0:
                    sjintersect += 1
                if tr1_exon[1] == tr2_exon[1] and i < len(tr1) - 1 and j < len(tr2) - 1:
                    sjintersect += 1
                i_end = min(tr1_exon[1], tr2_exon[1])
                i_start = max(tr1_exon[0], tr2_exon[0])
                intersect += i_end - i_start
            if tr1_exon[1] <= tr2_exon[0]:
                i, tr1_exon = next(tr1_enum)
            else:
                j, tr2_exon = next(tr2_enum)
    except StopIteration:
        return sjintersect, intersect


def _filter_function(expression, context_filters={}):
    """
    converts a string e.g. "all(x[0]/x[1]>3)" into a function
    if context_filters is provided, filter tags will be recursively replaced with their expression
    """
    assert isinstance(expression, str), "expression should be a string"
    # extract argument names
    used_filters = []
    depth = 0
    original_expression = expression
    while True:
        for filter in used_filters:
            if filter in context_filters:
                # brackets around expression prevent unintended "mixing" of neighboring filters
                expression = expression.replace(filter, f"({context_filters[filter]})")
        f = eval(f"lambda: {expression}")
        args = [n for n in f.__code__.co_names if n not in dir(builtins)]
        used_filters = [arg for arg in args if arg in context_filters]
        depth += 1
        if len(used_filters) == 0:
            break
        if depth > 10:
            raise ValueError(
                f"Filter expression evaluation max depth reached. Expression `{original_expression}` was evaluated to `{expression}`"
            )

    # potential issue: gene.coverage gets detected as ["gene", "coverage"], e.g. coverage is added. Probably not causing trubble
    return (
        eval(
            f'lambda {",".join([arg+"=None" for arg in args]+["**kwargs"])}: bool({expression})\n',
            {},
            {},
        ),
        args,
    )


def _interval_dist(a: tuple[int, int], b: tuple[int, int]):
    """compute the distance between two intervals a and b."""
    return max([a[0], b[0]]) - min([a[1], b[1]])


def _filter_event(
    coverage,
    event: ASEvent,
    segment_graph: "SegmentGraph",
    min_total=100,
    min_alt_fraction=0.1,
    min_dist_AB=0,
):
    """
    return True if the event satisfies the filter conditions and False otherwise

    :param coverage: 1D array of counts per transcript
    :param event: Event obtained from .find_splice_bubbles()
    :param min_total: The minimum total number of reads for an event to pass the filter
    :type min_total: int
    :param min_alt_fraction: The minimum fraction of read supporting the alternative
    :type min_alt_frction: float
    :param min_dist_AB: Minimum distance (in nucleotides) between node A and B in an event
    """

    tr_IDs = event[0] + event[1]
    tot_cov = coverage[tr_IDs].sum()

    if tot_cov < min_total:
        return False

    pri_cov = coverage[event[0]].sum()
    alt_cov = coverage[event[1]].sum()
    frac = min(pri_cov, alt_cov) / tot_cov

    if frac < min_alt_fraction:
        return False

    coordinates = segment_graph._get_event_coordinate(event)
    if coordinates[1] - coordinates[0] < min_dist_AB:
        return False

    return True


def _get_exonic_region(transcripts):
    exon_starts = iter(
        sorted([e[0] for transcript in transcripts for e in transcript["exons"]])
    )
    exon_ends = iter(
        sorted([e[1] for transcript in transcripts for e in transcript["exons"]])
    )
    exon_region = [[next(exon_starts), next(exon_ends)]]
    for next_start in exon_starts:
        if next_start <= exon_region[-1][1]:
            exon_region[-1][1] = next(exon_ends)
        else:
            exon_region.append([next_start, next(exon_ends)])
    return exon_region


def _get_overlap(exons, transcripts):
    """Compute the exonic overlap of a new transcript with the segment graph.
    Avoids the computation of segment graph, which provides the same functionality.

    :param exons: A list of exon tuples representing the new transcript
    :type exons: list
    :return: boolean array indicating whether the splice site is contained or not"""
    if not transcripts:
        return 0
    # 1) get exononic regions in transcripts
    exon_region = _get_exonic_region(transcripts)
    # 2) find overlap of exonic regions with exons
    ol = 0
    i = 0
    for exon in exons:
        while exon_region[i][1] < exon[0]:  # no overlap, go on
            i += 1
            if i == len(exon_region):
                return ol
        while exon_region[i][0] < exon[1]:
            i_end = min(exon[1], exon_region[i][1])
            i_start = max(exon[0], exon_region[i][0])
            ol += i_end - i_start
            if exon_region[i][1] > exon[1]:  # might overlap with next exon
                break
            i += 1
            if i == len(exon_region):
                return ol
    return ol


def _find_splice_sites(splice_junctions, transcripts):
    """Checks whether the splice sites of a new transcript are present in the set of transcripts.
    Avoids the computation of segment graph, which provides the same functionality.

    :param splice_junctions: A list of 2 tuples with the splice site positions
    :param transcripts: transcripts to scan
    :return: boolean array indicating whether the splice site is contained or not"""

    sites = np.zeros((len(splice_junctions)) * 2, dtype=bool)
    # check exon ends
    splice_junction_starts = {}
    splice_junction_ends = {}
    for i, splice_site in enumerate(splice_junctions):
        splice_junction_starts.setdefault(splice_site[0], []).append(i)
        splice_junction_ends.setdefault(splice_site[1], []).append(i)

    transcript_list = [
        iter(transcript["exons"][:-1])
        for transcript in transcripts
        if len(transcript["exons"]) > 1
    ]
    current = [next(transcript) for transcript in transcript_list]
    for splice_junction_start, idx in sorted(
        splice_junction_starts.items()
    ):  # splice junction starts, sorted by position
        for j, transcript_iter in enumerate(transcript_list):
            try:
                while splice_junction_start > current[j][1]:
                    current[j] = next(transcript_iter)
                if current[j][1] == splice_junction_start:
                    for i in idx:
                        sites[i * 2] = True
                    break
            except StopIteration:
                continue
    # check exon starts
    transcript_list = [
        iter(transcript["exons"][1:])
        for transcript in transcripts
        if len(transcript["exons"]) > 1
    ]
    current = [next(transcript) for transcript in transcript_list]
    for splice_junction_end, idx in sorted(
        splice_junction_ends.items()
    ):  # splice junction ends, sorted by position
        for j, transcript_iter in enumerate(transcript_list):
            try:
                while splice_junction_end > current[j][0]:
                    current[j] = next(transcript_iter)
                if current[j][0] == splice_junction_end:
                    for i in idx:
                        sites[i * 2 + 1] = True
                    break
            except StopIteration:
                continue
    return sites


def precompute_events_dict(
    transcriptome: "Transcriptome",
    event_type: list[ASEType] = ("ES", "5AS", "3AS", "IR", "ME"),
    min_cov=100,
    region=None,
    query=None,
    progress_bar=True,
):
    """
    Precomputes the events_dict, i.e. a dictionary of splice bubbles. Each key is a gene and each value is the splice bubbles
    object corresponding to that gene.
    :param region: The region to be considered. Either a string "chr:start-end", or a tuple (chr,start,end). Start and end is optional.
    """

    events_dict = {}

    for gene in transcriptome.iter_genes(
        region=region, query=query, progress_bar=progress_bar
    ):
        sg = gene.segment_graph
        events = [
            event
            for event in sg.find_splice_bubbles(types=event_type)
            if gene.coverage.sum(axis=0)[event[0] + event[1]].sum() >= min_cov
        ]
        if events:
            events_dict[gene.id] = events

    return events_dict


def get_quantiles(pos: list[tuple[int, int]], percentile=[0.5]):
    """provided a list of (positions,coverage) pairs, return the median position"""
    # percentile should be sorted, and between 0 and 1
    total = sum(cov for _, cov in pos)
    n = 0
    result_list: list[int] = []
    for p, cov in sorted(pos, key=lambda x: x[0]):
        n += cov
        while n >= total * percentile[len(result_list)]:
            result_list.append(p)
            if len(result_list) == len(percentile):
                return result_list
    raise ValueError(f"cannot find {percentile[len(result_list)]} percentile of {pos}")


def smooth(x, window_len=31):
    """smooth the data using a hanning window with requested size."""
    # padding with mirrored
    s = np.r_[x[window_len - 1: 0: -1], x, x[-2: -window_len - 1: -1]]
    # print(len(s))
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[int(window_len / 2 - (window_len + 1) % 2): -int(window_len / 2)]


def prepare_contingency_table(eventA: ASEvent, eventB: ASEvent, coverage):
    """
    Prepare the read counts and transcript id contingency tables for two events.

    Returns two 2x2 contingency tables, one with the read counts, one with the transcript events

    :param eventA: First alternative splicing event obtained from .find_splice_bubbles()
    :param eventB: Second alternative splicing event obtained from .find_splice_bubbles()
    :param coverage: Read counts per transcript.
    """

    con_tab = np.zeros((2, 2), dtype=int)
    transcript_id_table = np.zeros((2, 2), dtype=object)

    for m, n in itertools.product(range(2), range(2)):
        transcript_ids = sorted(
            set(eventA[m]) & set(eventB[n]), key=coverage.__getitem__, reverse=True
        )
        transcript_id_table[n, m] = transcript_ids
        con_tab[n, m] = coverage[transcript_ids].sum()
    return con_tab, transcript_id_table


def pairwise_event_test(
    con_tab, test: Literal["fisher", "chi2"] = "fisher", pseudocount=0.01
):
    """
    Performs an independence test on the contingency table and computes effect sizes.

    :param con_tab: contingency table with the read counts
    :param test: Test to be performed. One of ("chi2", "fisher")
    :type test: str
    """
    if test == "chi2":
        test_fun = chi2_contingency
    elif test == "fisher":
        test_fun = fisher_exact
    else:
        raise (ValueError('test should be "chi2" or "fisher"'))

    # add some small value for chi2
    # TODO: inconsistency: test_stat is the odds ratio for the fisher test and X^2 for the chisq test
    test_stat, p_value = test_fun(con_tab + pseudocount if test == "chi2" else con_tab)

    # priA_priB_trID, altA_altB_trID = tr_ID_tab[0, 0], tr_ID_tab[1, 1]
    # priA_altB_trID, altA_priB_trID = tr_ID_tab[1, 0], tr_ID_tab[0, 1]
    # priA_priB, altA_altB = con_tab[0, 0], con_tab[1, 1]
    # priA_altB, altA_priB = con_tab[1, 0], con_tab[0, 1]
    log2OR = _corrected_log2OR(con_tab)
    # logOR is a measure of the effect size. coordination between the events is either positive or negative.
    dcPSI_AB, dcPSI_BA = dcPSI(con_tab)
    # delta conditional PSI is another measure of the effect size.

    return p_value, test_stat, log2OR, dcPSI_AB, dcPSI_BA


def _corrected_log2OR(con_tab):
    con_tab_copy = np.zeros((2, 2), dtype=float)

    for m, n in itertools.product(range(2), range(2)):
        if con_tab[n, m] == 0:
            con_tab_copy[n, m] = 10**-9
        else:
            con_tab_copy[n, m] = con_tab[n, m]
    log2OR = np.log2((con_tab_copy[0, 0] * con_tab_copy[1, 1])) - np.log2(
        (con_tab_copy[0, 1] * con_tab_copy[1, 0])
    )
    return log2OR


def dcPSI(con_tab):
    """delta conditional PSI of a coordinated event"""
    # 1) dcPSI_AB= PSI(B | altA) - PSI(B)
    dcPSI_AB = con_tab[1, 1] / con_tab[:, 1].sum() - con_tab[1, :].sum() / con_tab.sum(
        None
    )
    # 2) dcPSI_BA= PSI(A | altB) - PSI(A)
    dcPSI_BA = con_tab[1, 1] / con_tab[1, :].sum() - con_tab[:, 1].sum() / con_tab.sum(
        None
    )
    return dcPSI_AB, dcPSI_BA


def genomic_position(tr_pos, exons, reverse_strand):
    tr_len = sum((e[1] - e[0]) for e in exons)
    assert all(
        p <= tr_len for p in tr_pos
    ), f"Requested positions {tr_pos} for transcript of length {tr_len}."
    if reverse_strand:
        tr_pos = [tr_len - p for p in tr_pos]
    tr_pos = sorted(set(tr_pos))
    intron_len = 0
    mapped_pos = []
    i = 0
    offset = exons[0][0]
    for e1, e2 in pairwise(exons):
        while offset + intron_len + tr_pos[i] < e1[1]:
            mapped_pos.append(offset + intron_len + tr_pos[i])
            i += 1
            if i == len(tr_pos):
                break
        else:
            intron_len += e2[0] - e1[1]
            continue
        break
    else:
        for i in range(i, len(tr_pos)):
            mapped_pos.append(offset + intron_len + tr_pos[i])
    if reverse_strand:  # get them back to the original
        tr_pos = [tr_len - p for p in tr_pos]
    return {p: mp for p, mp in zip(tr_pos, mapped_pos)}


def cmp_dist(a, b, min_dist=3):
    if a >= b + min_dist:
        return 1
    if b >= a + min_dist:
        return -1
    return 0


# region gene structure variation


def structure_feature_cov(transcripts, samples, feature="TSS"):
    """
    :param transcripts: A list of transcript annotations of a gene obtained from isoseq[gene].transcripts.
    :param feature: 'EC', 'TSS', 'PAS'.
    :param samples: A list of sample names to specify the samples to be considered.
    :return: Selected feature of input transcripts and corresponding coverage across samples.

    1) EC - exon_chain, query coverage matrix and exon positions, and return all the exon_chain and coverage.
    2) TSS/PAS, query TSS_unified or PAS_unified coverage matrix, and return all the positions and coverage.
    """

    assert feature in ["EC", "TSS", "PAS"], "choose feature from EC, TSS, PAS"

    cov = {}
    if feature == "EC":
        field = "coverage"
        for transcript in transcripts:
            if transcript[field] is None:
                continue

            # Convert list of exons to tuple of tuples to make it hashable as a key of a dictionary
            exon_chain = tuple(map(tuple, transcript["exons"]))
            for s, n in transcript[field].items():
                if s in samples:
                    cov[exon_chain] = cov.get(exon_chain, 0) + n
    else:
        field = f"{feature}_unified"
        for transcript in transcripts:
            if transcript[field] is None:
                continue

            for s, pos_dict in transcript[field].items():
                if s in samples:
                    for pos, n in pos_dict.items():
                        cov[pos] = cov.get(pos, 0) + n

    # keep ones with coverage > 0
    cov = {k: v for k, v in cov.items() if v > 0}

    if len(cov) == 0:
        return [], []

    # sort the coverage in descending order and return
    occurrence, abundance = zip(*sorted(cov.items(), key=lambda x: x[1], reverse=True))
    return abundance, occurrence


def count_distinct_pos(pos_list, strict_pos=15):
    """
    :param pos_list: A list of TSS/PAS positions, sorted by their abundance descendingly (output from structure_feature_cov).
    :param strict_pos: Difference allowed between two positions when considering identical TSS/PAS.
    :return: How many distinct positions are there.
    """

    tree = IntervalTree()
    picked = 0
    for pos in pos_list:
        if len(tree[pos]) == 0:
            tree[pos - strict_pos: pos + strict_pos + 1] = 1
            picked += 1
    return picked


def count_distinct_exon_chain(ec_list, strict_ec=0, strict_pos=15):
    """
    :param ec_list: A list of exon chains, sorted by their abundance descendingly (output from structure_feature_cov).
    :param strict_ec: Distance allowed between each position, except for the first/last, in two exon chains so that they can be considered as identical.
    :param strict_pos: Difference allowed between two positions when considering identical TSS/PAS.
    :return: How many distinct exon chains are there.
    """

    merged_idx = set()
    for x in range(len(ec_list) - 1):
        if x in merged_idx:
            continue
        for y in range(x + 1, len(ec_list)):
            if y in merged_idx:
                continue
            # if the number of exons is different, skip
            if len(ec_list[x]) != len(ec_list[y]):
                continue

            pos_in_x = [pos for exon in ec_list[x] for pos in exon]
            pos_in_y = [pos for exon in ec_list[y] for pos in exon]

            pos_diff = [abs(m - n) for m, n in zip(pos_in_x, pos_in_y)]

            if all(
                (
                    d <= strict_pos
                    if (i == 0 or i == len(pos_diff) - 1)
                    else d <= strict_ec
                )
                for i, d in enumerate(pos_diff)
            ):
                # keep the one with higher coverage
                merged_idx.add(y)

    return len(ec_list) - len(merged_idx)


def str_var_triplet(transcripts, samples, strict_ec=0, strict_pos=15):
    """
    Quantify the structure variation of transcripts in a gene across specified samples.

    :param transcripts: A list of transcript annotations of a gene obtained from isoseq[gene].transcripts.
    :param samples: A list of sample names to specify the samples to be considered..
    :param strict_ec: Distance allowed between each position, except for the first/last, in two exon chains so that they can be considered as identical.
    :param strict_pos: Difference allowed between two positions when considering identical TSS/PAS.
    :return (list): A triplet of numbers in the order of distinct TSS positions, exon chains, and PAS positions.
    """

    _, ec_list = structure_feature_cov(
        transcripts=transcripts, samples=samples, feature="EC"
    )
    n_ec = count_distinct_exon_chain(
        ec_list=ec_list, strict_ec=strict_ec, strict_pos=strict_pos
    )

    _, tss_list = structure_feature_cov(
        transcripts=transcripts, samples=samples, feature="TSS"
    )
    n_tss = count_distinct_pos(pos_list=tss_list, strict_pos=strict_pos)

    _, pas_list = structure_feature_cov(
        transcripts=transcripts, samples=samples, feature="PAS"
    )
    n_pas = count_distinct_pos(pos_list=pas_list, strict_pos=strict_pos)

    return [n_tss, n_ec, n_pas]


# endregion
