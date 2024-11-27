from scipy.stats import binom, norm, chi2, betabinom, nbinom  # pylint: disable-msg=E0611
from scipy.special import gammaln, polygamma  # pylint: disable-msg=E0611
from scipy.optimize import minimize, minimize_scalar
import statsmodels.stats.multitest as multi
import logging
import math
import numpy as np
import pandas as pd
import itertools
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .transcriptome import Transcriptome

# from .decorators import deprecated, debug, experimental
from .splice_graph import SegmentGraph
from ._utils import _filter_function, ASEType, str_var_triplet

logger = logging.getLogger('isotools')

# differential splicing


def proportion_test(x, n):
    # Normal approximation
    # x,n should be length 2(the two groups)
    # tests H0: proportions are equal vs H1: proportions are different (two sided)
    x = [xi.sum() for xi in x]
    n = [ni.sum() for ni in n]
    p1 = [x[i] / n[i] for i in range(2)]
    p0 = (x[0] + x[1]) / (n[0] + n[1])
    z = abs(p1[0] - p1[1]) / np.sqrt(p0 * (1 - p0) * (1 / n[0] + 1 / n[1]))
    return (2 * norm.sf(z)), (p1[0], 0, p1[1], 0, p0, 0)  # two sided alternative


def binom_lr_test(x, n):
    # likelihood ratio test
    # x,n should be length 2 (the two groups)
    # principle: log likelihood ratio of M0/M1 is chi2 distributed
    x = [xi.sum() for xi in x]
    n = [ni.sum() for ni in n]
    p1 = [x[i] / n[i] for i in range(2)]
    p0 = (x[0] + x[1]) / (n[0] + n[1])
    # calculate the log likelihoods
    l0 = binom.logpmf(x, n, p0).sum()
    l1 = binom.logpmf(x, n, p1).sum()
    # calculate the pvalue (sf=1-csf(), 1df)
    return chi2.sf(2 * (l1 - l0), 1), (p1[0], 0, p1[1], 0, p0, 0)


def loglike_betabinom(params, k, n):
    '''returns  log likelihood of betabinomial and its partial derivatives'''
    a, b = params
    logpdf = gammaln(n + 1) + gammaln(k + a) + gammaln(n - k + b) + gammaln(a + b) - \
        (gammaln(k + 1) + gammaln(n - k + 1) + gammaln(a) + gammaln(b) + gammaln(n + a + b))
    e = polygamma(0, a + b) - polygamma(0, n + a + b)
    da = e + polygamma(0, k + a) - polygamma(0, a)
    db = e + polygamma(0, n - k + b) - polygamma(0, b)
    return -np.sum(logpdf), np.array((-np.sum(da), -np.sum(db)))


def betabinom_ml(xi, ni):
    '''Calculate maximum likelihood parameter of beta binomial distribution for a group of samples with xi successes and ni trials.

    :param xi: number of successes, here coverage of the alternative for all samples of the group as 1d numpy array
    :param ni: number of trials, here total coverage for the two sample groups for all samples of the group as 1d numpy array
    '''
    # x and n must be np arrays
    if sum(ni) == 0:
        params = params_alt = None, None
        return params, params_alt, False
    xi, ni = xi[ni > 0], ni[ni > 0]  # avoid div by 0
    prob = xi / ni
    m = prob.mean()  # estimate initial parameters
    d = prob.var()
    success = True
    if d == 0:  # just one sample? or all exactly the same proportion
        params = params_alt = m, None  # in this case the betabinomial reduces to the binomial
    else:
        d = max(d, 1e-6)  # to avoid division by 0
        e = (m**2 - m + d)  # helper
        # find ml estimates for a and b
        mle = minimize(loglike_betabinom, x0=[-m * e / d, ((m - 1) * e) / d], bounds=((1e-6, None), (1e-6, None)),
                       args=(xi, ni), options={'maxiter': 250}, method='L-BFGS-B', jac=True)
        a, b = params = mle.x
        params_alt = (a / (a + b), a * b / ((a + b)**2 * (a + b + 1)))  # get alternative parametrization (mu and disp)
        # mle = minimize(loglike_betabinom2, x0=[-d/(m*e),d/((m-1)*e)],bounds=((1e-9,None),(1e-9,None)),
        #   args=(xi,ni),options={'maxiter': 250}, method='L-BFGS-B', tol=1e-6)
        # params=([1/p for p in mle.x])
        params_alt = (a / (a + b), a * b / ((a + b)**2 * (a + b + 1)))  # get alternative parametrization (mu and disp)

        if not mle.success:
            # should not happen to often, mainly with mu close to boundaries
            logger.debug(f'no convergence in betabinomial fit: k={xi}\nn={ni}\nparams={params}\nmessage={mle.message}')
            success = False  # prevent calculation of p-values based on non optimal parameters
    return params, params_alt, success


def betabinom_lr_test(x, n):
    ''' Likelihood ratio test with random-effects betabinomial model.

    This test modles x as betabinomial(n,a,b), eg a binomial distribution, where p follows beta ditribution with parameters a,b>0
    mean m=a/(a+b) overdispersion d=ab/((a+b+1)(a+b)^2) --> a=-m(m^2-m+d)/d b=(m-1)(m^2-m+d)/d
    principle: log likelihood ratio of M0/M1 is chi2 distributed

    :param x: coverage of the alternative for the two sample groups
    :param n: total coverage for the two sample groups'''

    if any(ni.sum() == 0 for ni in n):
        return (np.nan, [None, None])  # one group is not covered at all - no test possible. Checking this to avoid RuntimeWarnings (Mean of empty slice)
    x_all, n_all = (np.concatenate(x), np.concatenate(n))
    # calculate ml parameters
    ml_1 = betabinom_ml(x[0], n[0])
    ml_2 = betabinom_ml(x[1], n[1])
    ml_all = betabinom_ml(x_all, n_all)

    if not (ml_1[2] and ml_2[2] and ml_all[2]):  # check success
        return np.nan, list(ml_1[1] + ml_2[1] + ml_all[1])
    try:
        l0 = betabinom_ll(x_all, n_all, *ml_all[0]).sum()
        l1 = betabinom_ll(x[0], n[0], *ml_1[0]).sum() + betabinom_ll(x[1], n[1], *ml_2[0]).sum()
    except (ValueError, TypeError):
        logger.critical(f'betabinom error: x={x}\nn={n}\nparams={ml_1[0]}/{ml_2[0]}/{ml_all[0]}')  # should not happen
        raise
    return chi2.sf(2 * (l1 - l0), 2), list(ml_1[1] + ml_2[1] + ml_all[1])  # note that we need two degrees of freedom here as h0 hsa two parameters, h1 has 4


def betabinom_ll(x, n, a, b):
    if b is None:
        return binom.logpmf(x, n, a).sum()
    else:
        return betabinom.logpmf(x, n, a, b).sum()


TESTS = {'betabinom_lr': betabinom_lr_test,
         'binom_lr': binom_lr_test,
         'proportions': proportion_test}


def _check_groups(transcriptome: 'Transcriptome', groups, n_groups=2):
    assert len(groups) == n_groups, f"length of groups should be {n_groups}, but found {len(groups)}"
    # find groups and sample indices
    if isinstance(groups, dict):
        groupnames = list(groups)
        groups = list(groups.values())
    elif all(isinstance(groupname, str) and groupname in transcriptome.groups() for groupname in groups):
        groupnames = list(groups)
        groups = [transcriptome.groups()[gn] for gn in groupnames]
    elif all(isinstance(group, list) for group in groups):
        groupnames = [f'group{i+1}' for i in range(len(groups))]
    else:
        raise ValueError('groups not found in dataset (samples must be a str, list or dict)')
    notfound = [sample for group in groups for sample in group if sample not in transcriptome.samples]
    if notfound:
        raise ValueError(f"Cannot find the following samples: {notfound}")
    assert all((groupname1 not in groupname2 for groupname1, groupname2 in itertools.permutations(groupnames, 2))), 'group names must not be contained in other group names'
    sample_idx = {sample: idx for sample, idx in transcriptome._get_sample_idx().items()}
    grp_idx = [[sample_idx[sample] for sample in group] for group in groups]
    return groupnames, groups, grp_idx


def altsplice_test(self: 'Transcriptome', groups, min_total=100, min_alt_fraction=.1, min_n=10, min_sa=.51, test='auto', padj_method='fdr_bh',
                   types: Optional[list[ASEType]] = None, **kwargs):
    '''Performs the alternative splicing event test.

    :param groups: Dict with group names as keys and lists of sample names as values, defining the two groups for the test.
        If more then two groups are provided, test is performed between first two groups, but maximum likelihood parameters
        (expected PSI and dispersion) will be computed for the other groups as well.
    :param min_total: Minimum total coverage over all selected samples (for both groups combined).
    :param min_alt_fraction: Minimum fraction of reads supporting the alternative (for both groups combined).
    :param min_n: The minimum coverage of the event for an individual sample to be considered for the min_sa filter.
    :param min_sa: The fraction of samples within each group that must be covered by at least min_n reads.
    :param test: The name of one of the implemented statistical tests ('betabinom_lr','binom_lr','proportions').
    :param padj_method: Specify the method for multiple testing correction.
    :param types: Restrict the analysis on types of events. If omitted, all types are tested.
    :param kwargs: Additional keyword arguments are passed to iter_genes.'''

    noORF = (None, None, {'NMD': True})
    groupnames, groups, group_idx = _check_groups(self, groups)
    sidx = np.array(group_idx[0] + group_idx[1])

    if isinstance(test, str):
        if test == 'auto':
            test = 'betabinom_lr' if min(len(group) for group in groups[:2]) > 1 else 'proportions'
        test_name = test
        try:
            test = TESTS[test]
        except KeyError as e:
            raise ValueError(f'test must be one of {str(list(TESTS))}') from e
    else:
        test_name = 'custom'

    logger.info('testing differential splicing for %s using %s test', ' vs '.join(f'{groupnames[i]} ({len(groups[i])})' for i in range(2)), test_name)

    if min_sa < 1:
        min_sa *= sum(len(group) for group in groups[:2])
    res = []
    for gene in self.iter_genes(**kwargs):
        if gene.coverage[sidx, :].sum() < min_total:
            continue
        known = {}  # check for known events
        if gene.is_annotated and gene.n_transcripts:
            sg = gene.ref_segment_graph
            # find annotated alternatives for gene (e.g. known events)
            for _, _, nX, nY, splice_type in sg.find_splice_bubbles(types=types):
                if splice_type in ("TSS", "PAS"):
                    if (splice_type == "TSS") == (gene.strand == "+"):
                        known.setdefault(splice_type, set()).add((sg[nX].end))
                    else:
                        known.setdefault(splice_type, set()).add((sg[nY].start))
                else:
                    known.setdefault(splice_type, set()).add((sg[nX].end, sg[nY].start))
        sg = gene.segment_graph
        for setA, setB, nX, nY, splice_type in sg.find_splice_bubbles(types=types):
            junction_cov = gene.coverage[:, setB].sum(1)
            total_cov = gene.coverage[:, setA].sum(1) + junction_cov
            if total_cov[sidx].sum() < min_total:
                continue

            alt_fraction = junction_cov[sidx].sum() / total_cov[sidx].sum()
            if alt_fraction < min_alt_fraction or alt_fraction > 1 - min_alt_fraction:
                continue
            x = [junction_cov[grp] for grp in group_idx]
            n = [total_cov[grp] for grp in group_idx]
            if sum((ni >= min_n).sum() for ni in n[:2]) < min_sa:
                continue
            pval, params = test(x[:2], n[:2])
            params_other = tuple(v for xi, ni in zip(x[2:], n[2:]) for v in betabinom_ml(xi, ni)[1])
            if splice_type in ['TSS', 'PAS']:
                start, end = sg[nX].start, sg[nY].end
                if (splice_type == "TSS") == (gene.strand == "+"):
                    novel = end not in known.get(splice_type, set())
                else:
                    novel = start not in known.get(splice_type, set())
            else:
                start, end = sg[nX].end, sg[nY].start
                novel = (start, end) not in known.get(splice_type, set())

            nmdA = sum(gene.coverage[np.ix_(sidx, [transcript_id])].sum(None)
                       for transcript_id in setA if gene.transcripts[transcript_id].get('ORF', noORF)[2]['NMD'])/gene.coverage[np.ix_(sidx, setA)].sum(None)
            nmdB = sum(gene.coverage[np.ix_(sidx, [transcript_id])].sum(None)
                       for transcript_id in setB if gene.transcripts[transcript_id].get('ORF', noORF)[2]['NMD'])/gene.coverage[np.ix_(sidx, setB)].sum(None)
            res.append(tuple(itertools.chain((gene.name, gene.id, gene.chrom, gene.strand, start, end, splice_type, novel, pval,
                                             sorted(setA, key=lambda x: -gene.coverage[np.ix_(sidx, [x])].sum(0)),
                                             sorted(setB, key=lambda x: -gene.coverage[np.ix_(sidx, [x])].sum(0)), nmdA, nmdB),
                                             params, params_other,
                                             (val for lists in zip(x, n) for pair in zip(*lists) for val in pair))))
    colnames = ['gene', 'gene_id', 'chrom', 'strand', 'start', 'end', 'splice_type', 'novel', 'pvalue', 'trA', 'trB', 'nmdA', 'nmdB']
    colnames += [groupname + part for groupname in groupnames[:2] + ['total'] + groupnames[2:] for part in ['_PSI', '_disp']]
    colnames += [f'{sample}_{groupname}_{w}' for groupname, group in zip(groupnames, groups) for sample in group for w in ['in_cov', 'total_cov']]
    df = pd.DataFrame(res, columns=colnames)
    try:
        mask = np.isfinite(df['pvalue'])
        padj = np.empty(mask.shape)
        padj.fill(np.nan)
        padj[mask] = multi.multipletests(df.loc[mask, 'pvalue'], method=padj_method)[1]
        df.insert(8, 'padj', padj)
    except TypeError as e:  # apparently this happens if df is empty...
        logger.error(f'unexpected error during calculation of adjusted p-values: {e}')
    return df


def die_test(self: 'Transcriptome', groups, min_cov=25, n_isoforms=10, padj_method='fdr_bh', **kwargs):
    ''' Reimplementation of the DIE test, suggested by Joglekar et al in Nat Commun 12, 463 (2021):
    "A spatially resolved brain region- and cell type-specific isoform atlas of the postnatal mouse brain"

    Syntax and parameters follow the original implementation in https://github.com/noush-joglekar/scisorseqr/blob/master/inst/RScript/IsoformTest.R

    :param groups: Dict with group names as keys and lists of sample names as values, defining the two groups for the test.
    :param min_cov: Minimal number of reads per group for each gene.
    :param n_isoforms: Number of isoforms to consider in the test for each gene. All additional least expressed isoforms get summarized.
    :param kwargs: Additional keyword arguments are passed to iter_genes.'''

    groupnames, groups, grp_idx = _check_groups(self, groups)
    logger.info('testing differential isoform expression (DIE) for %s.', ' vs '.join(f'{groupnames[i]} ({len(groups[i])})' for i in range(2)))

    result = [(gene.id, gene.name, gene.chrom, gene.strand, gene.start, gene.end) +
              gene.die_test(grp_idx, min_cov, n_isoforms) for gene in self.iter_genes(**kwargs)]
    result = pd.DataFrame(result, columns=['gene_id', 'gene_name', 'chrom', 'strand', 'start', 'end', 'pvalue', 'deltaPI', 'transcript_ids'])
    mask = np.isfinite(result['pvalue'])
    padj = np.empty(mask.shape)
    padj.fill(np.nan)
    padj[mask] = multi.multipletests(result.loc[mask, 'pvalue'], method=padj_method)[1]
    result.insert(6, 'padj', padj)
    return result


def alternative_splicing_events(self, min_total=100, min_alt_fraction=.1, samples=None, **kwargs):
    '''Finds alternative splicing events.

    Finds alternative splicing events and potential transcription start sites/polyA sites
    by searching for splice bubbles in the Segment Graph.
    Genes may be specified by genomic "region", and/or by filter tags / novelty class using the "query" parameters.

    :param min_total: Minimum total coverage over all selected samples.
    :param min_alt_fraction: Minimum fraction of reads supporting the alternative.
    :param samples: Specify the samples to consider. If omitted, all samples are selected.
    :param kwargs: Additional keyword arguments are passed to iter_genes.
    :return: Table with alternative splicing events.'''
    bubbles = []
    if samples is None:
        samples = self.samples
    assert all(s in self.samples for s in samples), 'not all specified samples found'
    sample_dict = {sample: i for i, sample in enumerate(self.samples)}
    sidx = np.array([sample_dict[sample] for sample in samples])

    assert 0 < min_alt_fraction < .5, 'min_alt_fraction must be > 0 and < 0.5'
    for gene in self.iter_genes(**kwargs):
        if gene.coverage[sidx, :].sum() < min_total:
            continue
        known = {}  # check for known events
        if gene.is_annotated and gene.n_transcripts:
            ref_seg_graph: SegmentGraph = gene.ref_segment_graph
            # find annotated alternatives (known)
            for _, _, nX, nY, splice_type in ref_seg_graph.find_splice_bubbles():
                if splice_type in ("TSS", "PAS"):
                    if (splice_type == "TSS") == (gene.strand == "+"):
                        known.setdefault(splice_type, set()).add((ref_seg_graph[nX].end))
                    else:
                        known.setdefault(splice_type, set()).add((ref_seg_graph[nY].start))
                else:
                    known.setdefault(splice_type, set()).add((ref_seg_graph[nX].end, ref_seg_graph[nY].start))
        seg_graph: SegmentGraph = gene.segment_graph
        for setA, setB, nX, nY, splice_type in seg_graph.find_splice_bubbles():
            junction_cov = gene.coverage[np.ix_(sidx, setA)].sum(1)
            total_cov = gene.coverage[np.ix_(sidx, setB)].sum(1) + junction_cov
            if total_cov.sum() >= min_total and min_alt_fraction < junction_cov.sum() / total_cov.sum() < 1 - min_alt_fraction:
                if splice_type in ['TSS', 'PAS']:
                    start, end = seg_graph[nX].start, seg_graph[nY].end
                    if (splice_type == "TSS") == (gene.strand == "+"):
                        novel = end not in known.get(splice_type, set())
                    else:
                        novel = start not in known.get(splice_type, set())
                else:
                    start, end = seg_graph[nX].end, seg_graph[nY].start
                    novel = (start, end) not in known.get(splice_type, set())
                bubbles.append([gene.id, gene.chrom, start, end, splice_type, novel] + list(junction_cov) + list(total_cov))
    return pd.DataFrame(bubbles, columns=['gene', 'chr', 'start', 'end', 'splice_type', 'novel'] +
                        [f'{sample}_{what}' for what in ['in_cov', 'total_cov'] for sample in samples])

# summary tables (can be used as input to plot_bar / plot_dist)

# function to optimize (inverse nbinom cdf)


def _tpm_fun(tpm_th, n_reads, cov_th=2, p=.8):
    return (p-nbinom.cdf(n_reads - cov_th, n=cov_th, p=tpm_th * 1e-6))**2


def estimate_tpm_threshold(n_reads, cov_th=2, p=.8):
    '''Estimate the minimum expression level of observable transcripts at given coverage.

    The function returns the expression level in transcripts per million (TPM), that can be observed
    at the given sequencing depth.

    :param n_reads: The sequencing depth (total number of reads) for the sample.
    :param cov_th: The requested minimum number of reads per transcripts.
    :param p: The probability of a transcript at threshold expression level to be observed.
    '''
    return minimize_scalar(_tpm_fun,  bounds=(.01, 1000), args=(n_reads, cov_th, p))['x']


def altsplice_stats(self: 'Transcriptome', groups=None, weight_by_coverage=True, min_coverage=2, tr_filter={}):
    '''Summary statistics for novel alternative splicing.

    This function counts the novel alternative splicing events of LRTS isoforms with respect to the reference annotation.
    The result can be depicted by isotools.plots.plot_bar.

    :param groups: A dict {group_name:[sample_name_list]} specifying sample groups. If omitted, the samples are analyzed individually.
    :param weight_by_coverage: If True, each transcript is weighted by the coverage.
    :param min_coverage: Threshold to ignore poorly covered transcripts. This parameter gets applied for each sample group separately.
    :param tr_filter: Filter dict, that is passed to self.iter_transcripts().
    :return: Table with numbers of novel alternative splicing events, and suggested parameters for isotools.plots.plot_bar().'''
    weights = dict()
    # if groups is not None:
    #    gi={r:i for i,r in enumerate(runs)}
    #    groups={gn:[gi[r] for r in gr] for gn,gr in groups.items()}
    current = None
    if groups is not None:
        sample_idx = {sample: i for i, sample in enumerate(self.samples)}  # idx
        groups = {groupname: [sample_idx[sample] for sample in group] for groupname, group in groups.items()}

    for gene, transcript_id, transcript in self.iter_transcripts(**tr_filter):
        if gene != current:
            current = gene
            w = gene.coverage.copy() if groups is None else np.array([gene.coverage[grp, :].sum(0) for grp in groups.values()])
            w[w < min_coverage] = 0
            if not weight_by_coverage:
                w[w > 0] = 1
        if 'annotation' not in transcript or transcript['annotation'] is None:
            weights['unknown'] = weights.get('unknown', np.zeros(w.shape[0])) + w[:, transcript_id]
        else:
            for stype in transcript['annotation'][1]:
                weights[stype] = weights.get(stype, np.zeros(w.shape[0])) + w[:, transcript_id]
        weights['total'] = weights.get('total', np.zeros(w.shape[0])) + w[:, transcript_id]

    df = pd.DataFrame(weights, index=self.samples if groups is None else groups.keys()).T
    df = df.reindex(df.mean(1).sort_values(ascending=False).index, axis=0)  # sort by row mean
    if weight_by_coverage:
        title = 'Expressed Transcripts'
        ylab = 'fraction of reads'
    else:
        title = 'Different Transcripts'
        ylab = 'fraction of  different transcripts'
        if min_coverage > 1:
            title += f' > {min_coverage} reads'

    return df, {'ylabel': ylab, 'title': title}


def _check_customised_groups(transcriptome: 'Transcriptome', samples=None, groups=None, sample_idx=False):
    '''
    Check if the samples and all the samples in groups are consistent, and all found in transcriptome.samples.
    Customised group names not in transcriptome.groups() are allowed.

    :param samples: A list of sample names to specify the samples to be considered. If omitted, all samples in transcriptome.samples are selected.
    :param groups: A dict {group_name:[sample_name_list]} or a list of group names to tell how to group samples. If omitted, all the samples are considered as one group.
    :param sample_idx: If True, the samples are specified by sample indices. If False, the samples are specified by sample names.
    :return: A dict {group_name:[sample_list]} with sample names or indices.
    '''

    if samples is None:
        samples = transcriptome.samples
    else:
        assert all(s in transcriptome.samples for s in samples), 'not all specified samples found'
        if isinstance(groups, dict):
            assert list(set(sum(groups.values(), []))) == list(set(samples)), 'inconsistent samples specified in samples and in groups'

    if groups is None:
        group_dict = {'all' if len(samples) == len(transcriptome.samples) else 'selected': samples}
    elif isinstance(groups, dict):
        assert all(s in samples for s in sum(groups.values(), [])), 'not all the samples in specified groups are found'
        group_dict = groups
    elif isinstance(groups, list):
        assert all(gn in transcriptome.groups() for gn in groups), 'not all specified groups are found. To customize groups, use a dict {group_name:[sample_name_list]}'
        group_dict = {gn: [s for s in transcriptome.groups()[gn] if s in samples] for gn in groups if any(s in samples for s in transcriptome.groups()[gn])}
    else:
        raise ValueError('groups must be a dict or a list of group names')

    if sample_idx:
        group_dict = {gn:[transcriptome.samples.index(s) for s in sample_names] for gn, sample_names in group_dict.items()}
    
    return group_dict


def entropy_calculation(self: 'Transcriptome', samples=None, groups=None, min_total=1, relative=True, **kwargs):
    '''
    Calculates the entropy of genes based on the coverage of selected transcripts.

    :param samples: A list of sample names to specify the samples to be considered. If omitted, all samples are selected.
    :param groups: Entropy calculation done by groups of samples. A dict {group_name:[sample_name_list]} or a list of group names. If omitted, all the samples are considered as one group.
    :param min_total: Minimum total coverage of a gene over all the samples in a selected group.
    :param relative: If True, the entropy is normalized by log2 of the number of selected transcripts in the group.
    :param kwargs: Additional keyword arguments are passed to iter_transcripts.
    :return: A table of (relative) entropy of genes based on the coverage of selected transcripts.
    '''

    group_idxs = _check_customised_groups(self, samples, groups, sample_idx=True)

    entropy_tab = pd.DataFrame(columns=['gene_id', 'gene_name'] + [f'{g}_{c}' for g, c in itertools.product(group_idxs, ['ntr', 'rel_entropy' if relative else 'entropy'])])

    for gene, transcript_ids, _ in self.iter_transcripts(genewise=True, **kwargs):
        gene_entropy = [gene.id, gene.name]

        for sample_ids in group_idxs.values():
            cov = gene.coverage[np.ix_(sample_ids, transcript_ids)]
            if cov.sum() < min_total:
                gene_entropy += [np.nan, np.nan]
            else:
                transcript_number = sum(cov.sum(0) > 0)
                group_entropy = -sum(math.log2(p) * p for p in cov.sum(0)[cov.sum(0) > 0] / cov.sum())
                if relative:
                    group_entropy = (group_entropy / math.log2(transcript_number)) if transcript_number > 1 else np.nan
                gene_entropy += [transcript_number, group_entropy]
        
        entropy_tab = pd.concat([entropy_tab, pd.DataFrame([gene_entropy], columns=entropy_tab.columns)], ignore_index=True)
    
    # exclude rows with all empty or NA entries in entropy columns
    entropy_tab.dropna(subset=entropy_tab.columns[2:], how='all', inplace=True)

    return entropy_tab


def str_var_calculation(self: 'Transcriptome', samples=None, groups=None, strict_ec=0, strict_pos=15, count_number=False, **kwargs):
    '''
    Quantify the structural variation of genes based on selected transcripts.
    Structural variation includes (and in the same order of) distinct TSS positions, exon chains, and PAS positions.
    
    :param samples: A list of sample names to specify the samples to be considered. If omitted, all samples are selected.
    :param groups: Quantification done by groups of samples. A dict {group_name:[sample_name_list]} or a list of group names. If omitted, all the samples are considered as one group.
    :param strict_ec: Distance allowed between each position, except for the first/last, in two exon chains so that they can be considered as identical.
    :param strict_pos: Difference allowed between two positions when considering identical TSS/PAS.
    :param count_number: By default False. If True, the number of distinct TSSs, exon chains and PASs in genes directly.
    :param kwargs: Additional keyword arguments are passed to iter_transcripts.
    :return: A table of structural variation of genes based on selected transcripts, including: gene_id, gene_name, and the variation of TSS, exon chain, and PAS for each group of samples.
    '''

    group_sns = _check_customised_groups(self, samples, groups, sample_idx=False)

    str_var_tab = pd.DataFrame(columns=['gene_id', 'gene_name'] + [f'{g}_{c}' for g, c in itertools.product(group_sns, ['tss', 'ec', 'pas'])])

    for gene, _, selected_trs in self.iter_transcripts(genewise=True, **kwargs):
        gene_str_var = [gene.id, gene.name]

        for _, selected_samples in group_sns.items():
            group_var = str_var_triplet(transcripts=selected_trs, samples=selected_samples, strict_ec=strict_ec, strict_pos=strict_pos)

            if not count_number:
                # regress out the variation caused by TAS and PAS for exon chain
                splicing_ratio = 2 * group_var[1] / (group_var[0] + group_var[2]) if (group_var[0] > 0 and group_var[2] > 0) else 0
                ratio_triplet = [group_var[0], splicing_ratio, group_var[2]]
                # normalize to the sum of 1
                group_var = [n / sum(ratio_triplet) for n in ratio_triplet] if sum(ratio_triplet) > 0 else [0, 0, 0]
            gene_str_var += group_var
        
        str_var_tab = pd.concat([str_var_tab, pd.DataFrame([gene_str_var], columns=str_var_tab.columns)], ignore_index=True)

    # replace 0 with nan, and remove rows with all nan
    str_var_tab = str_var_tab.replace(0, np.nan)
    str_var_tab = str_var_tab.dropna(how='all', subset=str_var_tab.columns[2:])

    return str_var_tab


def filter_stats(self: 'Transcriptome', tags=None, groups=None, weight_by_coverage=True, min_coverage=2, **kwargs):
    '''Summary statistics for filter flags.

    This function counts the number of transcripts corresponding to filter tags.
    The result can be depicted by isotools.plots.plot_bar.

    :param tags: The filter tags to be evaluated. If omitted, all transcript tags are selected.
    :param groups: A dict {group_name:[sample_name_list]} specifying sample groups. If omitted, the samples are analyzed individually.
    :param weight_by_coverage: If True, each transcript is weighted by the number of supporting reads.
    :param min_coverage: Coverage threshold per sample to ignore poorly covered transcripts.
    :param kwargs: Additional parameters are passed to self.iter_transcripts().
    :return: Table with numbers of transcripts featuring the filter tag, and suggested parameters for isotools.plots.plot_bar().'''

    weights = dict()
    if tags is None:
        tags = list(self.filter['transcript'])
    assert all(t in self.filter['transcript'] for t in tags), '"Tags" contains invalid tags'
    filterfun = {tag: _filter_function(tag, self.filter['transcript'])[0] for tag in tags}
    if groups is not None:
        sample_indices = {sample: i for i, sample in enumerate(self.samples)}  # idx
        groups = {group_name: [sample_indices[sample] for sample in sample_group] for group_name, sample_group in groups.items()}
    current = None
    for gene, transcript_id, transcript in self.iter_transcripts(**kwargs):
        if gene != current:
            current = gene
            weight = gene.coverage.copy() if groups is None else np.array([gene.coverage[group, :].sum(0) for group in groups.values()])
            weight[weight < min_coverage] = 0
            if not weight_by_coverage:
                weight[weight > 0] = 1
        # relevant_filter=[filter for filter in transcript['filter'] if  consider is None or filter in consider]
        relevant_filter = [tag for tag in tags if filterfun[tag](gene=gene, transcript_id=transcript_id, **transcript)]
        for filter in relevant_filter:
            weights[filter] = weights.get(filter, np.zeros(weight.shape[0])) + weight[:, transcript_id]
        if not relevant_filter:
            weights['PASS'] = weights.get('PASS', np.zeros(weight.shape[0])) + weight[:, transcript_id]
        weights['total'] = weights.get('total', np.zeros(weight.shape[0])) + weight[:, transcript_id]

    df = pd.DataFrame(weights, index=self.samples if groups is None else groups.keys()).T

    df = df.reindex(df.mean(1).sort_values(ascending=False).index, axis=0)
    ylab = 'fraction of reads' if weight_by_coverage else 'fraction of different transcripts'
    if weight_by_coverage:
        title = 'Expressed Transcripts'
    else:
        title = 'Different Transcripts'
    if min_coverage > 1:
        title += f' > {min_coverage} reads'
    return df, {'ylabel': ylab, 'title': title}


def transcript_length_hist(self: 'Transcriptome', groups=None, add_reference=False, bins=50, x_range=(
        0, 10000), weight_by_coverage=True, min_coverage=2, use_alignment=True, tr_filter={}, ref_filter={}):
    '''Retrieves the transcript length distribution.

    This function counts the number of transcripts within length interval.
    The result can be depicted by isotools.plots.plot_dist.

    :param groups: A dict {group_name:[sample_name_list]} specifying sample groups. If omitted, the samples are analyzed individually.
    :param add_reference: Add the transcript length distribution of the reference annotation.
    :param bins: Define the length interval, either by a single number of bins, or by a list of lengths, defining the interval boundaries.
    :param x_range: The range of the intervals. Ignored if "bins" is provided as a list.
    :param weight_by_coverage: If True, each transcript is weighted by the coverage.
    :param min_coverage: Threshold to ignore poorly covered transcripts.
    :param use_alignment: use the transcript length as defined by the alignment (e.g. the sum of all exon lengths).
    :param tr_filter: Filter dict, that is passed to self.iter_transcripts().
    :param ref_filter: Filter dict, that is passed to self.iter_ref_transcripts() (relevant only if add_reference=True).
    :return: Table with numbers of transcripts within the length intervals, and suggested parameters for isotools.plots.plot_distr().'''

    trlen = []
    cov = []
    current = None
    for gene, transcript_id, transcript in self.iter_transcripts(**tr_filter):
        if gene != current:
            current = gene
            current_cov = gene.coverage
        cov.append(current_cov[:, transcript_id])
        trlen.append(sum(e[1] - e[0] for e in transcript['exons']) if use_alignment else transcript['source_len'])  # source_len is not set in the current version
    cov = pd.DataFrame(cov, columns=self.samples)
    if groups is not None:
        cov = pd.DataFrame({grn: cov[group].sum(1) for grn, group in groups.items()})
    if isinstance(bins, int):
        bins = np.linspace(x_range[0] - .5, x_range[1] - .5, bins + 1)
    cov[cov < min_coverage] = 0
    if not weight_by_coverage:
        cov[cov > 0] = 1
    counts = pd.DataFrame({gn: np.histogram(trlen, weights=g_cov, bins=bins)[0] for gn, g_cov in cov.items()})
    if add_reference:
        ref_len = [sum(exon[1] - exon[0] for exon in transcript['exons']) for _, _, transcript in self.iter_ref_transcripts(**ref_filter)]
        counts['reference'] = np.histogram(ref_len, bins=bins)[0]
    bin_df = pd.DataFrame({'from': bins[:-1], 'to': bins[1:]})
    params = dict(yscale='linear', title='transcript length', xlabel='transcript length [bp]', density=True)
    return pd.concat([bin_df, counts], axis=1).set_index(['from', 'to']), params


def transcript_coverage_hist(self, groups=None, bins=50, x_range=(1, 1001), tr_filter={}):
    '''Retrieves the transcript coverage distribution.

    This function counts the number of transcripts within coverage interval.
    The result can be depicted by isotools.plots.plot_dist.

    :param groups: A dict {group_name:[sample_name_list]} specifying sample groups. If omitted, the samples are analyzed individually.
    :param bins: Define the coverage interval, either by a single number of bins, or by a list of values, defining the interval boundaries.
    :param x_range: The range of the intervals. Ignored if "bins" is provided as a list.
    :param tr_filter: Filter dict, that is passed to self.iter_transcripts().
    :return: Table with numbers of transcripts within the coverage intervals, and suggested parameters for isotools.plots.plot_distr().'''
    # get the transcript coverage in bins for groups
    # return count dataframe and suggested default parameters for plot_distr
    cov = []
    current = None
    for gene, transcript_id, _ in self.iter_transcripts(**tr_filter):
        if gene != current:
            current = gene
            current_cov = gene.coverage
        cov.append(current_cov[:, transcript_id])
    cov = pd.DataFrame(cov, columns=self.samples)
    if groups is not None:
        cov = pd.DataFrame({grn: cov[grp].sum(1) for grn, grp in groups.items()})
    if isinstance(bins, int):
        bins = np.linspace(x_range[0] - .5, x_range[1] - .5, bins + 1)
    counts = pd.DataFrame({gn: np.histogram(g_cov, bins=bins)[0] for gn, g_cov in cov.items()})
    bin_df = pd.DataFrame({'from': bins[:-1], 'to': bins[1:]})
    params = dict(yscale='log', title='transcript coverage', xlabel='reads per transcript')
    return pd.concat([bin_df, counts], axis=1).set_index(['from', 'to']), params
    # plot histogram
    # cov.mask(cov.lt(x_range[0]) | cov.gt(x_range[1])).plot.hist(ax=ax, alpha=0.5, bins=n_bins)
    # ax=counts.plot.bar()
    # ax.plot(x, counts)


def transcripts_per_gene_hist(self, groups=None, add_reference=False, bins=49, x_range=(1, 50), min_coverage=2, tr_filter={}, ref_filter={}):
    '''Retrieves the histogram of number of transcripts per gene.

    This function counts the genes featuring transcript numbers within specified intervals.
    The result can be depicted by isotools.plots.plot_dist.

    :param groups: A dict {group_name:[sample_name_list]} specifying sample groups. If omitted, the samples are analyzed individually.
    :param add_reference: Add the transcript per gene histogram of the reference annotation.
    :param bins: Define the intervals, either by a single number of bins, or by a list of values, defining the interval boundaries.
    :param x_range: The range of the intervals. Ignored if "bins" is provided as a list.
    :param min_coverage: Threshold to ignore poorly covered transcripts.
    :param tr_filter: Filter dict, that is passed to self.iter_transcripts().
    :param ref_filter: Filter dict, that is passed to self.iter_ref_transcripts() (relevant only if add_reference=True).
    :return: Table with numbers of genes featuring transcript numbers within the specified intervals,
        and suggested parameters for isotools.plots.plot_distr().'''
    ntr = []
    current = None
    if groups is None:
        group_names = self.samples
    else:
        group_names = groups.keys()
        sidx = {sample: i for i, sample in enumerate(self.samples)}  # idx
        groups = {groupname: [sidx[sample] for sample in group] for groupname, group in groups.items()}
    n_sa = len(group_names)
    for gene, transcript_id, _ in self.iter_transcripts(**tr_filter):
        if gene != current:
            current = gene
            current_cov = gene.coverage if groups is None else np.array([gene.coverage[grp, :].sum(0) for grp in groups.values()])
            ntr.append(np.zeros(n_sa))
        ntr[-1] += current_cov[:, transcript_id] >= min_coverage

    ntr = pd.DataFrame((n for n in ntr if n.sum() > 0), columns=group_names)
    if isinstance(bins, int):
        bins = np.linspace(x_range[0] - .5, x_range[1] - .5, bins + 1)
    counts = pd.DataFrame({gn: np.histogram(n, bins=bins)[0] for gn, n in ntr.items()})
    if add_reference:
        if ref_filter:
            logger.warning('reference filter not implemented')
        ref_ntr = [gene.n_ref_transcripts for gene in self]  # todo: add reference filter
        counts['reference'] = np.histogram(ref_ntr, bins=bins)[0]
    bin_df = pd.DataFrame({'from': bins[:-1], 'to': bins[1:]})
    sub = f'counting transcripts covered by >= {min_coverage} reads'
    if 'query' in tr_filter:
        sub += f', filter query: {tr_filter["query"]}'
    params = dict(yscale='log', title='transcript per gene\n' + sub, xlabel='transcript per gene')
    return pd.concat([bin_df, counts], axis=1).set_index(['from', 'to']), params


def exons_per_transcript_hist(self, groups=None, add_reference=False, bins=34, x_range=(1, 69),
                              weight_by_coverage=True, min_coverage=2, tr_filter={}, ref_filter={}):
    '''Retrieves the histogram of number of exons per transcript.

    This function counts the transcripts featuring exon numbers within specified intervals.
    The result can be depicted by isotools.plots.plot_dist.

    :param groups: A dict {group_name:[sample_name_list]} specifying sample groups. If omitted, the samples are analyzed individually.
    :param add_reference: Add the exons per transcript histogram of the reference annotation.
    :param bins: Define the intervals, either by a single number of bins, or by a list of values, defining the interval boundaries.
    :param x_range: The range of the intervals. Ignored if "bins" is provided as a list.
    :param weight_by_coverage: If True, each transcript is weighted by the coverage.
    :param min_coverage: Threshold to ignore poorly covered transcripts.
    :param tr_filter: Filter dict, that is passed to self.iter_transcripts().
    :param ref_filter: Filter dict, that is passed to self.iter_ref_transcripts() (relevant only if add_reference=True).
    :return: Table with numbers of transcripts featuring exon numbers within the specified intervals,
        and suggested parameters for isotools.plots.plot_distr().'''
    n_exons = []
    cov = []
    current = None
    for gene, transcript_id, transcript in self.iter_transcripts(**tr_filter):
        if gene != current:
            current = gene
            current_cov = gene.coverage
        cov.append(current_cov[:, transcript_id])
        n_exons.append(len(transcript['exons']))
    cov = pd.DataFrame(cov, columns=self.samples)
    if groups is not None:
        cov = pd.DataFrame({grn: cov[grp].sum(1) for grn, grp in groups.items()})
    if isinstance(bins, int):
        bins = np.linspace(x_range[0] - .5, x_range[1] - .5, bins + 1)
    cov[cov < min_coverage] = 0
    if not weight_by_coverage:
        cov[cov > 0] = 1
    counts = pd.DataFrame({gn: np.histogram(n_exons, weights=g_cov, bins=bins)[0] for gn, g_cov in cov.items()})
    if add_reference:
        ref_n_exons = [len(transcript['exons']) for _, _, transcript in self.iter_ref_transcripts(**ref_filter)]
        counts['reference'] = np.histogram(ref_n_exons, bins=bins)[0]
    bin_df = pd.DataFrame({'from': bins[:-1], 'to': bins[1:]})
    sub = f'counting transcripts covered by >= {min_coverage} reads'
    if 'query' in tr_filter:
        sub += f', filter query: {tr_filter["query"]}'
    params = dict(yscale='log', title='exons per transcript\n' + sub, xlabel='number of exons per transcript')
    return pd.concat([bin_df, counts], axis=1).set_index(['from', 'to']), params


def downstream_a_hist(self, groups=None, add_reference=False, bins=30, x_range=(0, 1), weight_by_coverage=True, min_coverage=2, transcript_filter={}, ref_filter={}):
    '''Retrieves the distribution of downstream adenosine content.

    High downstream adenosine content is indicative for internal priming.

    :param groups: A dict {group_name:[sample_name_list]} specifying sample groups. If omitted, the samples are analyzed individually.
    :param add_reference: Add the distribution of downstream adenosine content of the reference annotation.
    :param bins: Define the intervals, either by a single number of bins, or by a list of values, defining the interval boundaries.
    :param x_range: The range of the intervals. Ignored if "bins" is provided as a list. Should not exceed (0,1), e.g. 0 to 100%.
    :param weight_by_coverage: If True, each transcript is weighted by the coverage.
    :param min_coverage: Threshold to ignore poorly covered transcripts.
    :param tr_filter: Filter dict, that is passed to self.iter_transcripts().
    :param ref_filter: Filter dict, that is passed to self.iter_ref_transcripts() (relevant only if add_reference=True).
    :return: Table with downstream adenosine content distribution, and suggested parameters for isotools.plots.plot_distr().'''
    acontent = []
    cov = []
    current = None
    for gene, transcript_id, transcript in self.iter_transcripts(**transcript_filter):
        if gene != current:
            current = gene
            current_cov = gene.coverage
        cov.append(current_cov[:, transcript_id])
        try:
            acontent.append(transcript['downstream_A_content'])
        except KeyError:
            acontent.append(-1)
    cov = pd.DataFrame(cov, columns=self.samples)
    if groups is not None:
        cov = pd.DataFrame({grn: cov[grp].sum(1) for grn, grp in groups.items()})
    if isinstance(bins, int):
        bins = np.linspace(x_range[0], x_range[1], bins + 1)
    cov[cov < min_coverage] = 0
    if not weight_by_coverage:
        cov[cov > 0] = 1
    counts = pd.DataFrame({group_name: np.histogram(acontent, weights=group_cov, bins=bins)[0] for group_name, group_cov in cov.items()})
    if add_reference:
        ref_acontent = [transcript['downstream_A_content'] for _, _, transcript in self.iter_ref_transcripts(**ref_filter) if 'downstream_A_content' in transcript]
        counts['reference'] = np.histogram(ref_acontent, bins=bins)[0]
    bin_df = pd.DataFrame({'from': bins[:-1], 'to': bins[1:]})
    params = dict(title='downstream genomic A content', xlabel='fraction of A downstream the transcript')
    return pd.concat([bin_df, counts], axis=1).set_index(['from', 'to']), params


def direct_repeat_hist(self, groups=None, bins=10, x_range=(0, 10), weight_by_coverage=True, min_coverage=2, tr_filter={}):
    '''Retrieves the distribution direct repeat length at splice junctions.

    Direct repeats are indicative for reverse transcriptase template switching.

    :param groups: A dict {group_name:[sample_name_list]} specifying sample groups. If omitted, the samples are analyzed individually.
    :param bins: Define the intervals, either by a single number of bins, or by a list of values, defining the interval boundaries.
    :param x_range: The range of the intervals. Ignored if "bins" is provided as a list.
    :param weight_by_coverage: If True, each transcript is weighted by the coverage.
    :param min_coverage: Threshold to ignore poorly covered transcripts.
    :param tr_filter: Filter dict, that is passed to self.iter_transcripts().
    :return: Table with direct repeat length distribution, and suggested parameters for isotools.plots.plot_distr().'''
    # find the direct repeat length distribution in FSM transcripts and putative RTTS
    # putative RTTS are identified by introns where both splice sites are novel but within annotated exons
    # TODO: actually no need to check annotation, could simply use filter flags (or the definition from the filter flags, which should be faster)
    rl = {cat: [] for cat in ('known', 'novel canonical', 'novel noncanonical')}
    for gene, transcript_id, transcript in self.iter_transcripts(**tr_filter):
        if 'annotation' in transcript and transcript['annotation'][0] == 0:  # e.g. FSM
            rl['known'].extend((drl, gene.coverage[:, transcript_id]) for drl in transcript['direct_repeat_len'])
        elif gene.is_annotated and 'novel_splice_sites' in transcript:
            novel_junction = [i // 2 for i in transcript['novel_splice_sites'] if i % 2 == 0 and i + 1 in transcript['novel_splice_sites']]
            nc = {v[0] for v in transcript.get('noncanonical_splicing', [])}
            rl['novel noncanonical'].extend((transcript['direct_repeat_len'][sj], gene.coverage[:, transcript_id]) for sj in novel_junction if sj in nc)
            rl['novel canonical'].extend((transcript['direct_repeat_len'][sj], gene.coverage[:, transcript_id]) for sj in novel_junction if sj not in nc)

    rl_cov = {cat: pd.DataFrame((v[1] for v in rl[cat]), columns=self.samples) for cat in rl}
    if groups is not None:
        rl_cov = {cat: pd.DataFrame({grn: rl_cov[cat][grp].sum(1) for grn, grp in groups.items()}) for cat in rl_cov}
    for cov_df in rl_cov.values():
        cov_df[cov_df < min_coverage] = 0
        if not weight_by_coverage:
            cov_df[cov_df > 0] = 1
    if isinstance(bins, int):
        bins = np.linspace(x_range[0] - .5, x_range[1] - .5, bins + 1)
    counts = pd.DataFrame({f'{sample} {cat}': np.histogram([val[0] for val in rl_list], weights=rl_cov[cat][sample], bins=bins)[
                          0] for cat, rl_list in rl.items() for sample in (self.samples if groups is None else groups)})

    bin_df = pd.DataFrame({'from': bins[:-1], 'to': bins[1:]})
    params = dict(title='direct repeat length', xlabel='length of direct repeats at splice junctons', ylabel='# transcripts')

    return pd.concat([bin_df, counts], axis=1).set_index(['from', 'to']), params


def rarefaction(self, groups=None, fractions=20, min_coverage=2, tr_filter={}):
    '''Rarefaction analysis

    Reads are sub-sampled according to the provided fractions, to estimate saturation of the transcriptome.

    :param groups: A dict {group_name:[sample_name_list]} specifying sample groups. If omitted, the samples are analyzed individually.
    :param fractions: the fractions of reads to be sub-sampled.
        Either a list of floats between 0 and 1, or a integer number, specifying the number of equally spaced fractions.
    :param min_coverage: Number of reads per transcript required to consider the transcribed discovered.
    :param tr_filter: Filter dict, that is passed to self.iter_transcripts().
    :return: Tuple with:
        1) Data frame containing the number of discovered transcripts, for each sub-sampling fraction and each sample / sample group.
        2) Dict with total number of reads for each group. '''

    cov = []
    current = None
    if isinstance(fractions, int):
        fractions = np.linspace(1 / fractions, 1, fractions)
    for gene, transcript_id, _ in self.iter_transcripts(**tr_filter):
        if gene != current:
            current = gene
            current_cov = gene.coverage
        cov.append(current_cov[:, transcript_id])
    cov = pd.DataFrame(cov, columns=self.samples)
    total = dict(self.sample_table.set_index('name').nonchimeric_reads)
    if groups is not None:
        cov = pd.DataFrame({grn: cov[grp].sum(1) for grn, grp in groups.items()})
        total = {groupname: sum(n for sample, n in total.items() if sample in group) for groupname, group in groups.items()}
    curves = {}
    for sample in cov:
        curves[sample] = [(np.random.binomial(n=cov[sample], p=th) >= min_coverage).sum() for th in fractions]
    return pd.DataFrame(curves, index=fractions), total


def coordination_test(self: 'Transcriptome', samples=None, test: Literal['fisher', 'chi2'] = "fisher", min_dist_AB=1, min_dist_events=1, min_total=100, min_alt_fraction=.1,
                      events_dict=None, event_type: list[ASEType] = ("ES", "5AS", "3AS", "IR", "ME"), padj_method="fdr_bh",
                      transcript_filter: Optional[str] = None, **kwargs) -> pd.DataFrame:
    '''Performs gene_coordination_test on all genes.

    :param samples: Specify the samples that should be considered in the test.
        The samples can be provided either as a single group name, a list of sample names, or a list of sample indices.
    :param test: Test to be performed. One of ("chi2", "fisher")
    :param min_dist_AB: Minimum distance (in nucleotides) between node A and B in an event
    :param min_dist_events: Minimum number of nucleotides between the end of the first event and the start of the second event in each tested pair of events
    :param min_total: The minimum total number of reads for an event to pass the filter
    :type min_total: int
    :param min_alt_fraction: The minimum fraction of read supporting the alternative
    :type min_alt_frction: float
    :param min_cov_pair: the minimum total number of a pair of the joint occurrence of a pair of event for it to be reported in the result
    :type min_cov_pair: int
    :param events_dict: Pre-computed dictionary of alternative splicing events, to speed up analysis of several groups of samples of the same data set.
        Can be generated with the function _utils.precompute_events_dict.
    :param event_type: A tuple with event types to test. Valid types are "ES", "3AS", "5AS", "IR", "ME", "TSS" and "PAS".
    :param padj_method: The multiple test adjustment method.
        Any value allowed by statsmodels.stats.multitest.multipletests (default: Benjamini-Hochberg)
    :param kwargs: Additional keyword arguments are passed to iter_genes.

    :return: a Pandas DataFrame, where each column corresponds to the p_values, the statistics
        (the chi squared statistic if the chi squared test is used and the odds-ratio if the Fisher
        test is used), the log2 OR, the gene Id, the gene name, the type of the first ASE, the type of the second ASE, the
        starting coordinate of the first ASE, the ending coordinate of the first ASE, the starting
        coordinate of the second ASE, the ending coordinate of the second ASE,
        and the four entries of the contingency table.'''

    test_res = []

    if samples is not None:
        _, _, groups = _check_groups(self, [samples], 1)
        samples = groups[0]

    for gene in self.iter_genes(**kwargs):
        events = events_dict.get(gene.id, []) if events_dict is not None else None
        try:
            next_test_res = gene.coordination_test(
                test=test,
                samples=samples,
                min_dist_AB=min_dist_AB,
                min_dist_events=min_dist_events,
                min_total=min_total,
                min_alt_fraction=min_alt_fraction,
                events=events,
                event_type=event_type,
                transcript_filter=transcript_filter,
            )
            test_res.extend(next_test_res)

        except Exception as e:
            logger.error(f"\nError encountered on {print(gene)} {gene.id}   :  {gene.name}.")
            raise e

    col_names = ("gene_id", "gene_name", "strand", "eventA_type", "eventB_type", "eventA_start", "eventA_end",
                 "eventB_start", "eventB_end", "pvalue", "statistic", "log2OR", "dcPSI_AB", "dcPSI_BA", "priA_priB", "priA_altB", "altA_priB",
                 "altA_altB", "priA_priB_transcript_ids", "priA_altB_transcript_ids", "altA_priB_transcript_ids", "altA_altB_transcript_ids")

    res = pd.DataFrame(test_res, columns=col_names)

    adj_p_value = multi.multipletests(res.pvalue, method=padj_method)[1] if len(res.pvalue) > 0 else []

    res.insert(10, "padj", adj_p_value)

    return res
