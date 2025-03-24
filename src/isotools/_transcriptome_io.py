from __future__ import annotations
import numpy as np
# from numpy.lib.function_base import percentile, quantile
import pandas as pd
from os import path
from intervaltree import IntervalTree, Interval
from collections.abc import Iterable
from collections import Counter, defaultdict
from pysam import TabixFile, AlignmentFile, FastaFile
from tqdm import tqdm
from contextlib import ExitStack
from .short_read import Coverage
from typing import Tuple, TYPE_CHECKING
from ._utils import junctions_from_cigar, splice_identical, is_same_gene, has_overlap, get_overlap, pairwise, \
    cigar_string2tuples, rc, get_intersects, _find_splice_sites, _get_overlap, get_quantiles  # , _get_exonic_region
from .gene import Gene, Transcript
from .decorators import experimental
import logging
import gzip as gziplib
from ._transcriptome_filter import SPLICE_CATEGORY
import math

if TYPE_CHECKING:
    from .transcriptome import Transcriptome

logger = logging.getLogger('isotools')

# io functions for the transcriptome class


def add_short_read_coverage(self: Transcriptome, bam_files, load=False):
    '''Adds short read coverage to the genes.

    By default (e.g. if load==False), this method does not actually read the bams,
    but import for each gene is done at first access.

    :param bam_files: A dict with the sample names as keys, and the path to aligned short reads in bam format as values.
    :param load: If True, the coverage of all genes is imported. WARNING: this may take a long time.'''
    self.infos.setdefault('short_reads', pd.DataFrame(columns=['name', 'file'], dtype='object'))

    bam_files = {k: v for k, v in bam_files.items() if k not in self.infos['short_reads']['name']}
    self.infos['short_reads'] = pd.concat([self.infos['short_reads'],
                                           pd.DataFrame({'name': bam_files.keys(), 'file': bam_files.values()})],
                                          ignore_index=True)
    if load:  # when loading coverage for all genes keep the filehandle open, hopefully a bit faster
        for i, bamfile in enumerate(self.infos['short_reads'].file):
            logger.info('Adding short read coverag from %s', bamfile)
            with AlignmentFile(bamfile, "rb") as align:
                for gene in tqdm(self):
                    gene.data.setdefault('short_reads', list())
                    if len(gene.data['short_reads']) == i:
                        gene.data['short_reads'].append(Coverage.from_alignment(align, gene))


def remove_short_read_coverage(self: Transcriptome):
    '''Removes short read coverage.

    Removes all short read coverage information from self.'''

    if 'short_reads' in self.infos:
        del self.infos['short_reads']
        for gene in self:
            if 'short_reads' in gene:
                del self.data['short_reads']
    else:
        logger.warning('No short read coverage to remove')


@experimental
def remove_samples(self: Transcriptome, sample_names):
    ''' Removes samples from the dataset.

    :params sample_names: A list of sample names to remove.'''

    if isinstance(sample_names, str):
        sample_names = [sample_names]
    assert all(s in self.samples for s in sample_names), 'Did not find all samples to remvoe in dataset'
    sample_table = self.sample_table
    rm_idx = sample_table.index[sample_table.name.isin(sample_names)]
    sample_table = sample_table.drop(index=sample_table.index[rm_idx])
    for gene in self:
        remove_transcript_ids = []
        for i, transcript in enumerate(gene.transcripts):
            if any(s in transcript['coverage'] for s in sample_names):
                transcript['coverage'] = {s: cov for s, cov in transcript['coverage'].items() if s not in sample_names}
                if not transcript['coverage']:
                    remove_transcript_ids.append(i)
        if remove_transcript_ids:  # remove the transcripts that is not expressed by remaining samples
            gene.data['transcripts'] = [transcript for i, transcript in enumerate(gene.transcripts) if i not in remove_transcript_ids]
        gene.data['segment_graph'] = None  # gets recomputed on next request
        gene.data['coverage'] = None


def add_sample_from_csv(self: Transcriptome, coverage_csv_file, transcripts_file, transcript_id_col=None, sample_cov_cols=None,
                        sample_properties=None, add_chromosomes=True, infer_genes=False, reconstruct_genes=True, fuzzy_junction=0,
                        min_exonic_ref_coverage=.25, sep='\t'):
    '''Imports expressed transcripts from coverage table and gtf/gff file, and adds it to the 'Transcriptome' object.

    Transcript to gene assignment is either taken from the transcript_file, or recreated,
    as specified by the reconstruct_genes parameter.
    In the first case, the genes are then matched to overlapping genes from the reference annotation by gene id.
    In absence of a overlapping gene with same id, the gene is matched by splice junction, and renamed.
    A map reflecting the the renaming is returned as a dictionary.

    :param coverage_csv_file: The name of the csv file with coverage information for all samples to be added.
        The file contains columns unique ids for the transcripts, and one column with the coverage for each sample
    :param transcripts_file: Transcripts to be added to the 'Transcriptome' object, in gtf or gff/gff3 format.
        Gene and transcript ids should correspond to ids provided in the coverage_csv_file.
    :param transcript_id_col: Column name with the transcript ids.
        Alternatively, a list of column names can be provided, in which case the transcript id is concatenated from the provided columns,
        separated by an underscore ("_"). If not specified, checks for columns 'transcript_id' or ['gene_id', 'transcript_nr'].
    :param sample_cov_cols: Dict with sample names for the new samples as keys, and corresponding coverage column names in coverage_csv_file as values.
        If not specified, a new sample is added for each <name>_coverage column.
    :param sample_properties: Additional properties of the samples, that get added to the sample table, and can be used to group or stratify the samples.
        Can be provided either as a dict with sample names as keys, and the respective properties dicts as the values,
        or as a data frame with a column "name" or with the sample names in the index, and the properties in the additional columns.
    :param add_chromosomes: If True, genes from chromosomes which are not in the Transcriptome yet are added.
    :param infer_genes: If True, gene structure is inferred from the transcripts. Useful for gtf files without gene information.
    :param reconstruct_genes: If True, transcript gene assignment from gtf is ignored, and transcripts are grouped to genes from scratch.
    :param min_exonic_ref_coverage: Minimal fraction of exonic overlap to assign to reference transcript if no splice junctions match.
        Also applies to mono-exonic transcripts
    :param progress_bar: Show the progress.
    :param sep: Specify the seperator for the coverage_csv_file.
    :return: Dict with map of renamed gene ids.
'''

    cov_tab = pd.read_csv(coverage_csv_file, sep=sep)
    if sample_cov_cols is None:
        sample_cov_cols = {c.replace('_coverage', ''): c for c in cov_tab.columns if '_coverage' in c}
    else:
        assert all(c in cov_tab for c in sample_cov_cols.values()), 'coverage cols missing in %s: %s' % (
            coverage_csv_file, ', '.join(c for c in sample_cov_cols.values() if c not in cov_tab))
    samples = list(sample_cov_cols)

    if transcript_id_col is None:
        if 'transcript_id' in cov_tab.columns:
            pass
        elif 'gene_id' in cov_tab.columns and 'transcript_nr' in cov_tab.columns:
            cov_tab['transcript_id'] = cov_tab.gene_id+'_'+cov_tab.transcript_nr.astype(str)
        else:
            raise ValueError('"transcript_id_col" not specified, and coverage table does not contain "transcript_id", nor "gene_id" and "transcript_nr"')
    elif isinstance(transcript_id_col, list):
        assert all(c in cov_tab for c in transcript_id_col), 'missing specified transcript_id_col'
        cov_tab['transcript_id'] = ['_'.join(str(v) for v in row) for _, row in cov_tab[transcript_id_col].iterrows()]
    else:
        assert transcript_id_col in cov_tab, 'missing specified transcript_id_col'
        cov_tab['transcript_id'] = cov_tab[transcript_id_col]
    # could be optimized, but code is easier when the id column always is transcript_id
    transcript_id_col = 'transcript_id'

    known_sa = set(samples).intersection(self.samples)
    assert not known_sa, 'Attempt to add known samples: %s' % known_sa
    # cov_tab.set_index('transcript_id')
    # assert cov_tab.index.is_unique, 'ambigous transcript ids in %s' % coverage_csv_file
    # check sample properties
    if sample_properties is None:
        sample_properties = {sample: {'group': sample} for sample in samples}
    elif isinstance(sample_properties, pd.DataFrame):
        if 'name' in sample_properties:
            sample_properties = sample_properties.set_index('name')
        sample_properties = {sample: {k: v for k, v in row.items() if k not in {'file', 'nonchimeric_reads', 'chimeric_reads'}}
                             for sample, row in sample_properties.iterrows()}
    assert all(sample in sample_properties for sample in samples), 'missing sample_properties for samples %s' % ', '.join(
        (sample for sample in samples if sample not in sample_properties))
    for sample in sample_properties:
        sample_properties[sample].setdefault('group', sample)

    logger.info('adding samples "%s" from csv', '", "'.join(samples))
    # consider chromosomes not in the reference?
    if add_chromosomes:
        chromosomes = None
    else:
        chromosomes = self.chromosomes

    logger.info('importing transcripts from %s. Please note transcripts with missing annotations will be skipped.', transcripts_file)
    file_format = path.splitext(transcripts_file)[1].lstrip('.')
    if file_format == 'gz':
        file_format = path.splitext(transcripts_file[:-3])[1].lstrip('.')
    if file_format == 'gtf':
        exons, transcripts, gene_infos, cds_start, cds_stop, skipped = _read_gtf_file(transcripts_file, chromosomes=chromosomes, infer_genes=infer_genes)
    elif file_format in ('gff', 'gff3'):  # gff/gff3
        exons, transcripts, gene_infos, cds_start, cds_stop, skipped = _read_gff_file(transcripts_file, chromosomes=chromosomes, infer_genes=infer_genes)
    else:
        logger.warning('unknown file format %s of the transcriptome file', file_format)

    logger.debug('sorting exon positions...')
    for tid in exons:
        exons[tid].sort()

    logger.info('adding coverage information for transcripts imported.')
    if skipped:
        cov_tab = cov_tab[~cov_tab['transcript_id'].isin(skipped['transcript'])]

    if 'gene_id' not in cov_tab:
        gene_id_dict = {tid: gid for gid, tids in transcripts.items() for tid in tids}
        try:
            cov_tab['gene_id'] = [gene_id_dict[tid] for tid in cov_tab.transcript_id]
        except KeyError as e:
            logger.warning('transcript_id %s from csv file not found in gtf.' % e.args[0])
    if 'chr' not in cov_tab:
        chrom_dict = {gid: chrom for chrom, gids in gene_infos.items() for gid in gids}
        try:
            cov_tab['chr'] = [chrom_dict[gid] for gid in cov_tab.gene_id]
        except KeyError as e:
            logger.warning('gene_id %s from csv file not found in gtf.', e.args[0])

    used_transcripts = set()
    for _, row in cov_tab.iterrows():
        if chromosomes is not None and row.chr not in chromosomes:
            continue
        if all(row[c] == 0 for c in sample_cov_cols.values()):
            continue
        try:
            assert row.transcript_id not in used_transcripts
            transcript = transcripts[row.gene_id][row.transcript_id]
            transcript['transcript_id'] = row.transcript_id
            transcript['exons'] = sorted([list(e) for e in exons[row.transcript_id]])  # needs to be mutable
            transcript['coverage'] = {sample: row[sample_cov_cols[sample]] for sample in samples if row[sample_cov_cols[sample]] > 0}
            transcript['strand'] = gene_infos[row.chr][row.gene_id][0]['strand']  # gene_infos is a 3 tuple (info, start, end)
            transcript['TSS'] = {sample: {transcript['exons'][0][0]: row[sample_cov_cols[sample]]} for sample in samples if row[sample_cov_cols[sample]] > 0}
            transcript['PAS'] = {sample: {transcript['exons'][-1][1]: row[sample_cov_cols[sample]]} for sample in samples if row[sample_cov_cols[sample]] > 0}
            if transcript['strand'] == '-':
                transcript['TSS'], transcript['PAS'] = transcript['PAS'], transcript['TSS']
            used_transcripts.add(row.transcript_id)
        except KeyError as e:
            logger.warning('skipping transcript %s from gene %s, missing infos in gtf: %s', row.transcript_id, row.gene_id, e.args[0])
        except AssertionError as e:
            logger.warning('skipping transcript %s from gene %s: duplicate transcript id; Error: %s', row.transcript_id, row.gene_id, e)
    id_map = {}
    novel_prefix = 'IT_novel_'
    if reconstruct_genes:
        # this approach ignores gene structure, and reassigns transcripts
        novel = {}
        for _, row in cov_tab.iterrows():
            if row.transcript_id not in used_transcripts:
                continue
            transcript = transcripts[row.gene_id][row.transcript_id]
            gene = _add_sample_transcript(self, transcript, row.chr, fuzzy_junction, min_exonic_ref_coverage)

            if gene is None:
                transcript['_original_ids'] = (row.gene_id, row.transcript_id)
                novel.setdefault(row.chr, []).append(Interval(transcript['exons'][0][0], transcript['exons'][-1][1], transcript))
            elif gene.id != row.gene_id:
                id_map.setdefault(row.gene_id, {})[row.transcript_id] = gene.id
        for chrom in novel:
            novel_genes = _add_novel_genes(self, IntervalTree(novel[chrom]), chrom, gene_prefix=novel_prefix)
            for novel_g in novel_genes:
                for novel_transcript in novel_g.transcripts:
                    import_id = novel_transcript.pop('_original_ids')
                    if novel_g.id != import_id[0]:
                        id_map.setdefault(import_id[0], {})[import_id[1]] = novel_g.id
    else:
        # use gene structure from gtf
        for chrom in gene_infos:
            for gid, (g, start, end) in gene_infos[chrom].items():
                # only transcripts with coverage
                import_id = g['ID']
                transcript_list = [transcript for transcript_id, transcript in transcripts[gid].items() if transcript_id in used_transcripts]
                # find best matching overlapping ref gene
                gene = _add_sample_gene(self, start, end, g, transcript_list, chrom, novel_prefix)
                if import_id != gene.id:
                    id_map[import_id] = gene.id
    # todo: extend sample_table
    for sample in samples:
        sample_properties[sample].update({'name': sample, 'file': coverage_csv_file, 'nonchimeric_reads': cov_tab[sample_cov_cols[sample]].sum(), 'chimeric_reads': 0})
        # self.infos['sample_table'] = self.sample_table.append(sample_properties[sample], ignore_index=True)
        self.infos['sample_table'] = pd.concat([self.sample_table, pd.DataFrame([sample_properties[sample]])], ignore_index=True)

    self.make_index()
    return id_map


def add_sample_from_bam(self: Transcriptome, fn, sample_name=None, barcode_file=None, fuzzy_junction=5, add_chromosomes=True, min_mapqual=0,
                        min_align_fraction=.75, chimeric_mincov=2, min_exonic_ref_coverage=.25, use_satag=False, save_readnames=False, progress_bar=True,
                        strictness=math.inf, **kwargs):
    '''Imports expressed transcripts from bam and adds it to the 'Transcriptome' object.

    :param fn: The bam filename of the new sample
    :param sample_name: Name of the new sample. If specified, all reads are assumed to belong to this sample.
    :param barcode_file: For barcoded samples, a file with assignment of sequencing barcodes to sample names.
        This file should be a tab separated text file with two columns: the barcode and the sample name
        Barcodes not listed in this file will be ignored.
        If sample_name is specified in addition to barcode_file, it will be used as a prefix.
        Barcodes will be read from the XC tag of the bam file.
    :param fuzzy_junction: maximum size for fuzzy junction correction
    :param add_chromosomes: If True, genes from chromosomes which are not in the Transcriptome yet are added.
    :param min_mapqual: Minimum mapping quality of the read to be considered.
        A mapping quality of 0 usually means ambigous alignment.
    :param min_align_fraction: Minimum fraction of the read sequence matching the reference.
    :param chimeric_mincov: Minimum number of reads for a chimeric transcript to be considered
    :param min_exonic_ref_coverage: Minimal fraction of exonic overlap to assign to reference transcript if no splice junctions match.
        Also applies to mono-exonic transcripts
    :param use_satag: If True, import secondary alignments (of chimeric alignments) from the SA tag.
        This should only be specified if the secondary alignment is not reported in a separate bam entry.
    :param save_readnames: Save a list of the readnames, that contributed to the transcript.
    :param progress_bar: Show the progress.
    :param strictness: Number of bp that two transcripts are allowed to differ for transcription start and end sites to be still considered one transcript.
    :param kwargs: Additional keyword arguments are added to the sample table.'''

    # todo: one alignment may contain several samples - this is not supported at the moment
    if barcode_file is None:
        assert sample_name is not None, 'Neither sample_name nor barcode_file was specified.'
        assert sample_name not in self.samples, 'sample %s is already in the data set.' % sample_name
        logger.info('adding sample %s from file %s', sample_name, fn)
        barcodes = {}
    else:
        # read the barcode file
        barcodes = pd.read_csv(barcode_file, sep='\t', names=['bc', 'name'], index_col='bc')['name']
        if sample_name is not None:
            barcodes = barcodes.apply(lambda x: '{}_{}'.format(sample_name, x))
        barcodes = barcodes.to_dict()
        assert all(sample not in self.samples for sample in barcodes), \
            'samples %s are already in the data set.' % ', '.join(sample for sample in barcodes if sample in self.samples)
        logger.info('adding %s transcriptomes in %s groups as specified in %s from file %s',
                    len(set(barcodes.keys())), len(set(barcodes.values())), barcode_file, fn)
        # add reverse complement
        barcodes.update({rc(k): v for k, v in barcodes.items()})

    kwargs['file'] = fn
    skip_bc = 0
    partial_count = 0
    # genome_fh=FastaFile(genome_fn) if genome_fn is not None else None
    with AlignmentFile(fn, "rb") as align:
        if add_chromosomes:
            chromosomes = align.references
        else:
            chromosomes = self.chromosomes

        # a sum of reads mapped/unmapped for each chromosome
        stats = align.get_index_statistics()
        # try catch if sam/ no index /not pacbio?
        total_alignments = sum([s.mapped for s in stats if s.contig in chromosomes])
        sample_nc_reads = dict()
        unmapped = n_secondary = n_lowqual = 0
        total_nc_reads_chr = {}
        chimeric = dict()

        with tqdm(total=total_alignments, unit='reads', unit_scale=True, disable=not progress_bar) as pbar:

            for chrom in chromosomes:  # todo: potential issue here - secondary/chimeric alignments to non listed chromosomes are ignored
                total_nc_reads_chr[chrom] = dict()
                pbar.set_postfix(chr=chrom)
                # transcripts=IntervalTree()
                # novel=IntervalTree()
                chr_len = align.get_reference_length(chrom)
                transcript_intervals: IntervalArray[Interval] = IntervalArray(chr_len)  # intervaltree was pretty slow for this context
                novel = IntervalArray(chr_len)
                n_reads = 0
                for read in align.fetch(chrom):
                    n_reads += 1
                    pbar.update(.5)
                    # unmapped
                    if read.flag & 0x4:
                        unmapped += 1
                        continue
                    # not primary alignment or failed qual check or PCR duplicate
                    if read.flag & 0x700:
                        # use only primary alignments
                        n_secondary += 1
                        continue
                    # Mapping quality of 0 usually means ambigous mapping.
                    if read.mapping_quality < min_mapqual:
                        n_lowqual += 1
                        continue
                    tags = dict(read.get_tags())
                    if barcodes:
                        if 'XC' not in tags or tags['XC'] not in barcodes:
                            skip_bc += 1
                            continue

                    s_name = sample_name if not barcodes else barcodes[tags['XC']]
                    strand = '-' if read.is_reverse else '+'
                    exons = junctions_from_cigar(read.cigartuples, read.reference_start)
                    transcript_range = (exons[0][0], exons[-1][1])
                    if transcript_range[0] < 0 or transcript_range[1] > chr_len:
                        logger.error('Alignment outside chromosome range: transcript at %s for chromosome %s of length %s', transcript_range, chrom, chr_len)
                        continue

                    if 'is' in tags:
                        # number of actual reads supporting this transcript
                        cov = tags['is']
                    else:
                        cov = 1

                    # part of a chimeric alignment
                    if 'SA' in tags or read.flag & 0x800:
                        # store it if it's part of a chimeric alignment and meets the minimum coverage threshold
                        # otherwise ignore chimeric read
                        if chimeric_mincov > 0:
                            chimeric.setdefault(read.query_name, [{s_name: cov}, []])
                            assert chimeric[read.query_name][0][s_name] == cov, \
                                'error in bam: parts of chimeric alignment for read %s has different coverage information %s != %s' % (
                                    read.query_name, chimeric[read.query_name][0], cov)
                            chimeric[read.query_name][1].append([chrom, strand, exons, aligned_part(read.cigartuples, read.is_reverse), None])
                            if use_satag and 'SA' in tags:
                                for snd_align in (sa.split(',') for sa in tags['SA'].split(';') if sa):
                                    snd_cigartuples = cigar_string2tuples(snd_align[3])
                                    snd_exons = junctions_from_cigar(snd_cigartuples, int(snd_align[1]))
                                    chimeric[read.query_name][1].append(
                                        [snd_align[0], snd_align[2], snd_exons, aligned_part(snd_cigartuples, snd_align[2] == '-'), None])
                                    # logging.debug(chimeric[read.query_name])
                        continue

                    # skipping low-quality alignments
                    try:
                        # if edit distance becomes large relative to read length, skip the alignment
                        if min_align_fraction > 0 and (1 - tags['NM'] / read.query_length) < min_align_fraction:
                            partial_count += 1
                            continue
                    except KeyError:
                        logging.warning('min_align_fraction set > 0 (%s), but reads found without "NM" tag. Setting min_align_fraction to 0',
                                        min_align_fraction)
                        min_align_fraction = 0

                    total_nc_reads_chr[chrom].setdefault(s_name, 0)
                    total_nc_reads_chr[chrom][s_name] += cov
                    # did we see this transcript already?
                    for transcript_interval in transcript_intervals.overlap(*transcript_range):
                        if transcript_interval.data['strand'] != strand:
                            continue
                        if splice_identical(exons, transcript_interval.data['exons'], strictness=strictness):
                            transcript = transcript_interval.data
                            transcript.setdefault('range', {}).setdefault(transcript_range, 0)
                            transcript['range'][transcript_range] += cov
                            if save_readnames:
                                transcript['reads'].setdefault(s_name, []).append(read.query_name)
                            break
                    else:
                        transcript = {'exons': exons, 'range': {transcript_range: cov}, 'strand': strand}
                        if barcodes:
                            transcript['bc_group'] = barcodes[tags['XC']]
                        if save_readnames:
                            transcript['reads'] = {s_name: [read.query_name]}
                        transcript_intervals.add(Interval(*transcript_range, transcript))
                    # if genome_fh is not None:
                    #    mutations=get_mutations(read.cigartuples, read.query_sequence, genome_fh, chrom,read.reference_start,read.query_qualities)
                    #    for pos,ref,alt,qual in mutations:
                    #        transcript.setdefault('mutations',{}).setdefault(sample_name,{}).setdefault(pos,{'ref':ref}).setdefault(alt,[0,[]])
                    #        transcript['mutations'][sample_name][pos][alt][0]+=cov
                    #        if qual:
                    #            transcript['mutations'][sample_name][pos][alt][1].append(qual) #assuming the quality already accounts for cov>1

                    if 4 in read.cigartuples:  # clipping
                        clip = get_clipping(read.cigartuples, read.reference_start)
                        transcript.setdefault('clipping', {}).setdefault(s_name, {}).setdefault(clip, 0)
                        transcript['clipping'][s_name][clip] += cov

                for transcript_interval in transcript_intervals:
                    transcript = transcript_interval.data
                    transcript_ranges = transcript.pop('range')
                    _set_ends_of_transcript(transcript, transcript_ranges, s_name)

                    gene = _add_sample_transcript(self, transcript, chrom, fuzzy_junction, min_exonic_ref_coverage, strictness=strictness)
                    if gene is None:
                        novel.add(transcript_interval)
                    else:
                        _ = transcript.pop('bc_group', None)

                    # update the total number of reads processed
                    n_reads -= cov
                    pbar.update(cov / 2)

                _ = _add_novel_genes(self, novel, chrom)

                # some reads are not processed here, add them to the progress: chimeric, unmapped, secondary alignment
                pbar.update(n_reads / 2)

                # logger.debug(f'imported {total_nc_reads_chr[chrom]} nonchimeric reads for {chrom}')
                for sample, nc_reads in total_nc_reads_chr[chrom].items():
                    sample_nc_reads[sample] = sample_nc_reads.get(sample, 0) + nc_reads

    if partial_count:
        logger.info('skipped %s reads aligned fraction of less than %s.', partial_count, min_align_fraction)
    if skip_bc:
        logger.warning('skipped %s reads with barcodes not found in the provided list.', skip_bc)
    if n_secondary > 0:
        logger.info('skipped %s secondary alignments (0x100), alignment that failed quality check (0x200) or PCR duplicates (0x400)', n_secondary)
    if unmapped > 0:
        logger.info('ignored %s reads marked as unaligned', unmapped)
    if n_lowqual > 0:
        logger.info('ignored %s reads with mapping quality < %s', n_lowqual, min_mapqual)

    # merge chimeric reads and assign gene names
    n_chimeric = dict()
    # split up chimeric reads in actually chimeric and long introns (non_chimeric)
    # chimeric are grouped by breakpoint (bp->{readname->(chrom, strand, exons)})
    chim, non_chimeric = _check_chimeric(chimeric)
    # sample_nc_reads[sa]

    n_chimeric = _add_chimeric(self, chim, chimeric_mincov, min_exonic_ref_coverage)
    n_long_intron = {}
    for nc_cov_dict, _, _ in non_chimeric.values():
        for sample, nc_cov in nc_cov_dict.items():
            n_long_intron[sample] = n_long_intron.get(sample, 0) + nc_cov
            sample_nc_reads[sample] = sample_nc_reads.get(sample, 0) + nc_cov
    chim_ignored = sum(len(chim) for chim in chimeric.values()) - sum(n_chimeric.values())
    if chim_ignored > 0:
        logger.info('ignoring %s chimeric alignments with less than %s reads', chim_ignored, chimeric_mincov)
    chained_msg = '' if not sum(n_long_intron.values()) else f' (including  {sum(n_long_intron.values())} chained chimeric alignments)'
    chimeric_msg = '' if sum(n_chimeric.values()) == 0 else f' and {sum(n_chimeric.values())} chimeric reads with coverage of at least {chimeric_mincov}'
    logger.info('imported %s nonchimeric reads%s%s.', sum(sample_nc_reads.values()), chained_msg, chimeric_msg)

    for readname, (cov, (chrom, strand, exons, _, _), introns) in non_chimeric.items():
        novel = dict()
        try:
            tss, pas = (exons[0][0], exons[-1][1]) if strand == '+' else (exons[-1][1], exons[0][0])
            transcript = {
                'exons': exons,
                'coverage': cov,
                'TSS': {sample: {tss: c} for sample, c in cov.items()},
                'PAS': {sample: {pas: c} for sample, c in cov.items()},
                'strand': strand,
                'chr': chrom,
                'long_intron_chimeric': introns
            }
            if save_readnames:
                transcript['reads'] = {s_name: [readname]}
        except BaseException:
            logger.error('\n\n-->%s\n\n', (exons[0][0], exons[-1][1]) if strand == "+" else (exons[-1][1], exons[0][0]))
            raise
        gene = _add_sample_transcript(self, transcript, chrom, fuzzy_junction, min_exonic_ref_coverage, strictness=strictness)
        if gene is None:
            novel.setdefault(chrom, []).append(transcript)
        for chrom in novel:
            _ = _add_novel_genes(self, IntervalTree(Interval(transcript['exons'][0][0], transcript['exons'][-1][1], transcript) for transcript in novel[chrom]), chrom)

        # self.infos.setdefault('chimeric',{})[s_name]=chimeric # save all chimeric reads (for debugging)
    for gene in self:
        if 'coverage' in gene.data and gene.data['coverage'] is not None:  # still valid splice graphs no new transcripts - add a row of zeros to coveage
            gene._set_coverage()
    for s_name in sample_nc_reads:
        kwargs['chimeric_reads'] = n_chimeric.get(s_name, 0)
        kwargs['nonchimeric_reads'] = sample_nc_reads.get(s_name, 0)
        kwargs['name'] = s_name
        # self.infos['sample_table'] = self.sample_table.append(kwargs, ignore_index=True)
        self.infos['sample_table'] = pd.concat([self.sample_table, pd.DataFrame([kwargs])], ignore_index=True)
    self.make_index()
    return total_nc_reads_chr


def _add_chimeric(transcriptome: Transcriptome, new_chimeric, min_cov, min_exonic_ref_coverage):
    ''' add new chimeric transcripts to transcriptome, if covered by > min_cov reads
    '''
    total = {}
    for new_bp, new_chim_dict in new_chimeric.items():
        n_reads = {}
        for cov_all, _ in new_chim_dict.values():
            for sample, cov in cov_all.items():
                n_reads[sample] = n_reads.get(sample, 0)+cov
        n_reads = {sample: cov for sample, cov in n_reads.items() if cov >= min_cov}
        if not n_reads:
            continue
        for sample in n_reads:
            total[sample] = total.get(sample, 0) + n_reads[sample]
        for _, new_chim in new_chim_dict.items():  # ignore the readname for now
            # should not contain: long intron, one part only (filtered by check_chimeric function),
            # todo: discard invalid (large overlaps, large gaps)
            # find equivalent chimeric reads
            for found in transcriptome.chimeric.setdefault(new_bp, []):
                if all(splice_identical(ch1[2], ch2[2]) for ch1, ch2 in zip(new_chim[1], found[1])):
                    # add coverage
                    for sample in n_reads:
                        found[0][sample] = found[0].get(sample, 0) + n_reads[sample]
                    # adjust start using the maximum range
                    # this part deviates from normal transcripts, where the median tss/pas is used
                    if found[1][0][1] == '+':  # strand of first part
                        found[1][0][2][0][0] = min(found[1][0][2][0][0], new_chim[1][0][2][0][0])
                    else:
                        found[1][0][2][0][1] = max(found[1][0][2][0][1], new_chim[1][0][2][0][1])
                    # adjust end
                    if found[1][-1][1] == '+':  # strand of last part
                        found[1][-1][2][-1][1] = max(found[1][-1][2][-1][1], new_chim[1][-1][2][-1][1])
                    else:
                        found[1][-1][2][-1][0] = min(found[1][-1][2][-1][0], new_chim[1][-1][2][-1][0])
                    break
            else:  # not seen
                transcriptome.chimeric[new_bp].append(new_chim)
                for part in new_chim[1]:
                    if part[0] in transcriptome.data:
                        genes_overlap = [gene for gene in transcriptome.data[part[0]][part[2][0][0]: part[2][-1][1]] if gene.strand == part[1]]
                        gene, _, _ = _find_matching_gene(genes_overlap, part[2], min_exonic_ref_coverage)  # take the best - ignore other hits here
                        if gene is not None:
                            part[4] = gene.name
                            gene.data.setdefault('chimeric', {})[new_bp] = transcriptome.chimeric[new_bp]
    return total


def _breakpoints(chimeric):
    ''' gets chimeric aligment as a list and returns list of breakpoints.
        each breakpoint is a tuple of (chr1, strand1, pos1, chr2, strand2, pos2)
    '''
    return tuple((a[0], a[1], a[2][-1][1] if a[1] == '+' else a[2][0][0],
                  b[0], b[1], b[2][0][0] if b[1] == '+' else b[2][-1][1])
                 for a, b in pairwise(chimeric))


def _check_chimeric(chimeric):
    ''' prepare the chimeric reads:
        1) sort parts according to read order
        2) compute breakpoints
        3) check if the chimeric read is actually a long introns - return list as nonchimeric
        4) sort into dict by breakpoint - return dict as chimeric

        chimeric[0] is the coverage dict
        chimeric[1] is a list of tuples: chrom,strand,exons,[aligned start, end] '''

    chimeric_dict = {}
    non_chimeric = {}
    skip = 0
    for readname, new_chim in chimeric.items():
        if len(new_chim[1]) < 2:
            skip += sum(new_chim[0].values())
            continue
        merged_introns = []

        # 1) sort
        new_chim[1].sort(key=lambda x: x[3][1])  # sort by end of aligned part
        # 2) compute breakpoints
        bpts = _breakpoints(new_chim[1])  # compute breakpoints
        # 3) check if long intron alignment splits
        merge = [i for i, bp in enumerate(bpts) if
                 bp[0] == bp[3] and  # same chr
                 bp[1] == bp[4] and  # same strand,
                 0 < (bp[5] - bp[2] if bp[1] == '+' else bp[2] - bp[5]) < 1e6]  # max 1mb gap -> long intron
        # todo: also check that the aligned parts have not big gap or overlap
        if merge:
            # new_chim[1]
            intron = 0
            for i, part in enumerate(new_chim[1]):
                intron += len(part[2])
                if i in merge:
                    merged_introns.append(intron)

            for i in merge:  # part i is merged into part i+1
                if new_chim[1][i][1] == '+':  # merge into next part
                    new_chim[1][i + 1][2] = new_chim[1][i][2] + new_chim[1][i + 1][2]
                    new_chim[1][i + 1][3] = new_chim[1][i][3] + new_chim[1][i + 1][3]
                else:
                    new_chim[1][i + 1][2] = new_chim[1][i + 1][2] + new_chim[1][i][2]
                    new_chim[1][i + 1][3] = new_chim[1][i + 1][3] + new_chim[1][i][3]
            # remove redundant parts (i)
            new_chim[1] = [part for i, part in enumerate(new_chim[1]) if i not in merge]
            bpts = tuple(bp for i, bp in enumerate(bpts) if i not in merge)
        # sort into chimeric (e.g. still breakpoints left) and nonchimeric (only one part and no breakpoints left)
        if bpts:
            chimeric_dict.setdefault(bpts, {})[readname] = new_chim
        else:
            assert len(new_chim[1]) == 1
            # coverage, part(chrom,strand,exons,[aligned start, end]), and "long introns"
            non_chimeric[readname] = [new_chim[0], new_chim[1][0], tuple(merged_introns)]
    if skip:
        logger.warning('ignored %s chimeric alignments with only one part aligned to specified chromosomes.', skip)
    return chimeric_dict, non_chimeric


def _set_ends_of_transcript(transcript: Transcript, transcript_ranges, sample_name):
    start, end = float('inf'), 0
    starts, ends = {}, {}
    for range, cov in transcript_ranges.items():
        if range[0] < start:
            start = range[0]
        if range[1] > end:
            end = range[1]
        starts[range[0]] = starts.get(range[0], 0) + cov
        ends[range[1]] = ends.get(range[1], 0) + cov

    # This will be changed again, when assigning to a gene
    transcript['exons'][0][0] = start
    transcript['exons'][-1][1] = end

    cov = sum(transcript_ranges.values())
    s_name = transcript.get('bc_group', sample_name)
    transcript['coverage'] = {s_name: cov}
    transcript['TSS'] = {s_name: starts if transcript['strand'] == '+' else ends}
    transcript['PAS'] = {s_name: ends if transcript['strand'] == '+' else starts}


def _add_sample_gene(transcriptome: Transcriptome, gene_start, gene_end, gene_infos, transcript_list, chrom, novel_prefix):
    '''add new gene to existing gene in chrom - return gene on success and None if no Gene was found.
    For matching transcripts in gene, transcripts are merged. Coverage, transcript TSS/PAS need to be reset.
    Otherwise, a new transcripts are added. In this case, splice graph and coverage have to be reset.'''
    if chrom not in transcriptome.data:
        for transcript in transcript_list:
            transcript['annotation'] = (4, {'intergenic': []})
        return transcriptome._add_novel_gene(chrom, gene_start, gene_end, gene_infos['strand'], {'transcripts': transcript_list}, novel_prefix)

    # try matching the gene by id
    try:
        best_gene = transcriptome[gene_infos['ID']]
        if not best_gene.is_annotated or not has_overlap((best_gene.start, best_gene.end), (gene_start, gene_end)):
            best_gene = None
        elsefound = []  # todo: to annotate as fusion, we would need to check for uncovered splice sites, and wether other genes cover them
    except KeyError:
        best_gene = None

    if best_gene is None:
        # find best matching reference gene from t
        genes_overlap_strand = [gene for gene in transcriptome.data[chrom][gene_start: gene_end] if gene.strand == gene_infos['strand']]

        if not genes_overlap_strand:
            covered_splice = 0
        else:
            splice_junctions = sorted(list({(exon1[1], exon2[0]) for transcript in transcript_list for exon1, exon2 in pairwise(transcript['exons'])}))
            splice_sites = np.array([gene.ref_segment_graph.find_splice_sites(splice_junctions) if gene.is_annotated
                                     else _find_splice_sites(splice_junctions, gene.transcripts) for gene in genes_overlap_strand])
            sum_ol = splice_sites.sum(1)
            covered_splice = np.max(sum_ol)
        if covered_splice > 0:
            best_idx = np.flatnonzero(sum_ol == covered_splice)
            best_idx = best_idx[0]
            not_in_best = np.where(~splice_sites[best_idx])[0]
            additional = splice_sites[:, not_in_best]  # additional= sites not covered by top gene
            elsefound = [(gene.name, not_in_best[a]) for gene, a in zip(genes_overlap_strand, additional) if a.sum() > 0]  # genes that cover additional splice sites
            # notfound = (splice_sites.sum(0) == 0).nonzero()[0].tolist()  # not covered splice sites
            # transcript['novel_splice_sites'] = not_found # cannot be done here, as gene is handled at once. TODO: maybe later?
            best_gene = genes_overlap_strand[best_idx]
        else:
            genes_overlap_anti = [gene for gene in transcriptome.data[chrom][gene_start: gene_end] if gene.strand != gene_infos['strand']]
            for transcript in transcript_list:
                transcript['annotation'] = (4, _get_novel_type(transcript['exons'], genes_overlap_anti, genes_overlap_strand))
            return transcriptome._add_novel_gene(chrom, gene_start, gene_end, gene_infos['strand'], {'transcripts': transcript_list}, novel_prefix)

    for transcript in transcript_list:
        for tr2 in best_gene.transcripts:  # check if correction made it identical to existing
            if splice_identical(tr2['exons'], transcript['exons']):
                _combine_transcripts(tr2, transcript)  # potentially loosing information
                break
        else:
            if best_gene.is_annotated and has_overlap((transcript['exons'][0][0], transcript['exons'][-1][1]), (best_gene.start, best_gene.end)):
                # potentially problematic: elsefound [(genename, idx),...] idx does not refere to transcript splice site
                try:
                    transcript['annotation'] = best_gene.ref_segment_graph.get_alternative_splicing(transcript['exons'], elsefound)
                except Exception:
                    logger.error('issue categorizing transcript %s with respect to %s', transcript['exons'], str(best_gene))
                    raise
            else:
                genes_overlap_strand = [gene for gene in transcriptome.data[chrom][gene_start: gene_end] if gene.strand == gene_infos['strand'] and gene.is_annotated]
                genes_overlap_anti = [gene for gene in transcriptome.data[chrom][gene_start: gene_end] if gene.strand != gene_infos['strand'] and gene.is_annotated]
                transcript['annotation'] = (4, _get_novel_type(transcript['exons'], genes_overlap_anti, genes_overlap_strand))  # actually may overlap other genes...
            best_gene.data.setdefault('transcripts', []).append(transcript)
    return best_gene


def _add_sample_transcript(transcriptome: Transcriptome, transcript: Transcript, chrom: str, fuzzy_junction: int, min_exonic_ref_coverage: float, genes_overlap=None, strictness=math.inf):
    '''add transcript to gene in chrom - return gene on success and None if no Gene was found.
    If matching transcript is found in gene, transcripts are merged. Coverage, transcript TSS/PAS need to be reset.
    Otherwise, a new transcript is added. In this case, splice graph and coverage have to be reset.'''

    if chrom not in transcriptome.data:
        transcript['annotation'] = (4, {'intergenic': []})
        return None
    if genes_overlap is None:
        # At this point the transcript still uses min and max from all reads for start and end
        genes_overlap = transcriptome.data[chrom][transcript['exons'][0][0]: transcript['exons'][-1][1]]
    genes_overlap_strand = [gene for gene in genes_overlap if gene.strand == transcript['strand']]
    # check if transcript is already there (e.g. from other sample, or in case of long intron chimeric alignments also same sample):
    for gene in genes_overlap_strand:
        for transcript2 in gene.transcripts:
            if splice_identical(transcript2['exons'], transcript['exons'], strictness=strictness):
                _combine_transcripts(transcript2, transcript)
                return gene
    # we have a new transcript (not seen in this or other samples)
    # check if gene is already there (e.g. from same or other sample):
    gene, additional, not_covered = _find_matching_gene(genes_overlap_strand, transcript['exons'], min_exonic_ref_coverage)
    if gene is not None:
        if gene.is_annotated:  # check for fuzzy junctions (e.g. same small shift at 5' and 3' compared to reference annotation)
            shifts = gene.correct_fuzzy_junctions(transcript, fuzzy_junction, modify=True)  # this modifies transcript['exons']
            if shifts:
                transcript.setdefault('fuzzy_junction', []).append(shifts)  # keep the info, mainly for testing/statistics
                for transcript2 in gene.transcripts:  # check if correction made it identical to existing
                    if splice_identical(transcript2['exons'], transcript['exons'], strictness=strictness):
                        transcript2.setdefault('fuzzy_junction', []).append(shifts)  # keep the info, mainly for testing/statistics
                        _combine_transcripts(transcript2, transcript)
                        return gene
            transcript['annotation'] = gene.ref_segment_graph.get_alternative_splicing(transcript['exons'], additional)

            if not_covered:
                transcript['novel_splice_sites'] = not_covered  # todo: might be changed by fuzzy junction

            # intersects might have changed due to fuzzy junctions
            # {'sj_i': sj_i, 'base_i':base_i,'category':SPLICE_CATEGORY[altsplice[1]],'subcategory':altsplice[1]}

        else:
            # add to existing novel (e.g. not in reference) gene
            start, end = min(transcript['exons'][0][0], gene.start), max(transcript['exons'][-1][1], gene.end)
            transcript['annotation'] = (4, _get_novel_type(transcript['exons'], genes_overlap, genes_overlap_strand))
            if start < gene.start or end > gene.end:  # range of the novel gene might have changed
                new_gene = Gene(start, end, gene.data, transcriptome)
                transcriptome.data[chrom].add(new_gene)  # todo: potential issue: in this case two genes may have grown together
                transcriptome.data[chrom].remove(gene)
                gene = new_gene
        # if additional:
        #    transcript['annotation']=(4,transcript['annotation'][1]) #fusion transcripts... todo: overrule transcript['annotation']
        # this transcript is seen for the first time. Asign sample specific attributes to sample name
        # for what in 'coverage', 'TSS', 'PAS':
        #    transcript[what] = {sample_name: transcript[what]}
        # if 'reads' in transcript:
        #    transcript['reads'] = {sample_name: transcript['reads']}
        gene.data.setdefault('transcripts', []).append(transcript)
        gene.data['segment_graph'] = None  # gets recomputed on next request
        gene.data['coverage'] = None
    else:
        # new novel gene
        transcript['annotation'] = (4, _get_novel_type(transcript['exons'], genes_overlap, genes_overlap_strand))
    return gene


def _combine_transcripts(established: Transcript, new_transcript: Transcript):
    'merge new_transcript into splice identical established transcript'
    try:
        for sample_name in new_transcript['coverage']:
            established['coverage'][sample_name] = established['coverage'].get(sample_name, 0) + new_transcript['coverage'][sample_name]
        for sample_name in new_transcript.get('reads', {}):
            established['reads'].setdefault(sample_name, []).extend(new_transcript['reads'][sample_name])
        for side in 'TSS', 'PAS':
            for sample_name in new_transcript[side]:
                for pos, cov in new_transcript[side][sample_name].items():
                    established[side].setdefault(sample_name, {}).setdefault(pos, 0)
                    established[side][sample_name][pos] += cov
        # find median tss and pas
        starts = [pos for sample in established['TSS'] for pos in established['TSS'][sample].items()]
        ends = [pos for sample in established['PAS'] for pos in established['PAS'][sample].items()]
        if established['strand'] == '-':
            starts, ends = ends, starts

        established['exons'][0][0] = get_quantiles(starts, [0.5])[0]
        established['exons'][-1][1] = get_quantiles(ends, [0.5])[0]
        if 'long_intron_chimeric' in new_transcript:
            new_introns = set(new_transcript['long_intron_chimeric'])
            established_introns = set(established.get('long_intron_chimeric', set()))
            established['long_intron_chimeric'] = tuple(new_introns.union(established_introns))
    except BaseException as e:
        logger.error(f'error when merging {new_transcript} into {established}')
        raise e


def _get_novel_type(exons, genes_overlap, genes_overlap_strand):
    if len(genes_overlap_strand):
        exonic_overlap = {gene.id: gene.ref_segment_graph.get_overlap(exons) for gene in genes_overlap_strand if gene.is_annotated}
        exonic_overlap_genes = [k for k, v in exonic_overlap.items() if v[0] > 0]
        if len(exonic_overlap_genes) > 0:
            return {'genic genomic': exonic_overlap_genes}
        return {'intronic': [gene.name for gene in genes_overlap_strand]}
    elif len(genes_overlap):
        return {'antisense': [gene.name for gene in genes_overlap]}
    else:
        return {'intergenic': []}


def _add_novel_genes(transcriptome: Transcriptome, novel, chrom, spj_iou_th=0, reg_iou_th=.5, gene_prefix='IT_novel_'):
    '"novel" is a tree of transcript intervals (not Gene objects) ,e.g. from one chromosome, that do not overlap any annotated or unanntoated gene'
    n_novel = transcriptome.novel_genes
    idx = {id(transcript): i for i, transcript in enumerate(novel)}
    novel_gene_list = []
    merge = list()
    for i, transcript in enumerate(novel):
        merge.append({transcript})
        candidates = [candidate for candidate in novel.overlap(transcript.begin, transcript.end) if candidate.data['strand'] == transcript.data['strand'] and idx[id(candidate)] < i]
        for candidate in candidates:
            if candidate in merge[i]:
                continue
            if is_same_gene(transcript.data['exons'], candidate.data['exons'], spj_iou_th, reg_iou_th):
                # add all transcripts of candidate
                merge[i].update(merge[idx[id(candidate)]])
        for candidate in merge[i]:  # update all overlapping (add the current to them)
            merge[idx[id(candidate)]] = merge[i]

    seen = set()
    for trS in merge:
        if id(trS) in seen:
            continue
        seen.add(id(trS))
        trL = [transcript.data for transcript in trS]
        strand = trL[0]['strand']
        start = min(transcript['exons'][0][0] for transcript in trL)
        end = max(transcript['exons'][-1][1] for transcript in trL)
        assert start < end, 'novel gene with start >= end'
        # for transcript in transcriptL:
        #    sample_name = transcript.pop('bc_group', sa)
        #    transcript['coverage'] = {sample_name: transcript['coverage']}
        #    transcript['TSS'] = {sample_name: transcript['TSS']}
        #    transcript['PAS'] = {sample_name: transcript['PAS']}
        #    if 'reads' in transcript:
        #        transcript['reads'] = {sample_name: transcript['reads']}
        novel_gene_list.append(transcriptome._add_novel_gene(chrom, start, end, strand, {'transcripts': trL}, gene_prefix))
        logging.debug('merging transcripts of novel gene %s: %s', n_novel, trL)
    return novel_gene_list


def _find_matching_gene(genes_overlap: list[Gene], exons: list[Tuple[int, int]], min_exon_coverage: float):
    '''check the splice site intersect of all overlapping genes and return
            1) the gene with most shared splice sites,
            2) names of genes that cover additional splice sites, and
            3) splice sites that are not covered.
        If no splice site is shared (and for mono-exon genes) return the gene with largest exonic overlap
        :param genes_ol: list of genes that overlap the transcript
        :param exons: exon list of the transcript
        :param min_exon_coverage: minimum exonic coverage with genes that do not share splice sites to be considered'''
    transcript_len = sum(exon[1] - exon[0] for exon in exons)
    splice_junctions = [(exon1[1], exon2[0]) for exon1, exon2 in pairwise(exons)]
    if genes_overlap:
        if len(exons) > 1:
            nomatch = np.zeros(len(splice_junctions) * 2, dtype=bool)
            # Check reference transcript of reference genes first
            splice_sites = np.array([gene.ref_segment_graph.find_splice_sites(splice_junctions) if gene.is_annotated else nomatch for gene in genes_overlap])
            sum_overlap = splice_sites.sum(1)
            # find index of reference gene that covers the most splice sites
            # resolved issue with tie here, missing FSM due to overlapping gene with extension of FSM transcript
            covered_splice = np.max(sum_overlap)
            if covered_splice == 0:
                # none found, consider novel genes
                # TODO: Consider to replace this with gene.segment_graph.find_splice_sites(splice_junctions)
                # since the graph is cached, this might be faster
                splice_sites = np.array([_find_splice_sites(splice_junctions, gene.transcripts) if not gene.is_annotated else nomatch for gene in genes_overlap])
                sum_overlap = splice_sites.sum(1)
                covered_splice = np.max(sum_overlap)
            if covered_splice > 0:
                best_idx = np.flatnonzero(sum_overlap == covered_splice)
                if len(best_idx > 1):  # handle ties
                    # find the transcript with the highest fraction of matching junctions
                    transcript_inter = []
                    for idx in best_idx:
                        transcript_list = genes_overlap[idx].ref_transcripts if genes_overlap[idx].is_annotated else genes_overlap[idx].transcripts
                        transcript_intersects = [(get_intersects(exons, transcript['exons'])) for transcript in transcript_list]
                        transcript_intersects_fraction = [(junction_intersection / len(transcript_list[idx]['exons']), exon_intersection)
                                                          for idx, (junction_intersection, exon_intersection) in enumerate(transcript_intersects)]
                        transcript_inter.append(max(transcript_intersects_fraction))
                    best_idx = best_idx[np.argmax(transcript_inter, axis=0)[0]]
                else:
                    best_idx = best_idx[0]
                not_in_best = np.where(~splice_sites[best_idx])[0]
                # sites not covered by top gene but in exons
                additional = splice_sites[:, not_in_best]
                # genes that cover additional splice sites
                elsefound = [(gene.name, not_in_best[a]) for gene, a in zip(genes_overlap, additional) if a.sum() > 0]
                # not covered splice sites
                notfound = (splice_sites.sum(0) == 0).nonzero()[0].tolist()
                return genes_overlap[best_idx], elsefound, notfound
            # no shared splice sites, return gene with largest overlap
            # first, check reference here:
            # 1) if >50% overlap with ref gene -> return best ref gene
            overlap = np.array([gene.ref_segment_graph.get_overlap(exons)[0] if gene.is_annotated else 0 for gene in genes_overlap])
            best_idx = overlap.argmax()
            if overlap[best_idx] >= min_exon_coverage * transcript_len:
                return genes_overlap[best_idx], None, list(range((len(exons) - 1) * 2))
        else:
            # len(exons)==1 check all ref transcripts for monoexon gene with overlap>=50% (or min_exon_coverage)
            overlap = []
            for gene in genes_overlap:
                overlap_gene = [get_overlap(exons[0], transcript['exons'][0]) for transcript in gene.ref_transcripts if len(transcript['exons']) == 1]
                overlap.append(max(overlap_gene, default=0))
            best_idx = np.argmax(overlap)
            if overlap[best_idx] >= min_exon_coverage * transcript_len:
                return genes_overlap[best_idx], None, []
        # else return best overlapping novel gene if more than minimum overlap fraction
        overlap = [(0, []) if not gene.is_annotated else gene.ref_segment_graph.get_overlap(exons) for gene in genes_overlap]
        max_overlap_frac = np.array([0 if overlap[0] == 0 else
                                     max(overlap_transcript / min(transcript_len, sum(exon[1] - exon[0] for exon in transcript["exons"]))
                                         for overlap_transcript, transcript in zip(overlap[1], gene.ref_transcripts))
                                     for gene, overlap in zip(genes_overlap, overlap)])
        best_idx = max_overlap_frac.argmax()
        if max_overlap_frac[best_idx] >= min_exon_coverage:
            return genes_overlap[best_idx], None, list(range((len(exons) - 1) * 2))  # none of the junctions are covered
        # else return best overlapping novel gene if more than minimum overlap fraction
        overlap = [0 if gene.is_annotated else _get_overlap(exons, gene.transcripts) for gene in genes_overlap]
        max_overlap_frac = np.array([0 if overlap == 0 else overlap_gene / transcript_len for overlap_gene in overlap])
        best_idx = max_overlap_frac.argmax()
        if max_overlap_frac[best_idx] >= min_exon_coverage:
            return genes_overlap[best_idx], None, list(range((len(exons) - 1) * 2))  # none of the junctions are covered
        # TODO: Issue: order matters here, if more than one novel gene with >50%overlap, join them all?)
    # none of the junctions are covered
    return None, None, list(range((len(exons) - 1) * 2))


def _read_gtf_file(file_name, chromosomes, infer_genes=False, progress_bar=True):
    exons = dict()  # transcript id -> exons
    transcripts = dict()  # gene_id -> transcripts
    skipped = defaultdict(set)
    gene_infos = dict()  # 4 tuple: info_dict, gene_start, gene_end, fixed_flag==True if start/end are fixed
    cds_start = dict()
    cds_stop = dict()
    # with tqdm(total=path.getsize(file_name), unit_scale=True, unit='B', unit_divisor=1024, disable=not progress_bar) as pbar, TabixFile(file_name) as gtf:
    # for line in gtf.fetch():
    #    file_pos = gtf.tell() >> 16
    #    if pbar.n < file_pos:
    #       pbar.update(file_pos-pbar.n)
    openfun = gziplib.open if file_name.endswith('.gz') else open

    with openfun(file_name, "rt") as gtf:
        for line in gtf:
            if line[0] == '#':  # ignore header lines
                continue
            ls = line.split(sep="\t")
            chr = ls[0]
            assert len(ls) == 9, 'unexpected number of fields in gtf line:\n%s' % line
            if chromosomes is not None and chr not in chromosomes:
                logger.debug('skipping line from chr ' + chr)
                continue
            try:
                info = dict([pair.lstrip().split(maxsplit=1) for pair in ls[8].strip().replace('"', '').split(";") if pair])
            except ValueError:
                logger.error('issue with key value pairs from gtf:\n%s', ls[8])
                raise

            # gtf of transcriptome reconstructed by external tools may include entries without strand, which can't be mapped to the genome, so skip them
            if ls[6] not in ('+', '-'):
                logger.warning('skipping line with unknown strand:\n%s', line)
                # add this entry to skipped
                # keys are feature types (ls[2], e.g. gene, transcript, exon) and values are sets of feature ids that are searched in ls[-1]
                feature_id = [i.split(' ')[-1].strip('"') for i in ls[-1].split(sep=';') if f'{ls[2]}_id' in i or f'{ls[2]}_number' in i]
                if len(feature_id) == 1:
                    skipped[ls[2]].add(feature_id[0])
                else:
                    logger.debug(f'found {"multiple" if len(feature_id) > 1 else "no"} feature ids in line:\n{line}')
                    pass
                continue

            start, end = [int(i) for i in ls[3:5]]
            start -= 1  # to make 0 based
            gene_infos.setdefault(chr, {})
            if ls[2] == "exon":
                # logger.debug(line)
                try:
                    _ = exons.setdefault(info['transcript_id'], list()).append((start, end))
                except KeyError:  # should not happen if GTF is OK
                    logger.error("gtf format error: exon without transcript_id\n%s", line)
                    raise
                if infer_genes and 'gene_id' in info:
                    if info['gene_id'] not in gene_infos[chr]:  # new gene
                        info['strand'] = ls[6]
                        info['chr'] = chr
                        _set_alias(info, {'ID': ['gene_id']})
                        _set_alias(info, {'name': ['gene_name', 'Name']}, required=False)
                        ref_info = {k: v for k, v in info.items() if k not in Gene.required_infos + ['name']}
                        info = {k: info[k] for k in Gene.required_infos + ['name'] if k in info}
                        info['properties'] = ref_info
                        gene_infos[chr][info['ID']] = (info, start, end)  # start/end not fixed yet (initially set to exon start end)
                    else:
                        known_info = gene_infos[chr][info['gene_id']]
                        gene_infos[chr][info['gene_id']] = (known_info[0], min(known_info[1], start), max(known_info[2], end))
                        if 'transcript_id' in info and info['transcript_id'] not in transcripts.setdefault(info['gene_id'], {}):
                            # new transcript
                            tr_info = {k: v for k, v in info.items() if 'transcript' in k and k != 'transcript_id'}
                            transcripts[info["gene_id"]][info["transcript_id"]] = tr_info
            elif ls[2] == 'gene':
                if 'gene_id' not in info:
                    logger.warning("gtf format error: gene without gene_id. Skipping line\n%s", line)
                else:  # overrule potential entries from exon line
                    info['strand'] = ls[6]
                    info['chr'] = chr
                    _set_alias(info, {'ID': ['gene_id']})
                    _set_alias(info, {'name': ['gene_name', 'Name']}, required=False)
                    ref_info = {k: v for k, v in info.items() if k not in Gene.required_infos + ['name']}
                    info = {k: info[k] for k in Gene.required_infos + ['name'] if k in info}
                    info['properties'] = ref_info
                    gene_infos[chr][info['ID']] = (info, start, end)
            elif ls[2] == 'transcript':  # overrule potential entries from exon line
                try:
                    # keep only transcript related infos (to avoid redundant gene infos)
                    tr_info = {k: v for k, v in info.items() if 'transcript' in k and k != 'transcript_id'}
                    _ = transcripts.setdefault(info["gene_id"], dict())[info["transcript_id"]] = tr_info
                except KeyError:
                    logger.warning("gtf format errror: transcript must have gene_id and transcript_id, skipping line\n%s", line)
            elif ls[2] == 'start_codon' and 'transcript_id' in info:
                cds_start[info['transcript_id']] = end if ls[6] == '-' else start
            elif ls[2] == 'stop_codon' and 'transcript_id' in info:
                cds_stop[info['transcript_id']] = start if ls[6] == '-' else end
            else:
                # skip other feature types. Only keep a record of feature type without further information in skipped
                # this usually happens to reference annotation, eg: UTR, CDS etc.
                skipped[ls[2]]

    return exons, transcripts, gene_infos, cds_start, cds_stop, skipped


def _get_tabix_end(tbx_fh):
    for line in tbx_fh.fetch(tbx_fh.contigs[-1]):
        pass
    end = tbx_fh.tell()
    tbx_fh.seek(0)
    return end


def _read_gff_file(file_name, chromosomes, progress_bar=True):
    exons = dict()  # transcript id -> exons
    transcripts = dict()  # gene_id -> transcripts
    skipped = defaultdict(set)
    genes = dict()
    cds_start = dict()
    cds_stop = dict()
    # takes quite some time... add a progress bar?
    with tqdm(total=path.getsize(file_name), unit_scale=True, unit='B', unit_divisor=1024, disable=not progress_bar) as pbar, TabixFile(file_name) as gff:
        chrom_ids = get_gff_chrom_dict(gff, chromosomes)
        for line in gff.fetch():
            file_pos = gff.tell() >> 16  # the lower 16 bit are the position within the zipped block
            if pbar.n < file_pos:
                pbar.update(file_pos-pbar.n)
            ls = line.split(sep="\t")
            if ls[0] not in chrom_ids:
                continue
            chrom = chrom_ids[ls[0]]
            if chromosomes is not None and chrom not in chromosomes:
                logger.debug('skipping line %s from chr %s', line, chrom)
                continue
            try:
                info = dict([pair.split('=', 1) for pair in ls[8].rstrip(';').split(";")])  # some gff lines end with ';' in gencode 36
            except ValueError:
                logger.warning("GFF format error in infos (should be ; separated key=value pairs). Skipping line:\n%s", line)
            start, end = [int(i) for i in ls[3:5]]
            start -= 1  # to make 0 based
            if ls[2] == "exon":
                try:
                    gff_id = info['Parent']
                    exons.setdefault(gff_id, list()).append((start, end))
                except KeyError:  # should not happen if GFF is OK
                    logger.warning("GFF format error: no parent found for exon. Skipping line:\n%s", line)
            elif ls[2] == 'gene' or 'ID' in info and info['ID'].startswith('gene'):
                info['strand'] = ls[6]
                info['chr'] = chrom
                _set_alias(info, {'ID': ['gene_id']})
                _set_alias(info, {'name': ['Name', 'gene_name']}, required=False)
                ref_info = {k: v for k, v in info.items() if k not in Gene.required_infos + ['name']}
                info = {k: info[k] for k in Gene.required_infos + ['name'] if k in info}
                info['properties'] = ref_info
                genes.setdefault(chrom, {})[info['ID']] = (info, start, end)
            elif all([v in info for v in ['Parent', "ID"]]) and (ls[2] == 'transcript' or info['Parent'].startswith('gene')):  # those denote transcripts
                tr_info = {k: v for k, v in info.items() if k.startswith('transcript_')}
                transcripts.setdefault(info["Parent"], {})[info['ID']] = tr_info
            elif ls[2] == 'start_codon' and 'Parent' in info:
                cds_start[info['Parent']] = end if ls[6] == '-' else start
            elif ls[2] == 'stop_codon' and 'Parent' in info:
                cds_stop[info['Parent']] = start if ls[6] == '-' else end
            else:
                # skip other feature types. Only keep a record of feature type without further information in skipped
                # this usually happens to reference annotation, eg: UTR, CDS etc.
                skipped[ls[2]]
    return exons, transcripts, genes, cds_start, cds_stop, skipped


def import_ref_transcripts(fn, transcriptome: Transcriptome, file_format, chromosomes=None, gene_categories=None, short_exon_th=25, **kwargs):
    '''import transcripts from gff/gtf file (e.g. for a reference)
    returns a dict interval trees for the genes'''
    if gene_categories is None:
        gene_categories = ['gene']
    if file_format == 'gtf':
        exons, transcripts, gene_infos, cds_start, cds_stop, skipped = _read_gtf_file(fn, chromosomes, **kwargs)
    else:  # gff/gff3
        exons, transcripts, gene_infos, cds_start, cds_stop, skipped = _read_gff_file(fn, chromosomes, **kwargs)

    if skipped:
        logger.info('skipped the following categories: %s', skipped.keys())
    
    logger.debug('construct interval trees for genes...')
    genes: dict[str, IntervalTree[Gene]] = {}
    for chrom in gene_infos:
        for info, _, _ in gene_infos[chrom].values():
            try:
                info['reference'] = info.pop('properties')
            except KeyError:
                logger.error(info)
                raise
        genes[chrom] = IntervalTree(Gene(start, end, info, transcriptome) for info, start, end in gene_infos[chrom].values())

    # sort the exons
    logger.debug('sorting exon positions...')
    for tid in exons:
        exons[tid].sort()
    all_genes = set().union(*(set(gene_info.keys()) for gene_info in gene_infos.values()))
    missed_genes = [gene_id for gene_id in transcripts.keys() if gene_id not in all_genes]
    if missed_genes:
        # logger.debug('/n'.join(gene_id+str(transcript) for gene_id, transcript in missed_genes.items()))
        notfound = len(missed_genes)
        found = sum((len(t) for t in genes.values()))
        logger.warning('Missing genes! Found gene information in categories %s for %s/%s genes', gene_categories, found, found + notfound)
    logger.debug('building gene data structure...')
    # add transcripts to genes
    for chrom in genes:
        for gene in genes[chrom]:
            gene_id = gene.id
            transcript = transcripts.get(gene_id, {gene_id: {}})
            for transcript_id, transcript_info in transcript.items():
                transcript_info['transcript_id'] = transcript_id
                try:
                    transcript_info['exons'] = exons[transcript_id]
                except KeyError:
                    # genes without exons get a single exons transcript
                    transcript_info['exons'] = [tuple(gene[:2])]
                # add cds
                if transcript_id in cds_start and transcript_id in cds_stop:
                    transcript_info['CDS'] = (cds_start[transcript_id], cds_stop[transcript_id]) if cds_start[transcript_id] < cds_stop[transcript_id] else (cds_stop[transcript_id], cds_start[transcript_id])
                gene.data['reference'].setdefault('transcripts', []).append(transcript_info)
            if short_exon_th is not None:
                short_exons = {exon for transcript in gene.data['reference']['transcripts'] for exon in transcript['exons'] if exon[1] - exon[0] <= short_exon_th}
                if short_exons:
                    gene.data['reference']['short_exons'] = short_exons
    return genes


def import_sqanti_classification(self: Transcriptome, path: str):
    """
    Import a SQANTI classification file.
    See https://github.com/ConesaLab/SQANTI3/wiki/Understanding-the-output-of-SQANTI3-QC#classifcols
    for details.
    """
    sqanti_df = pd.read_csv(path, sep='\t')
    for _, row in tqdm(sqanti_df.iterrows(), total=len(sqanti_df)):
        isoform = row['isoform']
        gene_id = '_'.join(isoform.split('_')[:-1])
        if gene_id not in self:
            raise KeyError(f'Gene {gene_id} not found in transcriptome. Make sure you passed the correct SQANTI file')
        gene = self[gene_id]
        transcript_id = int(isoform.split('_')[-1])
        gene.add_sqanti_classification(transcript_id, row)


def collapse_immune_genes(self: Transcriptome, maxgap=300000):
    ''' This function collapses annotation of immune genes (IG and TR) of a loci.

    As immune genes are so variable, classical annotation as a set of transcripts is not meaningfull for those genes.
    In consequence, each component of an immune gene is usually stored as an individual gene.
    This can cause issues when comparing transcripts to these genes, which naturally would overlap many of these components.
    To avoid these issues, immunoglobulin and T-cell receptor genes of a locus are combined to a single gene,
    without specifiying transcripts.
    Immune genes are recognized by the gff/gtf property "gene_type" set to "IG*_gene" or "TR*_gene".
    Components within the distance of "maxgap" get collapsed to a single gene called TR/IG_locus_X.
    :param maxgap: Specify maximum distance between components of the same locus.
    '''
    assert not self.samples, 'immune gene collapsing has to be called before adding long read samples'
    num = {'IG': 0, 'TR': 0}
    for chrom in self.data:
        for strand in ('+', '-'):
            immune = {'IG': [], 'TR': []}
            for gene in self.data[chrom]:
                if gene.strand != strand or not gene.is_annotated or 'gene_type' not in gene.data['reference']:
                    continue
                gene_type = gene.data['reference']['gene_type']
                if gene_type[:2] in immune and gene_type[-5:] == '_gene':
                    immune[gene_type[:2]].append(gene)
            for itype in immune:
                immune[itype].sort(key=lambda x: (x.start, x.end))
                offset = 0
                for i, gene in enumerate(immune[itype]):
                    self.data[chrom].remove(gene)
                    if i + 1 == len(immune[itype]) or gene.end - immune[itype][i + 1].start > maxgap:
                        ref_info = {'gene_type': f'{itype}_gene', 'transcripts': [t for gene in immune[itype][offset:i + 1] for t in gene.ref_transcripts]}
                        info = {'ID': f'{itype}_locus_{num[itype]}', 'strand': strand, 'chr': chrom, 'reference': ref_info}
                        start = immune[itype][offset].start
                        end = immune[itype][i].end
                        new_gene = Gene(start, end, info, self)
                        self.data[chrom].add(new_gene)
                        num[itype] += 1
                        offset = i + 1
    logger.info('collapsed %s immunoglobulin loci and %s T-cell receptor loci', num["IG"], num["TR"])


# io utility functions
@ experimental
def get_mutations_from_bam(bam_file, genome_file, region, min_cov=.05):
    '''not very efficient function to fetch mutations within a region from a bam file
    not exported so far'''
    mutations = dict()
    exons = []
    n = 0
    with AlignmentFile(bam_file, "rb") as align:
        for read in align.fetch(*region):
            n += 1

            exons.append(junctions_from_cigar(read.cigartuples, read.reference_start))
            mutations = get_mutations(read.cigartuples, read.query_sequence, read.reference_start, read.query_qualities)
            for pos, ref, alt, qual in mutations:
                mutations.setdefault(pos, {}).setdefault(alt, [0, ref, []])
                mutations[pos][alt][0] += 1
                if qual:
                    mutations[pos][alt][2].append(qual)
    if min_cov < 1:
        min_cov = n * min_cov

    mutations = {pos: v for pos, v in mutations.items() if sum(v[alt][0] for alt in v) > min_cov}
    with FastaFile(genome_file) as genome_fh:
        for pos, v in mutations.items():
            for alt in v.values():
                alt[1] = '' if alt[1] < 0 else genome_fh.fetch(region[0], pos, pos + alt[1])
            for transcript in exons:
                for exon in transcript:
                    if pos >= exon[0] and pos <= exon[1]:
                        mutations[pos]['cov'] = mutations[pos].get('cov', 0) + 1
    return mutations


def get_mutations(cigartuples, seq, ref_start, qual):
    'look up the bases affected by mutations as reported in the cigar string'
    # cigar numbers:
    # 012345678
    # MIDNSHP=X
    mutations = []
    ref_pos = ref_start
    seq_pos = 0
    for cigar in cigartuples:
        if cigar[0] in (1, 2, 8):  # I(ins), D(del) or X (missmatch):
            ref = -cigar[1] if cigar[0] == 1 else cigar[1]
            alt_base = '' if cigar[0] == 2 else seq[seq_pos:(seq_pos + cigar[1])]
            mutations.append((ref_pos, ref, alt_base, qual[seq_pos] if qual else None))
        if cigar[0] in (0, 2, 3, 7, 8):  # MDN=X -> move forward on reference
            ref_pos += cigar[1]
        if cigar[0] in (0, 1, 4, 7, 8):  # MIS=X -> move forward on seq
            seq_pos += cigar[1]
    return mutations


def aligned_part(cigartuples, is_reverse):
    "returns the interval of the transcript that is aligned (e.g. not clipped) according to cigar. Positions are according to transcript strand"
    start = end = 0
    for cigar in reversed(cigartuples) if is_reverse else cigartuples:
        if cigar[0] in (0, 1, 7, 8):  # MI=X -> move forward on read:
            end += cigar[1]
        elif cigar[0] in (4, 5):  # clipping at end
            if end > start:
                return (start, end)
            end += cigar[1]
            start = end
    return (start, end)  # clipping at begining or no clipping


def get_clipping(cigartuples, pos):
    if cigartuples[0][0] == 4:
        # clipping at the begining
        return (pos, -cigartuples[0][1])
    elif cigartuples[-1][0] == 4:
        # clipping at the end - get the reference position
        return (pos + sum(c[1] for c in cigartuples[:-1] if c[0] in (0, 2, 3, 7, 8)), cigartuples[-1][1])  # MDN=X -> move forward on reference:
    else:
        return None


def _set_alias(d, alias, required=True):
    for pref, alt in alias.items():
        alt = [a for a in alt if a in d]
        if pref not in d:
            try:
                d[pref] = next(d[a] for a in alt)
            except StopIteration:
                if not required:
                    continue
                logger.error('did not find alternative for %s- suggested terms are %s, but have only those keys: %s', pref, alt, list(d))
                raise
        for a in alt:
            d.pop(a, None)


# human readable output
def gene_table(self: Transcriptome, **filter_args):  # ideas: extra_columns
    '''Creates a gene summary table.

    Exports all genes within region to a table.

    :param filter_args: Parameters (e.g. "region", "query") are passed to Transcriptome.iter_genes.'''

    colnames = ['chr', 'start', 'end', 'strand', 'gene_id', 'gene_name', 'n_transcripts']
    rows = [(gene.chrom, gene.start, gene.end, gene.strand, gene.id, gene.name, gene.n_transcripts) for gene in self.iter_genes(**filter_args)]
    df = pd.DataFrame(rows, columns=colnames)
    return df


def transcript_table(self: Transcriptome, samples=None, groups=None, coverage=False, tpm=False, tpm_pseudocount=0, extra_columns=None, **filter_args):
    '''Creates a transcript table.

    Exports all transcript isoforms within region to a table.

    :param samples: provide a list of samples for which coverage / expression information is added.
    :param groups: provide groups as a dict (as from Transcriptome.groups()), for which coverage / expression information is added.
    :param coverage: If set, coverage information is added for specified samples / groups.
    :param tpm: If set, expression information (in tpm) is added for specified samples / groups.
    :param tpm_pseudocount: This value is added to the coverage for each transcript, before calculating tpm.
    :param extra_columns: Specify the additional information added to the table.
        These can be any transcript property as defined by the key in the transcript dict.
    :param filter_args: Parameters (e.g. "region", "query", "min_coverage",...) are passed to Transcriptome.iter_transcripts.'''

    if samples is None:
        if groups is None:
            samples = self.samples
        else:
            samples = []
    if groups is None:
        groups = {}
    if coverage is False and tpm is False:
        samples = []
        groups = {}
    if extra_columns is None:
        extra_columns = []
    if samples is None:
        samples = self.samples
    samples_set = set(samples)
    samples_set.update(*groups.values())
    assert all(s in self.samples for s in samples_set), 'Not all specified samples are known'
    if len(samples_set) == len(self.samples):
        all_samples = True
        sample_i = slice(None)
    else:
        all_samples = False
        sample_i = [i for i, sample in enumerate(self.samples) if sample in samples_set]

    if not isinstance(extra_columns, list):
        raise ValueError('extra_columns should be provided as list')

    colnames = ['chr', 'transcript_start', 'transcript_end', 'strand', 'gene_id', 'gene_name', 'transcript_nr',
                'transcript_length', 'num_exons', 'exon_starts', 'exon_ends', 'novelty_class', 'novelty_subclasses']
    colnames += extra_columns

    rows = []
    cov = []
    for gene, transcript_ids, transcripts in self.iter_transcripts(**filter_args, genewise=True):
        if sample_i:
            idx = (slice(None), transcript_ids) if all_samples else np.ix_(sample_i, transcript_ids)
            cov.append(gene.coverage[idx])
        for transcript_id, transcript in zip(transcript_ids, transcripts):
            exons = transcript['exons']
            trlen = sum(e[1]-e[0] for e in exons)
            nov_class, subcat = transcript['annotation']
            # subcat_string = ';'.join(k if v is None else '{}:{}'.format(k, v) for k, v in subcat.items())
            e_starts, e_ends = (','.join(str(exons[i][j]) for i in range(len(exons))) for j in range(2))
            row = [gene.chrom, exons[0][0], exons[-1][1], gene.strand, gene.id, gene.name, transcript_id, trlen, len(exons), e_starts, e_ends,
                   SPLICE_CATEGORY[nov_class], ','.join(subcat)]
            for k in extra_columns:
                val = transcript.get(k, 'NA')
                row.append(str(val) if isinstance(val, Iterable) else val)
            rows.append(row)

    # add coverage information
    df = pd.DataFrame(rows, columns=colnames)
    if cov:
        df_list = [df]
        cov = pd.DataFrame(np.concatenate(cov, 1).T, columns=self.samples if all_samples else [sample for i, sample in enumerate(self.samples) if i in sample_i])
        stab = self.sample_table.set_index('name')
        if samples:
            if coverage:
                df_list.append(cov[samples].add_suffix('_coverage'))
            if tpm:
                total = stab.loc[samples, 'nonchimeric_reads']+tpm_pseudocount*cov.shape[0]
                df_list.append(((cov[samples]+tpm_pseudocount)/total*1e6).add_suffix('_tpm'))
        if groups:
            cov_gr = pd.DataFrame({group_name: cov[sample].sum(1) for group_name, sample in groups.items()})
            if coverage:
                df_list.append(cov_gr.add_suffix('_sum_coverage'))
            if tpm:
                total = {group_name: stab.loc[sample, 'nonchimeric_reads'].sum()+tpm_pseudocount*cov.shape[0] for group_name, sample in groups.items()}
                df_list.append(((cov_gr+tpm_pseudocount)/total*1e6).add_suffix('_sum_tpm'))
        df = pd.concat(df_list, axis=1)

    return df


@ experimental
def chimeric_table(self: Transcriptome, region=None, query=None):  # , star_chimeric=None, illu_len=200):
    '''Creates a chimeric table

    This table contains relevant infos about breakpoints and coverage for chimeric genes.

    :param region: Specify the region, either as (chr, start, end) tuple or as "chr:start-end" string. If omitted specify the complete genome.
    :param query: Specify transcript filter query.
    '''
    # todo: correct handeling of three part fusion events not yet implemented
    # todo: ambiguous alignment handling not yet implemented

    # if star_chimeric is None:
    #    star_chimeric=dict()
    # assert isinstance(star_chimeric, dict)
    if region is not None or query is not None:
        raise NotImplementedError
    chim_tab = list()
    for bp, chimeric in self.chimeric.items():
        cov = tuple(sum(c.get(sample, 0) for c, _ in chimeric) for sample in self.samples)
        genes = [info[4] if info[4] is not None else 'intergenic' for info in chimeric[0][1]]
        for i, bp_i in enumerate(bp):
            chim_tab.append(('_'.join(genes),) + bp_i[: 3] + (genes[i],) + bp_i[3:] + (genes[i + 1],) + (sum(cov),) + cov)
    chim_tab = pd.DataFrame(chim_tab, columns=['name', 'chr1', 'strand1', 'breakpoint1', 'gene1', 'chr2', 'strand2',
                            'breakpoint2', 'gene2', 'total_cov'] + [s + '_cov' for s in self.infos['sample_table'].name])

    return chim_tab

    # todo: integrate short read coverage from star files
#   breakpoints = {}  # todo: this should be the isoseq breakpoints
#   offset = 10 + len(self.infos['sample_table'])
#   for sa_idx, sa in enumerate(star_chimeric):
#       star_tab = pd.read_csv(star_chimeric[sa], sep='\t')
#       for _, row in star_tab.iterrows():
#           if row['chr_donorA'] in breakpoints and row['chr_acceptorB'] in breakpoints:
#               idx1 = {bp.data for bp in breakpoints[row['chr_donorA']][row['brkpt_donorA']]}
#               if idx1:
#                   idx2 = {bp.data for bp in breakpoints[row['chr_acceptorB']][row['brkpt_acceptorB']]}
#                   idx_ol = {idx for idx, snd in idx2 if (idx, not snd) in idx1}
#                   for idx in idx_ol:
#                       chim_tab[idx][offset + sa_idx] += 1
#
#    chim_tab = pd.DataFrame(chim_tab, columns=['transcript_id', 'len', 'gene1', 'part1', 'breakpoint1', 'gene2', 'part2', 'breakpoint2',
#                            'total_cov'] + [s + '_cov' for s in self.infos['sample_table'].name] + [s + "_shortread_cov" for s in star_chimeric])
#    return chim_tab


def openfile(fn, gzip=False):
    if gzip:
        return gziplib.open(fn, 'wt')
    else:
        return open(fn, 'w', encoding="utf8")


def write_gtf(self: Transcriptome, fn, source='isotools', gzip=False, **filter_args):
    '''
    Exports the transcripts in gtf format to a file.

    :param fn: The filename to write the gtf.
    :param source: String for the source column of the gtf file.
    :param region: Specify genomic region to export to gtf. If omitted, export whole genome.
    :param gzip: Compress the output as gzip.
    :param filter_args: Specify transcript filter query.
    '''

    with openfile(fn, gzip) as f:
        logger.info('writing %sgtf file to %s', "gzip compressed " if gzip else "", fn)
        for gene, transcript_ids, _ in self.iter_transcripts(genewise=True, **filter_args):
            lines = gene._to_gtf(transcript_ids=transcript_ids, source=source)
            f.write('\n'.join(('\t'.join(str(field) for field in line) for line in lines)) + '\n')


def write_fasta(self: Transcriptome, genome_fn, fn, gzip=False, reference=False, protein=False, **filter_args):
    '''
    Exports the transcript sequences in fasta format to a file.

    :param genome_fn: Path to the genome in fastA format.
    :param reference: Specify whether the sequence is fetched for reference transcripts (True), or long read transcripts (False, default).
    :param protein: Return protein sequences (ORF) instead of transcript sequences.
    :param fn: The filename to write the fasta.
    :param gzip: Compress the output as gzip.
    :param filter_args: Additional filter arguments (e.g. "region", "gois", "query") are passed to iter_transcripts.
    '''

    with openfile(fn, gzip) as f:
        logger.info('writing %sfasta file to %s', "gzip compressed " if gzip else "", fn)
        for gene, transcript_ids, _ in self.iter_transcripts(genewise=True, **filter_args):
            tr_seqs = gene.get_sequence(genome_fn, transcript_ids, reference=reference, protein=protein)
            if len(tr_seqs) > 0:
                f.write('\n'.join(f'>{gene.id}_{k} gene={gene.name}\n{v}' for k,v in tr_seqs.items()) + '\n')


def export_alternative_splicing(self: Transcriptome, out_dir, out_format='mats', reference=False, min_total=100,
                                min_alt_fraction=.1, samples=None, region=None, query=None, progress_bar=True):
    '''Exports alternative splicing events defined by the transcriptome.

    This is intended to integrate splicing event analysis from short read data.
    Tools for short read data implement different formats for the import of events.
    These formats include several files and depend on specific file naming.
    Currently only MISO (out_format="miso") and rMATS (out_format='mats') are supported.
    rMATS is recommended.

    :param out_dir: Path to the directory where the event files are written to.
    :param out_format: Specify the output format. Must be either "miso" or "mats".
    :param min_total: Minimum total coverage over all selected samples.
    :param region: Specify the region, either as (chr, start, end) tuple or as "chr:start-end" string.
        If omitted specify the complete genome.
    :param query: Specify gene filter query.
    :param progress_bar: Show the progress.
    :param reference: If set to True, the LRTS data is ignored and the events are called from the reference.
        In this case the following parameters are ignored
    :param samples: Specify the samples to consider
    :param min_total: Minimum total coverage over all selected samples.
    :param min_alt_fraction: Minimum fraction of reads supporting the alternative.'''
    if out_format == 'miso':
        file_name = 'isotools_miso_{}.gff'
        alt_splice_export = _miso_alt_splice_export
    elif out_format == 'mats':
        file_name = 'fromGTF.{}.txt'
        alt_splice_export = _mats_alt_splice_export
    else:
        raise ValueError('out_format must be "miso" or "mats"')

    types = {'ES': 'SE', '3AS': 'A3SS', '5AS': 'A5SS', 'IR': 'RI', 'ME': 'MXE'}  # it looks like these are the "official" identifiers?
    out_file = {st: out_dir + '/' + file_name.format(st) for st in types.values()}
    if samples is None:
        samples = self.samples
    assert all(sample in self.samples for sample in samples), 'not all specified samples found'
    sample_dict = {sample: i for i, sample in enumerate(self.samples)}
    sidx = np.array([sample_dict[sample] for sample in samples])

    assert 0 < min_alt_fraction < .5, 'min_alt_fraction must be > 0 and < 0.5'
    count = {st: 0 for st in types.values()}
    with ExitStack() as stack:
        fh = {st: stack.enter_context(open(out_file[st], 'w')) for st in out_file}
        if out_format == 'mats':  # prepare mats header
            base_header = ['ID', 'GeneID', 'geneSymbol', 'chr', 'strand']
            add_header = {'SE': ['exonStart_0base', 'exonEnd', 'upstreamES', 'upstreamEE', 'downstreamES', 'downstreamEE'],
                          'RI': ['riExonStart_0base', 'riExonEnd', 'upstreamES', 'upstreamEE', 'downstreamES', 'downstreamEE'],
                          'MXE': ['1stExonStart_0base', '1stExonEnd', '2ndExonStart_0base', '2ndExonEnd', 'upstreamES',
                                  'upstreamEE', 'downstreamES', 'downstreamEE'],
                          'A3SS': ['longExonStart_0base', 'longExonEnd', 'shortES', 'shortEE', 'flankingES', 'flankingEE'],
                          'A5SS': ['longExonStart_0base', 'longExonEnd', 'shortES', 'shortEE', 'flankingES', 'flankingEE']}
            for st in fh:
                fh[st].write('\t'.join(base_header + add_header[st]) + '\n')
        for gene in self.iter_genes(region, query, progress_bar=progress_bar):
            if reference and not gene.is_annotated:
                continue
            elif not reference and gene.coverage[sidx, :].sum() < min_total:
                continue

            seg_graph = gene.ref_segment_graph if reference else gene.segment_graph
            for setA, setB, nodeX, nodeY, splice_type in seg_graph.find_splice_bubbles(types=('ES', '3AS', '5AS', 'IR', 'ME')):
                if not reference:
                    junction_cov = gene.coverage[np.ix_(sidx, setA)].sum(1)
                    total_cov = gene.coverage[np.ix_(sidx, setB)].sum(1) + junction_cov
                    if total_cov.sum() < min_total or (not min_alt_fraction < junction_cov.sum() / total_cov.sum() < 1 - min_alt_fraction):
                        continue
                st = types[splice_type]
                lines = alt_splice_export(setA, setB, nodeX, nodeY, st, reference, gene, count[st])
                if lines:
                    count[st] += len(lines)
                    fh[st].write('\n'.join(('\t'.join(str(field) for field in line) for line in lines)) + '\n')


def _miso_alt_splice_export(setA, setB, nodeX, nodeY, splice_type, reference, gene, offset):
    seg_graph = gene.ref_segment_graph if reference else gene.segment_graph

    event_id = f'{gene.chrom}:{seg_graph[nodeX].end}-{seg_graph[nodeY].start}_st'
    # TODO: Mutually exclusives extend beyond nodeY - and have potentially multiple A "mRNAs"
    # TODO: is it possible to extend exons at nodeX and Y - if all/"most" transcript from setA and B agree?
    # if st=='ME':
    #    nodeY=min(seg_graph._pas[setA])
    lines = []
    lines.append([gene.chrom, splice_type, 'gene', seg_graph[nodeX].start, seg_graph[nodeY].end, '.', gene.strand, '.', f'ID={event_id};gene_name={gene.name};gene_id={gene.id}'])
    # lines.append((gene.chrom, st, 'mRNA', seg_graph[nodeX].start, seg_graph[nodeY].end, '.',gene.strand, '.', f'Parent={event_id};ID={event_id}.A'))
    # lines.append((gene.chrom, st, 'exon', seg_graph[nodeX].start, seg_graph[nodeX].end, '.',gene.strand, '.', f'Parent={event_id}.A;ID={event_id}.A.up'))
    # lines.append((gene.chrom, st, 'exon', seg_graph[nodeY].start, seg_graph[nodeY].end, '.',gene.strand, '.', f'Parent={event_id}.A;ID={event_id}.A.down'))
    for i, exons in enumerate({tuple(seg_graph._get_all_exons(nodeX, nodeY, transcript)) for transcript in setA}):
        lines.append((gene.chrom, splice_type, 'mRNA', exons[0][0], exons[-1][1], '.', gene.strand, '.', f'Parent={event_id};ID={event_id}.A{i}'))
        lines[0][3] = min(lines[0][3], lines[-1][3])
        lines[0][4] = max(lines[0][4], lines[-1][4])
        for j, exon in enumerate(exons):
            lines.append((gene.chrom, splice_type, 'exon', exon[0], exon[1], '.', gene.strand, '.', f'Parent={event_id}.A{i};ID={event_id}.A{i}.{j}'))
    for i, exons in enumerate({tuple(seg_graph._get_all_exons(nodeX, nodeY, transcript)) for transcript in setB}):
        lines.append((gene.chrom, splice_type, 'mRNA', exons[0][0], exons[-1][1], '.', gene.strand, '.', f'Parent={event_id};ID={event_id}.B{i}'))
        lines[0][3] = min(lines[0][3], lines[-1][3])
        lines[0][4] = max(lines[0][4], lines[-1][4])
        for j, exon in enumerate(exons):
            lines.append((gene.chrom, splice_type, 'exon', exon[0], exon[1], '.', gene.strand, '.', f'Parent={event_id}.B{i};ID={event_id}.B{i}.{j}'))
    return lines


def _mats_alt_splice_export(setA, setB, nodeX, nodeY, st, reference, gene, offset,  use_top_isoform=True, use_top_alternative=True):
    # use_top_isoform and use_top_alternative are ment to simplify the output, in order to not confuse rMATS with to many options
    # 'ID','GeneID','geneSymbol','chr','strand'
    # and ES/EE for the relevant exons
    # in case of 'SE':['skipped', 'upstream', 'downstream'],
    # in case of 'RI':['retained', 'upstream', 'downstream'],
    # in case of 'MXE':['1st','2nd', 'upstream', 'downstream'],
    # in case of 'A3SS':['long','short', 'flanking'],
    # in case of 'A5SS':['long','short', 'flanking']}
    # upstream and downstream/ 1st/snd are with respect to the genome strand
    # offset is the event id offset
    seg_graph = gene.ref_segment_graph if reference else gene.segment_graph

    lines = []
    if gene.chrom[: 3] != 'chr':
        chrom = 'chr' + gene.chrom
    else:
        chrom = gene.chrom
    if use_top_isoform:  # use flanking exons from top isoform
        all_transcript_ids = setA+setB
        if not reference:
            # most covered isoform
            top_isoform = all_transcript_ids[gene.coverage[:, all_transcript_ids].sum(0).argmax()]  # top covered isoform across all samples
            nodeX_start = seg_graph._get_exon_start(top_isoform, nodeX)
            nodeY_end = seg_graph._get_exon_end(top_isoform, nodeY)
        else:
            # for reference: most frequent
            nodeX_start = Counter([seg_graph._get_exon_start(n, nodeX) for n in all_transcript_ids]).most_common(1)[0][0]
            nodeY_end = Counter([seg_graph._get_exon_end(n, nodeY) for n in all_transcript_ids]).most_common(1)[0][0]

        exonsA = ((seg_graph[nodeX_start].start, seg_graph[nodeX].end), (seg_graph[nodeY].start, seg_graph[nodeY_end].end))  # flanking/outer "exons" of setA()
    else:
        exonsA = ((seg_graph[nodeX].start, seg_graph[nodeX].end), (seg_graph[nodeY].start, seg_graph[nodeY].end))  # flanking/outer "exons" of setA

    # _get_all_exons does not extend the exons beyond the nodeX/Y
    alternatives = [tuple(seg_graph._get_all_exons(nodeX, nodeY, b_tr)) for b_tr in setB]
    if use_top_alternative:
        if reference:
            c = Counter(alternatives)
        else:
            weights = [gene.coverage[:, b_tr].sum() for b_tr in setB]
            c = Counter()
            for alt, w in zip(alternatives, weights):
                c.update({alt: w})
        alternatives = [c.most_common(1)[0][0]]
    else:
        alternatives = set(alternatives)
    for exonsB in alternatives:
        exons_sel = []
        if st in ['A3SS', 'A5SS'] and len(exonsB) == 2:
            if exonsA[0][1] == exonsB[0][1]:  # A5SS on - strand or A3SS on + strand
                exons_sel.append([(exonsB[1][0], exonsA[1][1]), exonsA[1], exonsA[0]])  # long short flanking
            else:  # A5SS on + strand or A3SS on - strand
                exons_sel.append([(exonsA[0][0], exonsB[0][1]), exonsA[0], exonsA[1]])  # long short flanking
        elif st == 'SE' and len(exonsB) == 3:
            # just to be sure everything is consistent
            assert exonsA[0][1] == exonsB[0][1] and exonsA[1][0] == exonsB[2][0], f'invalid exon skipping {exonsA} vs {exonsB}'
            # e_order = (1, 0, 2) if gene.strand == '+' else (1, 2, 0)
            exons_sel.append([exonsB[i] for i in (1, 0, 2)])  # skipped, upstream, downstream
        elif st == 'RI' and len(exonsB) == 1:
            exons_sel.append([(exonsA[0][0], exonsA[1][1]), exonsA[0], exonsA[1]])  # retained, upstream, downstream
            # if gene.strand == '+' else [exonsB[0], exonsA[1], exonsA[0]])
        elif st == 'MXE' and len(exonsB) == 3:
            # nodeZ=next(idx for idx,n in enumerate(seg_graph) if n.start==exonsB[-1][0])
            # multiple exonA possibilities, so we need to check all of them
            for exonsA_ME in {tuple(seg_graph._get_all_exons(nodeX, nodeY, a_tr)) for a_tr in setA}:
                if len(exonsA_ME) != 3:  # complex events are not possible in rMATS
                    continue
                assert exonsA_ME[0] == exonsB[0] and exonsA_ME[2] == exonsB[2]  # should always be true
                # assert exonsA_ME[0][1] == exonsA[0][1] and exonsA_ME[2][0] == exonsA[1][0]  # should always be true
                # '1st','2nd', 'upstream', 'downstream'
                exons_sel.append([exonsB[1], exonsA_ME[1], exonsA[0], exonsA[1]])
        for exons in exons_sel:  # usually there is only one rMATS event per exonB, but for MXE we may get several
            # lines.append([f'"{gene.id}"', f'"{gene.name}"', chrom, gene.strand] + [pos for exon in exons for pos in ((exon[1],exon[0]) if gene.strand == '-' else exon)])
            lines.append([f'"{gene.id}"', f'"{gene.name}"', chrom, gene.strand] + [pos for exon in exons for pos in exon])  # no need to reverse the order of exon start/end
    return [[offset + count] + l for count, l in enumerate(lines)]


def get_gff_chrom_dict(gff: TabixFile, chromosomes):
    'fetch chromosome ids - in case they use ids in gff for the chromosomes'
    chrom = {}
    for c in gff.contigs:
        # loggin.debug ("---"+c)
        for line in gff.fetch(c, 1, 2):  # chromosomes span the entire chromosome, so they can be fetched like that
            if line[1] == "C":
                ls = line.split(sep="\t")
                if ls[2] == "region":
                    info = dict([pair.split("=")
                                 for pair in ls[8].split(";")])
                    if "chromosome" in info.keys():
                        if chromosomes is None or info["chromosome"] in chromosomes:
                            chrom[ls[0]] = info["chromosome"]
                        break

        else:  # no specific regions entries - no aliases
            if chromosomes is None or c in chromosomes:
                chrom[c] = c
    gff.seek(0)
    return chrom


class IntervalArray:
    '''drop in replacement for the interval tree during construction, with faster lookup'''

    def __init__(self, total_size, bin_size=1e4):
        self.obj: dict[str, Interval] = {}
        self.data: list[set[int]] = [set() for _ in range(int((total_size) // bin_size) + 1)]
        self.bin_size = bin_size

    def overlap(self, begin, end):
        try:
            candidates = {obj_id
                          for idx in range(int(begin // self.bin_size), int(end // self.bin_size) + 1)
                          for obj_id in self.data[idx]}
        except IndexError:
            logger.error('requesting interval between %s and %s, but array is allocated only until position %s', begin, end, len(self.data)*self.bin_size)
            raise
        # this assumes object has range obj[0] to obj[1]
        return (self.obj[obj_id] for obj_id in candidates if has_overlap((begin, end), self.obj[obj_id]))

    def add(self, obj: Interval):
        self.obj[id(obj)] = obj
        try:
            for idx in range(int(obj.begin // self.bin_size), int(obj.end // self.bin_size) + 1):
                self.data[idx].add(id(obj))
        except IndexError:
            logger.error('adding interval from %s to %s, but array is allocated only until position %s', obj.begin, obj.end, len(self.data)*self.bin_size)
            raise

    def __len__(self):
        return len(self.obj)

    def __iter__(self):
        return (v for v in self.obj.values())
