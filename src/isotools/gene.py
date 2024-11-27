from __future__ import annotations
from operator import itemgetter
from intervaltree import Interval
from collections.abc import Iterable
from scipy.stats import chi2_contingency
from scipy.signal import find_peaks
from Bio.Seq import reverse_complement, translate
from Bio.Data.CodonTable import TranslationError
from pysam import FastaFile
import numpy as np
import copy
import itertools
from cpmodule import fickett, FrameKmer  # this is from the CPAT module
from .splice_graph import SegmentGraph
from .short_read import Coverage
from ._transcriptome_filter import SPLICE_CATEGORY
from ._utils import pairwise, _filter_event, find_orfs, DEFAULT_KOZAK_PWM, kozak_score, smooth, get_quantiles, \
    _filter_function, pairwise_event_test, prepare_contingency_table, cmp_dist
from typing import Any, Literal, Optional, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from .transcriptome import Transcriptome
    from pandas import Series
    from ._utils import ASEvent

import logging
logger = logging.getLogger('isotools')


class SQANTI_classification(TypedDict):
    dist_to_CAGE_peak: int
    within_CAGE_peak: bool
    dist_to_polyA_site: int
    within_polyA_site: bool
    polyA_motif: str
    polyA_dist: int
    polyA_motif_found: bool
    ratio_TSS: float

class Transcript(TypedDict, total=False):
    chr: str
    strand: Literal['+', '-']
    exons: list[tuple[int, int]]
    coverage: dict[str, int]
    'The coverage of the transcript in each sample.'
    TSS: dict[str, dict[int, int]]
    'The TSS of each sample with their coverage.'
    PAS: dict[str, dict[int, int]]
    'The PAS of each sample with their coverage.'
    clipping: dict[str, dict[str, int]]
    annotation: tuple[int, dict[str, Any]] # TODO: Switch the dict to a TypedDict, Replace novelty with Enum
    "The annotation of the transcript. The first element is the novelty class (0=FSM,1=ISM,2=NIC,3=NNC,4=Novel gene), the second a dictionary with the subcategories."
    reads: list[str]
    'TODO: This seems to be inconsistent. Sometimes it is a list of read names, sometimes a dict with sample names as keys and the lists as values.'
    novel_splice_sites: list[int]
    TSS_unified: dict[str, dict[int, int]]
    PAS_unified: dict[str, dict[int, int]]
    direct_repeat_len: list[int]
    downstream_A_content: float
    long_intron_chimeric: Any
    CDS: tuple[int, int]
    ORF: tuple[int, int, dict[str, Any]]
    fuzzy_junction: Any
    sqanti_classification: SQANTI_classification

class RefTranscript(TypedDict, total=False):
    transcript_id: str
    transcript_type: str
    transcript_name: str
    transcript_support_level: str
    exons: list[tuple[int, int]]
    CDS: tuple[int, int]

class ReferenceData(TypedDict, total=False):
    segment_graph: SegmentGraph
    transcripts: list[RefTranscript]

class GeneData(TypedDict, total=False):
    ID: str
    name: str
    chr: str
    strand: Literal['+', '-']
    short_reads: list[Coverage]
    coverage: np.ndarray
    reference: ReferenceData
    segment_graph: SegmentGraph
    transcripts: list[Transcript]

class Gene(Interval):
    'This class stores all gene information and transcripts. It is derived from intervaltree.Interval.'
    required_infos = ['ID', 'chr', 'strand']

    data: GeneData
    _transcriptome: 'Transcriptome'

    # initialization
    def __new__(cls, begin, end, data: GeneData, transcriptome):
        return super().__new__(cls, begin, end, data)  # required as Interval (and Gene) is immutable

    def __init__(self, begin, end, data: GeneData, transcriptome: Transcriptome):
        self._transcriptome = transcriptome

    def __str__(self):
        return 'Gene {} {}({}), {} reference transcripts, {} expressed transcripts'.format(
            self.name, self.region, self.strand, self.n_ref_transcripts, self.n_transcripts)

    def __repr__(self):
        return object.__repr__(self)

    from ._gene_plots import sashimi_plot, gene_track, sashimi_plot_short_reads, sashimi_figure, plot_domains
    from .domains import add_interpro_domains

    def short_reads(self, idx):
        '''Returns the short read coverage profile for a short read sample.

        :param idx: The index of the short read sample.
        :returns: The short read coverage profile.'''

        try:
            # raises key_error if no short reads added
            return self.data['short_reads'][idx]
        except (KeyError, IndexError):
            srdf = self._transcriptome.infos['short_reads']
            self.data.setdefault('short_reads', [])
            for i in range(len(self.data['short_reads']), len(srdf)):
                self.data['short_reads'].append(Coverage.from_bam(srdf.file[i], self))
        return self.data['short_reads'][idx]

    def correct_fuzzy_junctions(self, transcript: Transcript, size, modify=True):
        '''Corrects for splicing shifts.

         This function looks for "shifted junctions", e.g. same difference compared to reference annotation at both donor and acceptor)
         presumably caused by ambiguous alignments. In these cases the positions are adapted to the reference position (if modify is set).

         :param transcript_id: The index of the transcript to be checked.
         :param size: The maximum shift to be corrected.
         :param modify: If set, the exon positions are corrected according to the reference.
         :returns: A dictionary with the exon id as keys and the shifted bases as values.'''

        exons = transcript['exons']
        shifts = self.ref_segment_graph.fuzzy_junction(exons, size)
        if shifts and modify:
            for i, sh in shifts.items():
                if exons[i][0] <= exons[i][1] + sh and exons[i + 1][0] + sh <= exons[i + 1][1]:
                    exons[i][1] += sh
                    exons[i + 1][0] += sh
            transcript['exons'] = [e for e in exons if e[0] < e[1]]  # remove zero length exons
        return shifts

    def _to_gtf(self, transcript_ids, ref_transcript_ids=None, source='isoseq', ref_source='annotation'):
        '''Creates the gtf lines of the gene as strings.'''
        donotshow = {'transcripts', 'short_exons', 'segment_graph'}
        info = {'gene_id': self.id, 'gene_name': self.name}
        lines = [None]
        starts = []
        ends = []
        ref_fsm = []
        for i in transcript_ids:
            transcript = self.transcripts[i]
            info['transcript_id'] = f'{info["gene_id"]}_{i}'
            starts.append(transcript['exons'][0][0] + 1)
            ends.append(transcript['exons'][-1][1])
            transcript_info = info.copy()
            if 'downstream_A_content' in transcript:
                transcript_info['downstream_A_content'] = f'{transcript["downstream_A_content"]:0.3f}'
            if transcript['annotation'][0] == 0:  # FSM
                refinfo = {}
                for refid in transcript['annotation'][1]['FSM']:
                    ref_fsm.append(refid)
                    for k in self.ref_transcripts[refid]:
                        if k == 'exons':
                            continue
                        elif k == 'CDS':
                            if self.strand == '+':
                                cds_start, cds_end = self.ref_transcripts[refid]['CDS']
                            else:
                                cds_end, cds_start = self.ref_transcripts[refid]['CDS']
                            refinfo.setdefault('CDS_start', []).append(str(cds_start))
                            refinfo.setdefault('CDS_end', []).append(str(cds_end))
                        else:
                            refinfo.setdefault(k, []).append(str(self.ref_transcripts[refid][k]))
                for k, vlist in refinfo.items():
                    transcript_info[f'ref_{k}'] = ','.join(vlist)
            else:
                transcript_info['novelty'] = ','.join(k for k in transcript['annotation'][1])
            lines.append((self.chrom, source, 'transcript', transcript['exons'][0][0] + 1, transcript['exons'][-1][1], '.',
                         self.strand, '.', '; '.join(f'{k} "{v}"' for k, v in transcript_info.items())))
            noncanonical = transcript.get('noncanonical_splicing', [])
            for enr, pos in enumerate(transcript['exons']):
                exon_info = info.copy()
                exon_info['exon_id'] = f'{info["gene_id"]}_{i}_{enr}'
                if enr in noncanonical:
                    exon_info['noncanonical_donor'] = noncanonical[enr][:2]
                if enr+1 in noncanonical:
                    exon_info['noncanonical_acceptor'] = noncanonical[enr+1][2:]
                lines.append((self.chrom, source, 'exon', pos[0] + 1, pos[1], '.', self.strand, '.', '; '.join(f'{k} "{v}"' for k, v in exon_info.items())))
        if ref_transcript_ids:
            # add reference transcripts not covered by FSM
            for i, transcript in enumerate(self.ref_transcripts):
                if i in ref_fsm:
                    continue
                starts.append(transcript['exons'][0][0] + 1)
                ends.append(transcript['exons'][-1][1])
                info['transcript_id'] = f'{info["gene_id"]}_ref{i}'
                refinfo = info.copy()
                for k in transcript:
                    if k == 'exons':
                        continue
                    elif k == 'CDS':
                        if self.strand == '+':
                            cds_start, cds_end = transcript['CDS']
                        else:
                            cds_end, cds_start = transcript['CDS']
                        refinfo['CDS_start'] = str(cds_start)
                        refinfo['CDS_end'] = str(cds_end)
                    else:
                        refinfo.setdefault(k, []).append(str(transcript[k]))
                    lines.append((self.chrom, ref_source, 'transcript', transcript['exons'][0][0] + 1, transcript['exons'][-1][1], '.',
                                  self.strand, '.', '; '.join(f'{k} "{v}"' for k, v in refinfo.items())))
                    for enr, pos in enumerate(transcript['exons']):
                        exon_info = info.copy()
                        exon_id = f'{info["gene_id"]}_ref{i}_{enr}'
                        lines.append((self.chrom, ref_source, 'exon', pos[0] + 1, pos[1], '.', self.strand, '.', f'exon_id "{exon_id}"'))

        if len(lines) > 1:
            # add gene line
            if 'reference' in self.data:
                # add reference gene specific fields
                info.update({k: v for k, v in self.data['reference'].items() if k not in donotshow})
            lines[0] = (self.chrom, source, 'gene', min(starts), max(ends), '.', self.strand, '.', '; '.join(f'{k} "{v}"' for k, v in info.items()))
            return lines
        return []

    def add_noncanonical_splicing(self, genome_fh):
        '''Add information on noncanonical splicing.

        For all transcripts of the gene, scan for noncanonical (i.e. not GT-AG) splice sites.
        If noncanonical splice sites are present, the corresponding intron index (in genomic orientation) and the sequence
        i.e. the di-nucleotides of donor and acceptor as XX-YY string are stored in the "noncannoncical_splicing" field of the transcript dicts.
        True noncanonical splicing is rare, thus it might indicate technical artifacts (template switching, misalignment, ...)

        :param genome_fh: A file handle of the genome fastA file.'''
        ss_seq = {}
        for transcript in self.transcripts:
            pos = [(transcript['exons'][i][1], transcript['exons'][i + 1][0] - 2) for i in range(len(transcript['exons']) - 1)]
            new_ss_seq = {site: genome_fh.fetch(self.chrom, site, site + 2).upper() for intron in pos for site in intron if site not in ss_seq}
            if new_ss_seq:
                ss_seq.update(new_ss_seq)

            if self.strand == '+':
                sj_seq = [ss_seq[d] + ss_seq[a] for d, a in pos]
            else:
                sj_seq = [reverse_complement(ss_seq[d] + ss_seq[a]) for d, a in pos]

            nc = [(i, seq) for i, seq in enumerate(sj_seq) if seq != 'GTAG']
            if nc:
                transcript['noncanonical_splicing'] = nc

    def add_direct_repeat_len(self, genome_fh, delta=15, max_mm=2, wobble=2):
        '''Computes direct repeat length.

        This function counts the number of consecutive equal bases at donor and acceptor sites of the splice junctions.
        This information is stored in the "direct_repeat_len" filed of the transcript dictionaries.
        Direct repeats longer than expected by chance indicate template switching.

        :param genome_fh: The file handle to the genome fastA.
        :param delta: The maximum length of direct repeats that can be found.
        :param max_mm: The maximum length of direct repeats that can be found.
        :param wobble: The maximum length of direct repeats that can be found.'''

        intron_seq = {}
        score = {}

        for transcript in self.transcripts:
            for intron in ((transcript['exons'][i][1], transcript['exons'][i + 1][0]) for i in range(len(transcript['exons']) - 1)):
                for pos in intron:
                    try:
                        intron_seq.setdefault(pos, genome_fh.fetch(self.chrom, pos - delta, pos + delta))
                    except (ValueError, IndexError):  # N padding at start/end of the chromosomes
                        chr_len = genome_fh.get_reference_length(self.chrom)
                        seq = genome_fh.fetch(self.chrom, max(0, pos - delta), min(chr_len, pos + delta))
                        if pos - delta < 0:
                            seq = ''.join(['N'] * (pos - delta)) + seq
                        if pos + delta > chr_len:
                            seq += ''.join(['N'] * (pos + delta - chr_len))
                        intron_seq.setdefault(pos, seq)
                if intron not in score:
                    score[intron] = repeat_len(intron_seq[intron[0]], intron_seq[intron[1]], wobble=wobble, max_mm=max_mm)

        for transcript in self.transcripts:
            transcript['direct_repeat_len'] = [min(score[(exon_1[1], exon_2[0])], delta) for exon_1, exon_2 in pairwise(transcript['exons'])]

    def add_threeprime_a_content(self, genome_fh, length=30):
        '''Adds the information of the genomic A content downstream the transcript.

        High values of genomic A content indicate internal priming and hence genomic origin of the LRTS read.
        This function populates the 'downstream_A_content' field of the transcript dictionaries.

        :param geneome_fh: A file handle for the indexed genome fastA file.
        :param length: The length of the downstream region to be considered.
        '''
        a_content = {}
        for transcript in (t for tL in (self.transcripts, self.ref_transcripts) for t in tL):
            if self.strand == '+':
                pos = transcript['exons'][-1][1]
            else:
                pos = transcript['exons'][0][0] - length
            if pos not in a_content:
                seq = genome_fh.fetch(self.chrom, max(0, pos), pos + length)
                if self.strand == '+':
                    a_content[pos] = seq.upper().count('A') / length
                else:
                    a_content[pos] = seq.upper().count('T') / length
            transcript['downstream_A_content'] = a_content[pos]

    def add_sqanti_classification(self, transcript_id: int, classification_row: Series):
        assert transcript_id < self.n_transcripts, f'Transcript id {transcript_id} not found in gene {self.id}'
        infos = classification_row[[
            'dist_to_CAGE_peak',
            'within_CAGE_peak',
            'dist_to_polyA_site',
            'within_polyA_site',
            'polyA_motif',
            'polyA_dist',
            'polyA_motif_found',
            'ratio_TSS'
        ]].to_dict()
        self.transcripts[transcript_id]['sqanti_classification'] = infos

    def get_sequence(self, genome_fh, transcript_ids=None, reference=False, protein=False):
        '''Returns the nucleotide sequence of the specified transcripts.

        :param genome_fh: The path to the genome fastA file, or FastaFile handle.
        :param transcript_ids: List of transcript ids for which the sequence are requested.
        :param reference: Specify whether the sequence is fetched for reference transcripts (True)
            or long read transcripts (False, default).
        :param protein: Return protein sequences instead of transcript sequences.
        :returns: A dictionary of transcript ids and their sequences.
        '''

        trL = [(i, transcript) for i, transcript in enumerate(self.ref_transcripts if reference else self.transcripts) if transcript_ids is None or i in transcript_ids]
        if not trL:
            return {}
        pos = (min(transcript['exons'][0][0] for _, transcript in trL), max(transcript['exons'][-1][1] for _, transcript in trL))
        try:  # assume its a FastaFile file handle
            seq = genome_fh.fetch(self.chrom, *pos)
        except AttributeError:
            genome_fn = genome_fh
            with FastaFile(genome_fn) as genome_fh:
                seq = genome_fh.fetch(self.chrom, *pos)
        transcript_seqs = {}
        for i, transcript in trL:
            transcript_seq = ''
            for exon in transcript['exons']:
                transcript_seq += seq[exon[0]-pos[0]:exon[1]-pos[0]]
            transcript_seqs[i] = transcript_seq

        if self.strand == '-':
            transcript_seqs = {i: reverse_complement(ts) for i, ts in transcript_seqs.items()}
        if not protein:
            return transcript_seqs
        prot_seqs = {}
        for i, transcript in trL:
            orf = transcript.get("CDS", transcript.get("ORF"))
            if not orf:
                continue
            pos = sorted(self.find_transcript_positions(i, orf[:2], reference=reference))
            try:
                prot_seqs[i] = translate(transcript_seqs[i][pos[0]:pos[1]], cds=True)
            except TranslationError:
                logger.warning(f'CDS sequence of {self.id} {"reference" if reference else ""} transcript {i} cannot be translated.')
        return prot_seqs

    def _get_ref_cds_pos(self, transcript_ids=None):
        '''find the position of annotated CDS initiation '''
        if transcript_ids is None:
            transcript_ids = range(len(self.transcripts))
        reverse_strand = self.strand == '-'
        utr, anno_cds = {}, {}
        match_cds = {}
        for i, transcript in enumerate(self.ref_transcripts):
            if 'CDS' in transcript:
                anno_cds[i] = transcript['CDS']
                if not reverse_strand:
                    utr[i] = [exon for exon in transcript['exons'] if exon[0] < transcript['CDS'][0]]
                else:
                    utr[i] = [exon for exon in transcript['exons'] if exon[1] > transcript['CDS'][1]]

        for transcript_id in transcript_ids:
            transcript = self.transcripts[transcript_id]
            match_cds[transcript_id] = {}
            for i, reg in utr.items():
                if not any(start <= anno_cds[i][reverse_strand] <= end for start, end in transcript['exons']):  # no overlap of CDS init with exons
                    continue
                if not reverse_strand:
                    to_check = zip(pairwise(reg), pairwise(transcript['exons']))
                else:
                    to_check = zip(pairwise((end, start) for start, end in reversed(reg)), pairwise((e, s) for s, e in reversed(transcript['exons'])))
                for ((e1reg, e2reg), (e1, e2)) in to_check:
                    if (e1reg[1] != e1[1] or e2reg[0] != e2[0]):
                        break
                else:
                    pos = self.find_transcript_positions(transcript_id, anno_cds[i], reference=False)[reverse_strand]
                    match_cds[transcript_id].setdefault(pos, []).append(i)
        return match_cds

    def add_orfs(self, genome_fh, tr_filter={}, reference=False, minlen=300, min_kozak=None, max_5utr_len=0, prefer_annotated_init=True,  start_codons=["ATG"],
                 stop_codons=['TAA', 'TAG', 'TGA'], kozak_matrix=DEFAULT_KOZAK_PWM, get_fickett=True, coding_hexamers=None, noncoding_hexamers=None):
        '''Predict the CDS for each transcript.

        For each transcript, one ORF is selected as the coding sequence. Depending on the parameters, this is either the first ORF
        (sequence starting with start_condon, and ending with in frame stop codon), or the longest ORF,
        starting with a codon that is annotated as CDS initiation site in a reference transcript.
        The genomic and transcript positions of these codons, and the length of the ORF, as well as  the number of upstream start codons
        is added to the transcript properties transcript["ORF"]. Additionally, the Fickett score, and the hexamer score are computed. For the
        latter, hexamer frequencies in coding and noncoding transcripts are needed. See CPAT python module for prebuilt tables and
        instructions.
        :param tr_filter: dict with filtering parameters passed to iter_transcripts or iter_ref_transcripts
        :param min_len: Minimum length of the ORF, Does not apply to annotated initiation sites.
        :param min_kozak: Minimal score for translation initiation site. Does not apply to annotated initiation sites.
        :param max_5utr_len: Maximal length of the 5'UTR region. Does not apply to annotated initiation sites.
        :param prefer_annotated_init: If True, the initiation sites of annotated CDS are preferred.
        :param start_codons: List of base triplets that are considered as start codons. By default ['ATG'].
        :param stop_codons: List of base triplets that are considered as stop codons. By default ['TAA', 'TAG', 'TGA'].
        :param kozak_matrix: A PWM (log odds ratios) to compute the Kozak sequence similarity
        :param get_fickett: If true, the fickett score for the CDS is computed.
        :param coding_hexamers: The hexamer frequencies for coding sequences.
        :param noncoding_hexamers: The hexamer frequencies for non-coding sequences (background).'''
        if tr_filter:
            if reference:
                tr_dict = {i: self.ref_transcripts[i] for i in self.filter_ref_transcripts(**tr_filter)}
            else:
                tr_dict = {i: self.transcripts[i] for i in self.filter_transcripts(**tr_filter)}
        else:
            tr_dict = {i: transcript for i, transcript in enumerate(self.ref_transcripts if reference else self.transcripts)}
        assert min_kozak is None or kozak_matrix is not None, 'Kozak matrix missing for min_kozak'
        if not tr_dict:
            return
        if prefer_annotated_init:
            if reference:
                ref_cds = {}
                for i, transcript in tr_dict.items():
                    if 'CDS' not in transcript:
                        ref_cds[i] = {}
                    else:
                        pos = self.find_transcript_positions(i, transcript['CDS'], reference=True)[self.strand == '-']
                        ref_cds[i] = {pos: [i]}
            else:
                ref_cds = self._get_ref_cds_pos(transcript_ids=tr_dict.keys())

        for transcript_id, tr_seq in self.get_sequence(genome_fh, transcript_ids=tr_dict.keys(), reference=reference).items():
            orfs = find_orfs(
                    tr_seq, start_codons, stop_codons, ref_cds[transcript_id] if prefer_annotated_init else [])
            if not orfs:  # No ORF
                continue
            # select best ORF
            # prefer annotated CDS init
            anno_orfs = [orf for orf in orfs if bool(orf[6]) and orf[1] is not None]
            kozak = None
            if anno_orfs:
                start, stop, frame, seq_start, seq_end, uORFs, ref_transcript_ids = max(anno_orfs, key=lambda x: x[1]-x[0])
            else:
                valid_orfs = [orf for orf in orfs if orf[1] is not None and orf[1]-orf[0] > minlen and orf[0] <= max_5utr_len]
                if not valid_orfs:
                    continue
                if min_kozak is not None:
                    for orf in valid_orfs:
                        kozak = kozak_score(tr_seq, orf[0], kozak_matrix)
                        if kozak > min_kozak:
                            start, stop, frame, seq_start, seq_end, uORFs, ref_transcript_ids = orf
                            break
                    else:
                        continue
                else:
                    start, stop, frame, seq_start, seq_end, uORFs, ref_transcript_ids = valid_orfs[0]

                start, stop, frame, seq_start, seq_end, uORFs, ref_transcript_ids = valid_orfs[0]
            # if stop is None or stop - start < minlen:
            #    continue

            transcript = tr_dict[transcript_id]
            transcript_start = transcript['exons'][0][0]
            cum_exon_len = np.cumsum([end-start for start, end in transcript['exons']])  # cumulative exon length
            cum_intron_len = np.cumsum([0]+[end-start for (_, start), (end, _) in pairwise(transcript['exons'])])  # cumulative intron length
            if self.strand == '-':
                fwd_start, fwd_stop = cum_exon_len[-1]-stop, cum_exon_len[-1]-start
            else:
                fwd_start, fwd_stop = start, stop  # start/stop position wrt genomic fwd strand
            start_exon = next(i for i in range(len(cum_exon_len)) if cum_exon_len[i] >= fwd_start)
            stop_exon = next(i for i in range(start_exon, len(cum_exon_len)) if cum_exon_len[i] >= fwd_stop)
            genome_pos = (transcript_start+fwd_start+cum_intron_len[start_exon],
                          transcript_start+fwd_stop+cum_intron_len[stop_exon])
            dist_pas = 0  # distance of termination codon to last upstream splice site
            if self.strand == '+' and stop_exon < len(cum_exon_len)-1:
                dist_pas = cum_exon_len[-2]-fwd_stop
            if self.strand == '-' and start_exon > 0:
                dist_pas = fwd_start-cum_exon_len[0]
            orf_dict = {"5'UTR": start, 'CDS': stop-start, "3'UTR": cum_exon_len[-1]-stop,
                        'start_codon': seq_start, 'stop_codon': seq_end, 'NMD': dist_pas > 55, 'uORFs': uORFs, 'ref_ids': ref_transcript_ids}
            if kozak_matrix is not None:
                if kozak is None:
                    orf_dict['kozak'] = kozak_score(tr_seq, start, kozak_matrix)
                else:
                    orf_dict['kozak'] = kozak

            if coding_hexamers is not None and noncoding_hexamers is not None:
                orf_dict['hexamer'] = FrameKmer.kmer_ratio(tr_seq[start:stop], 6, 3, coding_hexamers, noncoding_hexamers)
            if get_fickett:
                orf_dict['fickett'] = fickett.fickett_value(tr_seq[start:stop])
            transcript['ORF'] = (*genome_pos, orf_dict)

    def add_fragments(self):
        '''Checks for transcripts that are fully contained in other transcripts.

        Transcripts that are fully contained in other transcripts are potential truncations.
        This function populates the 'fragment' field of the transcript dictionaries with the indices of the containing transcripts,
        and the exon ids that match the first and last exons.'''

        for transcript_id, containers in self.segment_graph.find_fragments().items():
            self.transcripts[transcript_id]['fragments'] = containers  # list of (containing transcript id, first 5' exons, first 3'exons)

    def coding_len(self, transcript_id):
        '''Returns length of 5\'UTR, coding sequence and 3\'UTR.

        :param transcript_id: The transcript index for which the coding length is requested. '''

        try:
            exons = self.transcripts[transcript_id]['exons']
            cds = self.transcripts[transcript_id]['CDS']
        except KeyError:
            return None
        else:
            coding_len = _coding_len(exons, cds)
        if self.strand == '-':
            coding_len.reverse()
        return coding_len

    def get_infos(self, transcript_id, keys, sample_i, group_i, **kwargs):
        '''Returns the transcript information specified in "keys" as a list.'''
        return [value for k in keys for value in self._get_info(transcript_id, k, sample_i, group_i)]

    def _get_info(self, transcript_id, key, sample_i, group_i, **kwargs):
        # returns tuples (as some keys return multiple values)
        if key == 'length':
            return sum((e - b for b, e in self.transcripts[transcript_id]['exons'])),
        elif key == 'n_exons':
            return len(self.transcripts[transcript_id]['exons']),
        elif key == 'exon_starts':
            return ','.join(str(e[0]) for e in self.transcripts[transcript_id]['exons']),
        elif key == 'exon_ends':
            return ','.join(str(e[1]) for e in self.transcripts[transcript_id]['exons']),
        elif key == 'annotation':
            # sel=['sj_i','base_i', 'as']
            if 'annotation' not in self.transcripts[transcript_id]:
                return ('NA',) * 2
            nov_class, subcat = self.transcripts[transcript_id]['annotation']
            # subcat_string = ';'.join(k if v is None else '{}:{}'.format(k, v) for k, v in subcat.items())
            return SPLICE_CATEGORY[nov_class], ','.join(subcat)  # only the names of the subcategories
        elif key == 'coverage':
            return self.coverage[sample_i, transcript_id]
        elif key == 'tpm':
            return self.tpm(kwargs.get('pseudocount', 1))[sample_i, transcript_id]
        elif key == 'group_coverage_sum':
            return tuple(self.coverage[si, transcript_id].sum() for si in group_i)
        elif key == 'group_tpm_mean':
            return tuple(self.tpm(kwargs.get('pseudocount', 1))[si, transcript_id].mean() for si in group_i)
        elif key in self.transcripts[transcript_id]:
            val = self.transcripts[transcript_id][key]
            if isinstance(val, Iterable):  # iterables get converted to string
                return str(val),
            else:
                return val,  # atomic (e.g. numeric)
        return 'NA',

    def _set_coverage(self, force=False):
        samples = self._transcriptome.samples
        cov = np.zeros((len(samples), self.n_transcripts), dtype=int)
        if not force:  # keep the segment graph if no new transcripts
            known = self.data.get('coverage', None)
            if known is not None and known.shape[1] == self.n_transcripts:
                if known.shape == cov.shape:
                    return
                cov[:known.shape[0], :] = known
                for i in range(known.shape[0], len(samples)):
                    for j, transcript in enumerate(self.transcripts):
                        cov[i, j] = transcript['coverage'].get(samples[i], 0)
                self.data['coverage'] = cov
                return
        for i, sample in enumerate(samples):
            for j, transcript in enumerate(self.transcripts):
                cov[i, j] = transcript['coverage'].get(sample, 0)
        self.data['coverage'] = cov
        self.data['segment_graph'] = None

    def tpm(self, pseudocount=1):
        '''Returns the transcripts per million (TPM).

        TPM is returned as a numpy array, with samples in columns and transcript isoforms in the rows.'''
        return (self.coverage+pseudocount)/self._transcriptome.sample_table['nonchimeric_reads'].values.reshape(-1, 1)*1e6

    def find_transcript_positions(self, transcript_id, pos, reference=False):
        '''Converts genomic positions to positions within the transcript.

        :param transcript_id: The transcript id
        :param pos: List of sorted genomic positions, for which the transcript positions are computed.'''

        tr_pos = []
        exons = self.ref_transcripts[transcript_id]['exons'] if reference else self.transcripts[transcript_id]['exons']
        e_idx = 0
        offset = 0
        for p in sorted(pos):
            try:
                while p > exons[e_idx][1]:
                    offset += (exons[e_idx][1]-exons[e_idx][0])
                    e_idx += 1

            except IndexError:
                for _ in range(len(pos)-len(tr_pos)):
                    tr_pos.append(None)
                break
            tr_pos.append(offset+p-exons[e_idx][0] if p >= exons[e_idx][0] else None)
        if self.strand == '-':
            trlen = sum(end-start for start, end in exons)
            tr_pos = [None if p is None else trlen-p for p in tr_pos]
        return tr_pos

    @property
    def coverage(self):
        '''Returns the transcript coverage.

        Coverage is returned as a numpy array, with samples in columns and transcript isoforms in the rows.'''
        cov = self.data.get('coverage', None)
        if cov is not None:
            return cov
        self._set_coverage()
        return self.data['coverage']

    @property
    def gene_coverage(self):
        '''Returns the gene coverage.

        Total Coverage of the gene for each sample.'''

        return self.coverage.sum(1)

    @property
    def chrom(self):
        '''Returns the genes chromosome.'''
        return self.data['chr']

    @property
    def start(self):  # alias for begin
        return self.begin

    @property
    def region(self):
        '''Returns the region of the gene as a string in the format "chr:start-end".'''
        try:
            return '{}:{}-{}'.format(self.chrom, self.start, self.end)
        except KeyError:
            raise

    @property
    def id(self):
        '''Returns the gene id'''
        try:
            return self.data['ID']
        except KeyError:
            logger.error(self.data)
            raise

    @property
    def name(self):
        '''Returns the gene name'''
        try:
            return self.data['name']
        except KeyError:
            return self.id  # e.g. novel genes do not have a name (but id)

    @property
    def is_annotated(self):
        '''Returns "True" iff reference annotation is present for the gene.'''
        return 'reference' in self.data

    @property
    def is_expressed(self):
        '''Returns "True" iff gene is covered by at least one long read in at least one sample.'''
        return bool(self.transcripts)

    @property
    def strand(self):
        '''Returns the strand of the gene, e.g. "+" or "-"'''
        return self.data['strand']

    @property
    def transcripts(self) -> list[Transcript]:
        '''Returns the list of transcripts of the gene, as found by LRTS.'''
        try:
            return self.data['transcripts']
        except KeyError:
            return []

    @property
    def ref_transcripts(self) -> list[Transcript]:
        '''Returns the list of reference transcripts of the gene.'''
        try:
            return self.data['reference']['transcripts']
        except KeyError:
            return []

    @property
    def n_transcripts(self):
        '''Returns number of transcripts of the gene, as found by LRTS.'''
        return len(self.transcripts)

    @property
    def n_ref_transcripts(self):
        '''Returns number of reference transcripts of the gene.'''
        return len(self.ref_transcripts)

    @property
    def ref_segment_graph(self):  # raises key error if not self.is_annotated
        '''Returns the segment graph of the reference transcripts for the gene'''

        assert self.is_annotated, "reference segment graph requested on novel gene"
        if 'segment_graph' not in self.data['reference'] or self.data['reference']['segment_graph'] is None:
            transcript_exons = [transcript['exons'] for transcript in self.ref_transcripts]
            self.data['reference']['segment_graph'] = SegmentGraph(transcript_exons, self.strand)
        return self.data['reference']['segment_graph']

    @property
    def segment_graph(self):
        '''Returns the segment graph of the LRTS transcripts for the gene'''
        if 'segment_graph' not in self.data or self.data['segment_graph'] is None:
            transcript_exons = [transcript['exons'] for transcript in self.transcripts]
            try:
                self.data['segment_graph'] = SegmentGraph(transcript_exons, self.strand)
            except Exception:
                logger.error('Error initializing Segment Graph on %s with exons %s', self.strand, transcript_exons)
                raise
        return self.data['segment_graph']

    def segment_graph_filtered(self, query=None, min_coverage=None, max_coverage=None):
        '''Returns a filtered segment graph of the LRTS transcripts for the gene'''
        transcript_ids = self.filter_transcripts(query, min_coverage, max_coverage)
        transcript_exons = [transcript['exons'] for i, transcript in enumerate(self.transcripts) if i in transcript_ids]
        return SegmentGraph(transcript_exons, self.strand)

    def __copy__(self):
        return Gene(self.start, self.end, self.data, self._transcriptome)

    def __deepcopy__(self, memo):  # does not copy _transcriptome!
        return Gene(self.start, self.end, copy.deepcopy(self.data, memo), self._transcriptome)

    def __reduce__(self):
        return Gene, (self.start, self.end, self.data, self._transcriptome)

    def copy(self):
        'Returns a shallow copy of self.'
        return self.__copy__()

    def filter_transcripts(self, query=None, min_coverage=None, max_coverage=None):
        if query:
            transcript_filter = self._transcriptome.filter['transcript']
            # used_tags={tag for tag in re.findall(r'\b\w+\b', query) if tag not in BOOL_OP}
            query_fun, used_tags = _filter_function(query)
            msg = 'did not find the following filter rules: {}\nvalid rules are: {}'
            assert all(f in transcript_filter for f in used_tags), msg.format(
                ', '.join(f for f in used_tags if f not in transcript_filter), ', '.join(transcript_filter))
            transcript_filter_fun = {tag: _filter_function(tag, transcript_filter)[0] for tag in used_tags if tag in transcript_filter}
        transcript_ids = []
        for i, transcript in enumerate(self.transcripts):
            if min_coverage and self.coverage[:, i].sum() < min_coverage:
                continue
            if max_coverage and self.coverage[:, i].sum() > max_coverage:
                continue
            if query is None or query_fun(
                    **{tag: f(gene=self, transcript_id=i, **transcript) for tag, f in transcript_filter_fun.items()}):
                transcript_ids.append(i)
        return transcript_ids

    def filter_ref_transcripts(self, query=None):
        if query:
            transcript_filter = self._transcriptome.filter['reference']
            # used_tags={tag for tag in re.findall(r'\b\w+\b', query) if tag not in BOOL_OP}
            query_fun, used_tags = _filter_function(query)
            msg = 'did not find the following filter rules: {}\nvalid rules are: {}'
            assert all(f in transcript_filter for f in used_tags), msg.format(
                ', '.join(f for f in used_tags if f not in transcript_filter), ', '.join(transcript_filter))
            transcript_filter_func = {tag: _filter_function(transcript_filter[tag])[0] for tag in used_tags if tag in transcript_filter}
        else:
            return list(range(len(self.ref_transcripts)))
        transcript_ids = []
        for i, transcript in enumerate(self.ref_transcripts):
            filter_transcript = transcript.copy()
            if 'transcript_id' in filter_transcript:
                filter_transcript['ref_transcript_id'] = filter_transcript.pop('transcript_id')
            if query_fun(**{tag: f(gene=self, transcript_id=i, **filter_transcript) for tag, f in transcript_filter_func.items()}):
                transcript_ids.append(i)
        return transcript_ids

    def _find_splice_sites(exons, transcripts):
        '''Checks whether the splice sites of a new transcript are present in the set of transcripts.
        avoids the computation of segment graph, which provides the same functionality.

        :param exons: A list of exon tuples representing the transcript
        :type exons: list
        :return: boolean array indicating whether the splice site is contained or not'''

        intron_iter = [pairwise(transcript['exons']) for transcript in transcripts]
        current = [next(transcript) for transcript in intron_iter]
        contained = np.zeros(len(exons)-1)
        for j, (exon1, exon2) in enumerate(pairwise(exons)):
            for i, transcript in enumerate(intron_iter):
                while current[i][0][1] < exon1[1]:
                    try:
                        current[i] = next(transcript)
                    except StopIteration:
                        continue
                if exon1[1] == current[i][0][1] and exon2[0] == current[i][1][0]:
                    contained[j] = True
        return current

    def coordination_test(self, samples=None, test: Literal['fisher', 'chi2'] = "fisher", min_dist_AB=1, min_dist_events=1, min_total=100, min_alt_fraction=.1,
                          events: Optional[list[ASEvent]] = None, event_type=("ES", "5AS", "3AS", "IR", "ME"),
                          transcript_filter: Optional[str] = None) -> list[tuple]:
        '''Performs pairwise independence test for all pairs of Alternative Splicing Events (ASEs) in a gene.

        For all pairs of ASEs in a gene creates a contingency table and performs an independence test.
        All ASEs A have two states, pri_A and alt_A, the primary and the alternative state respectivley.
        Thus, given two events A and B, we have four possible ways in which these events can occur on
        a transcript, that is, pri_A and pri_B, pri_A and alt_B, alt_A and pri_B, and alt_A and alt_B.
        These four values can be put in a contingency table and independence, or coordination,
        between the two events can be tested.

        :param samples: Specify the samples that should be considered in the test.
            The samples can be provided either as a single group name, a list of sample names, or a list of sample indices.
        :param test: Test to be performed. One of ("chi2", "fisher")
        :type test: str
        :param min_dist_AB: Minimum distance (in nucleotides) between node A and B in an event
        :param min_dist_events: Minimum number of nucleotides between the end of the first event and the start of the second event in each tested pair of events
        :param min_total: The minimum total number of reads for an event pair to pass the filter.
        :type min_total: int
        :param min_alt_fraction: The minimum fraction of reads supporting the minor alternative of the two events.
        :type min_alt_fraction: float
        :param events: To speed up testing on different groups of the same transcriptome objects, events can be precomputed
            with the isotools._utils.precompute_events_dict function.
        :param event_type:  A tuple with event types to test. Valid types are
            ("ES", "3AS", "5AS", "IR", "ME", "TSS", "PAS"). Default is ("ES", "5AS", "3AS", "IR", "ME").
            Not used if the event parameter is already given.
        :return: A list of tuples with the test results: (gene_id, gene_name, strand, eventA_type, eventB_type,
            eventA_start, eventA_end, eventB_start, eventB_end, p_value, test_stat, log2OR, dcPSI_AB, dcPSI_BA,
            priA_priB, priA_altB, altA_priB, altA_altB, priA_priB_transcript_ids, priA_altB_transcript_ids, altA_priB_transcript_ids, altA_altB_transcript_ids).
        '''

        if samples is None:
            cov = self.coverage.sum(axis=0)
        else:
            try:
                # Fast mode when testing several genes
                cov = self.coverage[samples].sum(0)
            except IndexError:
                # Fall back to looking up the sample indices
                from isotools._transcriptome_stats import _check_groups
                _, _, groups = _check_groups(self._transcriptome, [samples], 1)
                cov = self.coverage[groups[0]].sum(0)

        if transcript_filter:
            sg = self.segment_graph_filtered(query=transcript_filter)
        else:
            sg = self.segment_graph

        if events is None:
            events = sg.find_splice_bubbles(types=event_type)

        events = [event for event in events
                  if _filter_event(cov, event, segment_graph=sg, min_total=min_total,
                                   min_alt_fraction=min_alt_fraction, min_dist_AB=min_dist_AB)]
        # make sure its sorted (according to gene strand)
        if self.strand == '+':
            events.sort(key=itemgetter(2, 3), reverse=False)  # sort by starting node
        else:
            events.sort(key=itemgetter(3, 2), reverse=True)  # reverse sort by end node
        test_res = []

        for event1, event2 in itertools.combinations(events, 2):
            if sg.events_dist(event1, event2) < min_dist_events:
                continue
            if (event1[4], event2[4]) == ("TSS", "TSS") or (event1[4], event2[4]) == ("PAS", "PAS"):
                continue

            con_tab, tr_ID_tab = prepare_contingency_table(event1, event2, cov)

            if con_tab.sum(None) < min_total:  # check that the joint occurrence of the two events passes the threshold
                continue
            if min(con_tab.sum(1).min(), con_tab.sum(0).min())/con_tab.sum(None) < min_alt_fraction:
                continue
            test_result = pairwise_event_test(con_tab, test=test)  # append to test result
            coordinate1 = sg._get_event_coordinate(event1)
            coordinate2 = sg._get_event_coordinate(event2)

            attr = (self.id, self.name, self.strand, event1[4], event2[4]) + \
                coordinate1 + coordinate2 + test_result + \
                tuple(con_tab.flatten()) + tuple(tr_ID_tab.flatten())

            # event1[4] is the event1 type
            # coordinate1[0] is the starting coordinate of event 1
            # coordinate1[0] is the ending coordinate of event 1
            # coordinate2[0] is the starting coordinate of event 2
            # coordinate2[1] is the ending coordinate of event 2

            test_res.append(attr)

        return test_res

    def die_test(self, groups, min_cov=25, n_isoforms=10):
        ''' Reimplementation of the DIE test, suggested by Joglekar et al in Nat Commun 12, 463 (2021):
        "A spatially resolved brain region- and cell type-specific isoform atlas of the postnatal mouse brain"

        Syntax and parameters follow the original implementation in
        https://github.com/noush-joglekar/scisorseqr/blob/master/inst/RScript/IsoformTest.R
        :param groups: Define the columns for the groups.
        :param min_cov: Minimal number of reads per group for the gene.
        :param n_isoforms: Number of isoforms to consider in the test for the gene. All additional least expressed isoforms get summarized.'''
        # select the samples and sum the group counts
        try:
            # Fast mode when testing several genes
            cov = np.array([self.coverage[grp].sum(0) for grp in groups]).T
        except IndexError:
            # Fall back to looking up the sample indices
            from isotools._transcriptome_stats import _check_groups
            _, _, groups = _check_groups(self._transcriptome, groups)
            cov = np.array([self.coverage[grp].sum(0) for grp in groups]).T

        if np.any(cov.sum(0) < min_cov):
            return np.nan, np.nan, []
        # if there are more than 'numIsoforms' isoforms of the gene, all additional least expressed get summarized.
        if cov.shape[0] > n_isoforms:
            idx = np.argpartition(-cov.sum(1), n_isoforms)  # take the n_isoforms most expressed isoforms (random order)
            additional = cov[idx[n_isoforms:]].sum(0)
            cov = cov[idx[:n_isoforms]]
            cov[n_isoforms-1] += additional
            idx[n_isoforms-1] = -1  # this isoform gets all other - I give it index
        elif cov.shape[0] < 2:
            return np.nan, np.nan, []
        else:
            idx = np.array(range(cov.shape[0]))
        try:
            _, pval, _, _ = chi2_contingency(cov)
        except ValueError:
            logger.error(f'chi2_contingency({cov})')
            raise
        iso_frac = cov/cov.sum(0)
        deltaPI = iso_frac[..., 0]-iso_frac[..., 1]
        order = np.argsort(deltaPI)
        pos_idx = [order[-i] for i in range(1, 3) if deltaPI[order[-i]] > 0]
        neg_idx = [order[i] for i in range(2) if deltaPI[order[i]] < 0]
        deltaPI_pos = deltaPI[pos_idx].sum()
        deltaPI_neg = deltaPI[neg_idx].sum()
        if deltaPI_pos > -deltaPI_neg:
            return pval, deltaPI_pos, idx[pos_idx]
        else:
            return pval, deltaPI_neg, idx[neg_idx]

    def _unify_ends(self, smooth_window=31, rel_prominence=1, search_range: tuple[float, float] = (.1, .9), correct_tss=True):
        ''' Find common TSS/PAS for transcripts of the gene'''
        if not self.transcripts:
            # nothing to do here
            return
        assert 0 <= search_range[0] <= .5 <= search_range[1] <= 1
        # get gene tss/pas profiles
        tss: dict[int, int] = {}
        pas: dict[int, int] = {}
        strand = 1 if self.strand == '+' else -1
        for transcript in self.transcripts:
            for sample in transcript['TSS']:
                for pos, c in transcript['TSS'][sample].items():
                    tss[pos] = tss.get(pos, 0)+c
            for sample in transcript['PAS']:
                for pos, c in transcript['PAS'][sample].items():
                    pas[pos] = pas.get(pos, 0)+c

        tss_pos = [min(tss), max(tss)]
        if tss_pos[1]-tss_pos[0] < smooth_window:
            tss_pos[0] = tss_pos[1] - smooth_window + 1
        pas_pos = [min(pas), max(pas)]
        if pas_pos[1]-pas_pos[0] < smooth_window:
            pas_pos[0] = pas_pos[1] - smooth_window + 1
        tss = [tss.get(pos, 0) for pos in range(tss_pos[0], tss_pos[1]+1)]
        pas = [pas.get(pos, 0) for pos in range(pas_pos[0], pas_pos[1]+1)]
        # smooth profiles and find maxima
        tss_smooth = smooth(np.array(tss), smooth_window)
        pas_smooth = smooth(np.array(pas), smooth_window)
        # at least half of smooth_window reads required to call a peak
        # minimal distance between peaks is > ~ smooth_window
        # rel_prominence=1 -> smaller peak must have twice the height of valley to call two peaks
        tss_peaks, _ = find_peaks(np.log2(tss_smooth+1), prominence=(rel_prominence, None))
        tss_peak_pos: list[int] = tss_peaks+tss_pos[0]-1
        pas_peaks, _ = find_peaks(np.log2(pas_smooth+1), prominence=(rel_prominence, None))
        pas_peak_pos: list[int] = pas_peaks+pas_pos[0]-1

        # find transcripts with common first/last splice site
        first_junction: dict[int, list[int]] = {}
        last_junction: dict[int, list[int]] = {}
        for transcript_id, transcript in enumerate(self.transcripts):
            first_junction.setdefault(transcript['exons'][0][1], []).append(transcript_id)
            last_junction.setdefault(transcript['exons'][-1][0], []).append(transcript_id)
        # first / last junction with respect to direction of transcription
        if self.strand == '-':
            first_junction, last_junction = last_junction, first_junction
        # for each site, find consistent "peaks" TSS/PAS
        # if none found use median of all read starts
        for junction_pos, transcript_ids in first_junction.items():
            profile = {}
            for transcript_id in transcript_ids:
                for sample_tss in self.transcripts[transcript_id]['TSS'].values():
                    for pos, c in sample_tss.items():
                        profile[pos] = profile.get(pos, 0) + c
            quantiles = get_quantiles(sorted(profile.items()), [search_range[0], .5, search_range[1]])
            # one/ several peaks within base range? -> quantify by next read_start
            # else use median
            ol_peaks = [p for p in tss_peak_pos if quantiles[0] < p <= quantiles[-1]]
            if not ol_peaks:
                ol_peaks = [quantiles[1]]
            for transcript_id in transcript_ids:
                transcript = self.transcripts[transcript_id]
                transcript['TSS_unified'] = {}
                for sample, sample_tss in transcript['TSS'].items():
                    tss_unified: dict[int, int] = {}
                    # for each read start position, find closest peak
                    for pos, c in sample_tss.items():
                        next_peak = min((p for p in ol_peaks if cmp_dist(junction_pos, p, min_dist=3) == strand),
                                        default=pos, key=lambda x: abs(x-pos))
                        tss_unified[next_peak] = tss_unified.get(next_peak, 0)+c
                    transcript['TSS_unified'][sample] = tss_unified
        # same for PAS
        for junction_pos, transcript_ids in last_junction.items():
            profile = {}
            for transcript_id in transcript_ids:
                for sa_pas in self.transcripts[transcript_id]['PAS'].values():
                    for pos, c in sa_pas.items():
                        profile[pos] = profile.get(pos, 0)+c
            quantiles = get_quantiles(sorted(profile.items()), [search_range[0], .5, search_range[1]])
            # one/ several peaks within base range? -> quantify by next read_start
            # else use median
            ol_peaks = [p for p in pas_peak_pos if quantiles[0] < p <= quantiles[-1]]
            if not ol_peaks:
                ol_peaks = [quantiles[1]]
            for transcript_id in transcript_ids:
                transcript = self.transcripts[transcript_id]
                transcript['PAS_unified'] = {}
                for sample, sa_pas in transcript['PAS'].items():
                    pas_unified: dict[int, int] = {}
                    for pos, c in sa_pas.items():
                        next_peak = min((p for p in ol_peaks if cmp_dist(p, junction_pos, min_dist=3) == strand),
                                        default=pos, key=lambda x: abs(x-pos))
                        pas_unified[next_peak] = pas_unified.get(next_peak, 0)+c
                    transcript['PAS_unified'][sample] = pas_unified
        for transcript in self.transcripts:
            # find the most common tss/pas per transcript, and set the exon boundaries
            sum_tss: dict[int, int] = {}
            sum_pas: dict[int, int] = {}
            start = end = max_tss = max_pas = 0
            for sample_tss in transcript['TSS_unified'].values():
                for pos, cov in sample_tss.items():
                    sum_tss[pos] = sum_tss.get(pos, 0)+cov
            for pos, cov in sum_tss.items():
                if cov > max_tss:
                    max_tss = cov
                    start = pos
            for sa_pas in transcript['PAS_unified'].values():
                for pos, cov in sa_pas.items():
                    sum_pas[pos] = sum_pas.get(pos, 0) + cov
            for pos, cov in sum_pas.items():
                if cov > max_pas:
                    max_pas = cov
                    end = pos
            if self.strand == '-':
                start, end = end, start
            if start >= end:  # for monoexons this may happen in rare situations
                assert len(transcript['exons']) == 1
                transcript['TSS_unified'] = None
                transcript['PAS_unified'] = None
            else:
                try:
                    # issues if the new exon start is behind the exon end
                    assert start < transcript['exons'][0][1] or len(transcript['exons']) == 1, 'error unifying %s: %s>=%s' % (transcript["exons"], start, transcript['exons'][0][1])
                    transcript['exons'][0][0] = start
                    assert end > transcript['exons'][-1][0] or len(transcript['exons']) == 1, 'error unifying %s: %s<=%s' % (transcript["exons"], end, transcript['exons'][-1][0])
                    transcript['exons'][-1][1] = end
                except AssertionError:
                    logger.error('%s TSS= %s, PAS=%s -> TSS_unified= %s, PAS_unified=%s', self, transcript['TSS'], transcript['PAS'],  transcript['TSS_unified'], transcript['PAS_unified'])
                    raise
            if correct_tss:
                self._TSS_correction(transcript)


    def _TSS_correction(self, transcript: Transcript):
        '''
        Correct TSS to the closest upstream reference TSS from best peak.
        Don't extend past exon ends, because that can introduce artificial intron retention events.
        '''
        if self.is_annotated:
            if transcript['strand'] == '+':
                tss = transcript['exons'][0][0]
                ref_tsss = [ref_transcript['exons'][0][0] for ref_transcript in self.ref_transcripts if ref_transcript['exons'][0][0] <= tss]
                if ref_tsss:
                    new_tss = max(ref_tsss)
                    # Find ref upstream exon ends that are between the old and the new TSS
                    ref_exon_ends = [exon[1] for transcript in self.ref_transcripts for exon in transcript["exons"] if exon[1] <= tss and exon[1] >= new_tss]
                    # Don't extend past exon ends
                    if not ref_exon_ends:
                        logger.debug(f'Corrected TSS ({transcript['strand']} strand) from {tss} to {new_tss}')
                        transcript['exons'][0][0] = new_tss
            else:
                tss = transcript['exons'][-1][1]
                ref_tsss = [ref_transcript['exons'][-1][1] for ref_transcript in self.ref_transcripts if ref_transcript['exons'][-1][1] >= tss]
                if ref_tsss:
                    new_tss = min(ref_tsss)
                    ref_exon_ends = [exon[0] for transcript in self.ref_transcripts for exon in transcript["exons"] if exon[0] >= tss and exon[0] <= new_tss]
                    if not ref_exon_ends:
                        logger.debug(f'Corrected TSS ({transcript['strand']} strand) from {tss} to {new_tss}')
                        transcript['exons'][-1][1] = new_tss


def _coding_len(exons: list[tuple[int, int]], cds):
    coding_len = [0, 0, 0]
    state = 0
    for exon in exons:
        if state < 2 and exon[1] >= cds[state]:
            coding_len[state] += cds[state] - exon[0]
            if state == 0 and cds[1] <= exon[1]:  # special case: CDS start and end in same exon
                coding_len[1] = cds[1] - cds[0]
                coding_len[2] = exon[1] - cds[1]
                state += 2
            else:
                coding_len[state + 1] = exon[1] - cds[state]
                state += 1
        else:
            coding_len[state] += exon[1] - exon[0]
    return coding_len


def repeat_len(seq1, seq2, wobble, max_mm):
    ''' Calcluate direct repeat length between seq1 and seq2
    '''
    score = [0]*(2*wobble+1)
    delta = int(len(seq1)/2-wobble)
    for w in range(2*wobble+1):  # wobble
        s1 = seq1[w:len(seq1)-(2*wobble-w)]
        s2 = seq2[wobble:len(seq2)-wobble]
        align = [a == b for a, b in zip(s1, s2)]
        score_left = find_runlength(reversed(align[:delta]), max_mm)
        score_right = find_runlength(align[delta:], max_mm)
        score[w] = max([score_left[fmm]+score_right[max_mm-fmm] for fmm in range(max_mm+1)])
    return max(score)


def find_runlength(align, max_mm):
    '''Find the runlength, e.g. the number of True in the list before the max_mm+1 False occur.
    '''
    score = [0]*(max_mm+1)
    mm = 0
    for a in align:
        if not a:
            mm += 1
            if mm > max_mm:
                return score
            score[mm] = score[mm-1]
        else:
            score[mm] += 1
    for i in range(mm+1, max_mm+1):
        score[i] = score[i-1]
    return score
