from __future__ import annotations
import numpy as np
import logging
import itertools
from sortedcontainers import SortedDict  # for SpliceGraph
from ._utils import pairwise, has_overlap, _interval_dist, ASEType, ASEvent
from .decorators import deprecated, experimental
from typing import Generator, Literal, Optional, Union

logger = logging.getLogger("isotools")


class SegmentGraph:
    '''Segment Graph Implementation

    Nodes in the Segment Graph represent disjoint exonic bins (aka segments) and have start (genomic 5'), end (genomic 3'),
    and a dict of successors and predesessors (edges)
    Edges link two exonic bins x and y that follow successively in a transcript.
    They represent either introns (if x.end<y.start) or connect exonic bins of the same exon (x.end==y.start).

    :param transcripts: A list of transcripts, which are lists of exons, which in turn are (start,end) tuples
    :type transcripts: list
    :param strand: the strand of the gene, either "+" or "-"'''

    strand: Literal["+", "-"]
    _graph: list[SegGraphNode]
    _tss: list[int]
    "List of start-nodes for each transcript. TODO: Name is misleading. If the strand is '-', this is actually the PAS."
    _pas: list[int]
    "List of end-nodes for each transcript. TODO: Name is misleading. If the strand is '-', this is actually the TSS."

    def __init__(
        self, transcript_exons: list[list[tuple[int, int]]], strand: Literal["+", "-"]
    ):
        self.strand = strand
        assert strand in "+-", 'strand must be either "+" or "-"'
        open_exons: dict[int, int] = dict()
        for exons in transcript_exons:
            for exon in exons:
                open_exons[exon[0]] = open_exons.get(exon[0], 0) + 1
                open_exons[exon[1]] = open_exons.get(exon[1], 0) - 1
        # sort by value
        boundaries = sorted(open_exons)
        open_count = 0
        self._graph = list()
        for i, start in enumerate(boundaries[:-1]):
            open_count += open_exons[start]
            if open_count > 0:
                self._graph.append(SegGraphNode(start, boundaries[i + 1]))

        # get the links
        start_idx = {node.start: i for i, node in enumerate(self._graph)}
        end_idx = {node.end: i for i, node in enumerate(self._graph)}
        self._tss = [start_idx[exons[0][0]] for exons in transcript_exons]
        self._pas = [end_idx[exons[-1][1]] for exons in transcript_exons]

        for i, exons in enumerate(transcript_exons):
            for j, exon in enumerate(exons):
                # pseudojunctions within exon
                if start_idx[exon[0]] < end_idx[exon[1]]:
                    self._graph[start_idx[exon[0]]].suc[i] = start_idx[exon[0]] + 1
                    self._graph[end_idx[exon[1]]].pre[i] = end_idx[exon[1]] - 1
                    for node_idx in range(start_idx[exon[0]] + 1, end_idx[exon[1]]):
                        self._graph[node_idx].suc[i] = node_idx + 1
                        self._graph[node_idx].pre[i] = node_idx - 1
                # real junctions
                if j < len(exons) - 1:
                    exon2 = exons[j + 1]
                    self._graph[end_idx[exon[1]]].suc[i] = start_idx[exon2[0]]
                    self._graph[start_idx[exon2[0]]].pre[i] = end_idx[exon[1]]

    def _restore(self, i: int) -> list:  # mainly for testing
        """Restore the i_{th} transcript from the Segment graph by traversing from 5' to 3'

        :param i: The index of the transcript to restore
        :type i: int
        :return: A list of exon tuples representing the transcript
        :rtype: list"""
        idx = self._tss[i]
        exons = [[self._graph[idx].start, self._graph[idx].end]]
        while True:
            if idx == self._pas[i]:
                break
            idx = self._graph[idx].suc[i]
            if self._graph[idx].start == exons[-1][1]:  # extend
                exons[-1][1] = self._graph[idx].end
            else:
                exons.append([self._graph[idx].start, self._graph[idx].end])
            # print(exons)

        return exons

    def _restore_reverse(self, i: int) -> list:  # mainly for testing
        """Restore the ith transcript from the Segment graph by traversing from 3' to 5'

        :param i: The index of the transcript to restore
        :type i: int
        :return: A list of exon tuples representing the transcript
        :rtype: list"""
        idx = self._pas[i]
        exons = [[self._graph[idx].start, self._graph[idx].end]]
        while True:
            if idx == self._tss[i]:
                break
            idx = self._graph[idx].pre[i]
            if self._graph[idx].end == exons[-1][0]:  # extend
                exons[-1][0] = self._graph[idx].start
            else:
                exons.append([self._graph[idx].start, self._graph[idx].end])
            # print(exons)

        exons.reverse()
        return exons

    @deprecated
    def search_transcript2(self, exons: list[tuple[int, int]]):
        """Tests if a transcript (provided as list of exons) is contained in self and return the corresponding transcript indices.

        :param exons: A list of exon tuples representing the transcript
        :type exons: list
        :return: a list of supporting transcript indices
        :rtype: list"""

        # fst special case: exons extends segment graph
        if exons[0][1] <= self[0].start or exons[-1][0] >= self[-1].end:
            return []
        # snd special case: single exon transcript: return all overlapping single exon transcripts form sg
        if len(exons) == 1:
            return [
                transcript_id
                for transcript_id, (j1, j2) in enumerate(zip(self._tss, self._pas))
                if self._is_same_exon(transcript_id, j1, j2)
                and self[j1].start <= exons[0][1]
                and self[j2].end >= exons[0][0]
            ]
        # all junctions must be contained and no additional
        transcript = set(range(len(self._tss)))
        j = 0
        for i, e in enumerate(exons[:-1]):
            while (
                j < len(self) and self[j].end < e[1]
            ):  # check exon (no junction allowed)
                transcript -= set(
                    transcript_id
                    for transcript_id, j2 in self[j].suc.items()
                    if self[j].end != self[j2].start
                )
                j += 1
            if self[j].end != e[1]:
                return []
            # check junction (must be present)
            transcript &= set(
                transcript_id
                for transcript_id, j2 in self[j].suc.items()
                if self[j2].start == exons[i + 1][0]
            )
            j += 1
            if len(transcript) == 0:
                return transcript

        while j < len(self):  # check last exon (no junction allowed)
            transcript -= set(
                transcript_id
                for transcript_id, j2 in self[j].suc.items()
                if self[j].end != self[j2].start
            )
            j += 1
        return [transcript_id for transcript_id in transcript]

    def search_transcript(
        self, exons: list[tuple[int, int]], complete=True, include_ends=False
    ):
        """Tests if a transcript (provided as list of exons) is contained in sg and return the corresponding transcript indices.

        Search the splice graph for transcripts that match the introns of the provided list of exons.

        :param exons: A list of exon tuples representing the transcript
        :type exons: list
        :param complete: If True, yield only splice graph transcripts that match all introns from the exon list.
            If False, yield also splice graph transcripts having additional exons at the beginning and/or end.
        :param include_ends: If True, yield only splice graph transcripts that include the first and last exon.
            If False, also yield splice graph transcripts that extend first and/or last exon but match the intron chain.
        :return: a list of supporting transcript indices
        :rtype: list"""

        # fst special case: exons extends segment graph
        if include_ends:
            if exons[0][0] < self[0].start or exons[-1][1] > self[-1].end:
                return []
        else:
            if exons[0][1] <= self[0].start or exons[-1][0] >= self[-1].end:
                return []
        # snd special case: single exon transcript: return all overlapping /including single exon transcripts form sg
        if len(exons) == 1 and complete:
            if include_ends:
                return [
                    transcript_id
                    for transcript_id, (j1, j2) in enumerate(zip(self._tss, self._pas))
                    if self._is_same_exon(transcript_id, j1, j2)
                    and self[j1].start >= exons[0][0]
                    and self[j2].end <= exons[0][1]
                ]
            else:
                return [
                    transcript_id
                    for transcript_id, (j1, j2) in enumerate(zip(self._tss, self._pas))
                    if self._is_same_exon(transcript_id, j1, j2)
                    and self[j1].start <= exons[0][1]
                    and self[j2].end >= exons[0][0]
                ]

        # j is index of last overlapping node
        j = (
            next((i for i, n in enumerate(self) if n.start >= exons[0][1]), len(self))
            - 1
        )
        if self[j].end > exons[0][1] and len(exons) > 1:
            return []
        if complete:
            # for include_ends we need to find first node of exon
            # j_first is index of first node from the first exon,
            j_first = next(
                (i for i in range(j, 0, -1) if self[i - 1].end < self[i].start), 0
            )
            transcript = [
                transcript_id
                for transcript_id, i in enumerate(self._tss)
                if (not include_ends or self[i].start <= exons[0][0])
                and j_first <= i <= j
                and self._is_same_exon(transcript_id, i, j)
            ]
        else:
            if include_ends:
                j_first = next(
                    (i for i in range(j, 0, -1) if self[i].start <= exons[0][0]), 0
                )
                transcript = [
                    transcript_id
                    for transcript_id in self[j].pre
                    if self._is_same_exon(transcript_id, j_first, j)
                ]
                if j_first == j:
                    transcript += [
                        transcript_id
                        for transcript_id, i in enumerate(self._tss)
                        if i == j_first
                    ]
            else:
                transcript = list(self[j].pre) + [
                    transcript_id for transcript_id, i in enumerate(self._tss) if i == j
                ]
        # all junctions must be contained and no additional
        for i, e in enumerate(exons[:-1]):
            while (
                j < len(self) and self[j].end < e[1]
            ):  # check exon (no junction allowed)
                transcript = [
                    transcript_id
                    for transcript_id in transcript
                    if transcript_id in self[j].suc
                    and self[j].end == self[self[j].suc[transcript_id]].start
                ]
                j += 1
            if self[j].end != e[1]:
                return []
            # check junction (must be present)
            transcript = [
                transcript_id
                for transcript_id in transcript
                if transcript_id in self[j].suc
                and self[self[j].suc[transcript_id]].start == exons[i + 1][0]
            ]
            if not transcript:
                return []
            j = self[j].suc[transcript[0]]
        if include_ends:
            while self[j].end < exons[-1][1]:
                transcript = [
                    transcript_id
                    for transcript_id in transcript
                    if transcript_id in self[j].suc
                    and self[j].end == self[self[j].suc[transcript_id]].start
                ]
                j += 1
        if not complete:
            return transcript
        # ensure that all transcripts end (no junctions allowed)
        while j < len(self):  # check last exon (no junction allowed)
            transcript = [
                transcript_id
                for transcript_id in transcript
                if transcript_id not in self[j].suc
                or self[j].end == self[self[j].suc[transcript_id]].start
            ]
            j += 1
        return transcript

    def _is_same_exon(self, transcript_number, j1, j2):
        """Tests if nodes j1 and j2 belong to same exon in transcript transcript_number."""
        for j in range(j1, j2):
            if (
                transcript_number not in self[j].suc
                or self[j].suc[transcript_number] > j + 1
                or self[j].end != self[j + 1].start
            ):
                return False
        return True

    def _count_introns(self, transcript_number, j1, j2):
        """Counts the number of junctions between j1 and j2."""
        logger.debug(
            "counting introns of transcript %i between nodes %i and %i",
            transcript_number,
            j1,
            j2,
        )
        delta = 0
        if j1 == j2:
            return 0
        assert (
            transcript_number in self[j1].suc
        ), f"transcript {transcript_number} does not contain node {j1}"
        while j1 < j2:
            j_next = self[j1].suc[transcript_number]
            if j_next > j1 + 1 or self[j1].end != self[j1 + 1].start:
                delta += 1
            j1 = j_next
        return delta

    def get_node_matrix(self) -> np.array:
        """Gets the node matrix representation of the segment graph."""
        return np.array(
            [
                [tss == j or transcript_id in n.pre for j, n in enumerate(self)]
                for transcript_id, tss in enumerate(self._tss)
            ]
        )

    def find_fragments(self):
        """Finds all fragments (e.g. transcript contained in other transcripts) in the segment graph."""
        truncated: set[int] = set()
        contains: dict[int, set[int]] = {}
        nodes = self.get_node_matrix()
        for transcript_id, (tss, pas) in enumerate(zip(self._tss, self._pas)):
            if transcript_id in truncated:
                continue
            contains[transcript_id] = {
                transcript_id2
                for transcript_id2, (tss2, pas2) in enumerate(zip(self._tss, self._pas))
                if transcript_id2 != transcript_id
                and tss2 >= tss
                and pas2 <= pas
                and all(
                    nodes[transcript_id2, tss2 : pas2 + 1]
                    == nodes[transcript_id, tss2 : pas2 + 1]
                )
            }
            truncated.update(contains[transcript_id])  # those are not checked

        fragments = {}
        for big, smallL in contains.items():
            if big not in truncated:
                for transcript_id in smallL:
                    delta1 = self._count_introns(
                        big, self._tss[big], self._tss[transcript_id]
                    )
                    delta2 = self._count_introns(
                        big, self._pas[transcript_id], self._pas[big]
                    )
                    fragments.setdefault(transcript_id, []).append(
                        (big, delta1, delta2)
                        if self.strand == "+"
                        else (big, delta2, delta1)
                    )
        return fragments

    def get_alternative_splicing(self, exons: list[tuple[int, int]], alternative=None):
        """Compares exons to segment graph and returns list of novel splicing events.

        This function computes the novelty class of the provided transcript compared to (reference annotation) transcripts
        from the segment graph. It returns the "squanti category" (0=FSM,1=ISM,2=NIC,3=NNC,4=Novel gene) and the subcategory.

        :param exons: A list of exon tuples representing the transcript
        :type exons: list
        :param alternative: list of splice site indices that match other genes
        :return: pair with the squanti category number and the subcategories as list of novel splicing events
            that produce the provided transcript from the transcripts in splice graph
        :rtype: tuple"""

        # returns a tuple
        # the sqanti category: 0=FSM,1=ISM,2=NIC,3=NNC,4=Novel gene
        # subcategories: a list of novel splicing events or splice_identical

        # a list of tuples with (1) gene names and (2) junction numbers covered by other genes (e.g. readthrough fusion)
        if alternative is not None and len(alternative) > 0:
            category = 4
            fusion_exons = {int((i + 1) / 2) for j in alternative for i in j[1]}
            altsplice = {
                "readthrough fusion": alternative
            }  # other novel events are only found in the primary reference transcript
        else:
            transcript = self.search_transcript(exons)
            if transcript:
                return 0, {"FSM": transcript}
            category = 1
            altsplice = {}
            fusion_exons = set()

        is_reverse = self.strand == "-"
        j1 = next((j for j, n in enumerate(self) if n.end > exons[0][0]))
        # j1: index of first segment ending after exon start (i.e. first overlapping segment)
        j2 = next(
            (j - 1 for j in range(j1, len(self)) if self[j].start >= exons[0][1]),
            len(self) - 1,
        )
        # j2: index of last segment starting before exon end (i.e. last overlapping segment)

        # check truncation at begining (e.g. low position)
        if (
            len(exons) > 1  # no mono exon
            and not any(
                j in self._tss for j in range(j1, j2 + 1)
            )  # no tss/pas within exon
            and self[j1].start <= exons[0][0]
        ):  # start of first exon is exonic in ref
            j0 = max(
                self._tss[transcript_id] for transcript_id in self[j1].pre
            )  # j0 is the closest start node
            if any(
                self[j].end < self[j + 1].start for j in range(j0, j1)
            ):  # assure there is an intron between closest tss/pas and exon
                end = "5" if is_reverse else "3"
                altsplice.setdefault(f"{end}' fragment", []).append(
                    [self[j0].start, exons[0][0]]
                )  # at start (lower position)

        for i, ex1 in enumerate(exons):
            ex2 = None if i + 1 == len(exons) else exons[i + 1]
            if (
                i not in fusion_exons
            ):  # exon belongs to other gene (read through fusion)
                # finds intron retention (NIC), novel exons, novel splice sites, novel pas/tss (NNC)
                exon_altsplice, exon_cat = self._check_exon(
                    j1, j2, i == 0, is_reverse, ex1, ex2
                )
                category = max(exon_cat, category)
                for k, v in exon_altsplice.items():
                    altsplice.setdefault(k, []).extend(v)
                # find j2: index of last segment starting befor exon2 end (i.e. last overlapping  segment)
                if ex2 is not None:
                    if j2 + 1 < len(self):
                        j1, j2, junction_altsplice = self._check_junction(
                            j1, j2, ex1, ex2
                        )  # finds exon skipping and novel junction (NIC)
                        if junction_altsplice and i + 1 not in fusion_exons:
                            category = max(2, category)
                            for k, v in junction_altsplice.items():
                                altsplice.setdefault(k, []).extend(v)
                    else:
                        j1 = len(self)

        # check truncation at end (e.g. high position)
        if (
            len(exons) > 1
            and j2 >= j1
            and not any(
                j in self._pas for j in range(j1, j2 + 1)
            )  # no tss/pas within exon
            and self[j2].end >= exons[-1][1]
        ):  # end of last exon is exonic in ref
            try:
                j3 = min(
                    self._pas[transcript_id] for transcript_id in self[j2].suc
                )  # j3 is the next end node (pas/tss on fwd/rev)
            except ValueError:
                logger.error(
                    "\n".join(
                        [
                            str(exons),
                            str(self._pas),
                            str((j1, j2)),
                            str([(j, n) for j, n in enumerate(self)]),
                        ]
                    )
                )
                raise
            if any(
                self[j].end < self[j + 1].start for j in range(j2, j3)
            ):  # assure there is an intron between closest tss/pas and exon
                end = "3" if is_reverse else "5"
                altsplice.setdefault(f"{end}' fragment", []).append(
                    [exons[-1][1], self[j3].end]
                )

        if not altsplice:  # all junctions are contained but not all in one transcript
            altsplice = {"novel combination": []}
            category = 2

        return category, altsplice

    def _check_exon(
        self, j1, j2, is_first, is_reverse, exon: tuple[int, int], exon2=None
    ):
        """checks whether exon is supported by splice graph between nodes j1 and j2

        :param j1: index of first segment ending after exon start (i.e. first overlapping segment)
        :param j2: index of last segment starting before exon end (i.e. last overlapping  segment)
        """

        logger.debug(
            "exon %s between sg node %s and %s/%s (first=%s,rev=%s,e2=%s)",
            exon,
            j1,
            j2,
            len(self),
            is_first,
            is_reverse,
            exon2,
        )
        is_last = exon2 is None
        altsplice = {}
        category = 0
        if (
            j1 > j2
        ):  # exon is not contained at all -> novel exon (or TSS/PAS if first/last)
            category = 3
            if is_first or is_last:
                altsplice = {
                    (
                        "novel intronic PAS"
                        if is_first == is_reverse
                        else "novel intronic TSS"
                    ): [exon]
                }
            else:
                altsplice = {"novel exon": [exon]}
            j2 = j1
        elif (
            is_first and is_last
        ):  # mono-exon (should not overlap a reference monoexon transcript, this is caught earlier)
            altsplice["mono-exon"] = []
            category = 1
        else:  # check splice sites
            if self[j1][0] != exon[0]:  # first splice site missmatch
                if not is_first:
                    # pos="intronic" if self[j1][0]>e[0] else "exonic"
                    kind = "5" if is_reverse else "3"
                    dist = min(
                        (self[j][0] - exon[0] for j in range(j1, j2 + 1)), key=abs
                    )  # the distance to next junction
                    altsplice[f"novel {kind}' splice site"] = [(exon[0], dist)]
                    category = 3
                elif self[j1][0] > exon[0] and not any(
                    j in self._tss for j in range(j1, j2 + 1)
                ):  # exon start is intronic in ref
                    site = "PAS" if is_reverse else "TSS"
                    altsplice.setdefault(f"novel exonic {site}", []).append(
                        (exon[0], self[j1][0])
                    )
                    category = max(1, category)
            if self[j2][1] != exon[1]:  # second splice site missmatch
                if not is_last:
                    # pos="intronic" if self[j2][1]<e[1] else "exonic"
                    # TODO: could also be a "novel intron", if the next "first" splice site is also novel.
                    kind = "3" if is_reverse else "5"
                    dist = min(
                        (self[j][1] - exon[1] for j in range(j1, j2 + 1)), key=abs
                    )  # the distance to next junction
                    altsplice.setdefault(f"novel {kind}' splice site", []).append(
                        (exon[1], dist)
                    )
                    category = 3
                elif self[j2][1] < exon[1] and not any(
                    j in self._pas for j in range(j1, j2 + 1)
                ):  # exon end is intronic in ref & not overlapping pas
                    site = "TSS" if is_reverse else "PAS"
                    altsplice.setdefault(f"novel exonic {site}", []).append(
                        (self[j2][1], exon[1])
                    )
                    category = max(1, category)

        # find intron retentions
        if j1 < j2 and any(
            self[ji + 1].start - self[ji].end > 0 for ji in range(j1, j2)
        ):
            gaps = [ji for ji in range(j1, j2) if self[ji + 1].start - self[ji].end > 0]
            if (
                gaps
                and not (
                    is_first and any(j in self._tss for j in range(gaps[-1] + 1, j2))
                )
                and not (
                    is_last and any(j in self._pas for j in range(j1, gaps[0] + 1))
                )
            ):
                ret_introns = []
                troi = set(self[j1].suc.keys()).intersection(self[j2].pre.keys())
                if troi:
                    j = j1
                    while j < j2:
                        nextj = min(
                            js
                            for transcript_id, js in self[j].suc.items()
                            if transcript_id in troi
                        )
                        if self[nextj].start - self[j].end > 0 and any(
                            self[ji + 1].start - self[ji].end > 0
                            for ji in range(j, nextj)
                        ):
                            ret_introns.append((self[j].end, self[nextj].start))
                        j = nextj
                    if ret_introns:
                        altsplice["intron retention"] = ret_introns
                        category = max(2, category)
        logger.debug("check exon %s resulted in %s", exon, altsplice)
        return altsplice, category

    def _check_junction(self, j1, j2, e, e2):
        """check a junction in the segment graph

        * check presence e1-e2 junction in ref (-> if not exon skipping or novel junction)
            * presence is defined a direct junction from an ref exon (e.g. from self) overlapping e1 to an ref exon overlapping e2
        * AND find j3 and j4: first node overlapping e2  and last node overlapping e2
        * more specifically:
            * j3: first node ending after e2 start, or len(self)
            * j4: last node starting before e2 end (assuming there is such a node)"""
        altsplice = {}
        j3 = next(
            (j for j in range(j2 + 1, len(self)) if self[j][1] > e2[0]), len(self)
        )
        j4 = next(
            (j - 1 for j in range(j3, len(self)) if self[j].start >= e2[1]),
            len(self) - 1,
        )
        if j3 == len(self) or self[j3].start > e2[1]:
            return j3, j4, altsplice  # no overlap with e2
        if (
            e[1] == self[j2].end
            and e2[0] == self[j3].start
            and j3 in self[j2].suc.values()
        ):
            return j3, j4, altsplice  # found direct junction
        # find skipped exons within e1-e2 intron
        exon_skipping = set()
        for j in range(j1, j2 + 1):
            for transcript, j_suc in self[j].suc.items():
                if j_suc <= j2 or j_suc > j4:  # befor junction or exceeding it
                    continue  # -> no exon skipping
                if self._pas[transcript] < j4:  # transcripts ends within e1-e2 intron
                    continue  # -> no exon skipping
                j_spliced = self._get_next_spliced(transcript, j)
                if j_spliced is None:  # end of transcript
                    continue  # -> no exon skipping
                if j_spliced < j3:  # found splice junction from e1 into e1-e2 intron
                    j_spliced_end = self._get_exon_end(transcript, j_spliced)
                    if j_spliced_end < j3:  # ref exon ends within e1-e2 intron
                        exon_skipping.add(transcript)  # found exon skipping
        if exon_skipping:  # path from e1 over another exon e_skip to e2 present
            # now we need to find the exon boundaries of the skipped exon
            exons = []
            e_start = e[1] - 1
            e_end = e[1]
            for j in range(j2 + 1, j3 + 1):
                if e_start and self[j].start > self[j - 1].end:  # intron befor j
                    exons.append([e_start, e_end])
                    e_start = 0
                if exon_skipping.intersection(self[j].suc):  # exon at j
                    if e_start == 0:  # new exon start
                        e_start = self[j].start
                    e_end = self[j].end
                elif e_start:  # intron at j
                    exons.append([e_start, e_end])
                    e_start = 0
            if len(exons) > 1:
                altsplice.setdefault("exon skipping", []).extend(exons[1:])
        elif (
            e[1] == self[j2].end and e2[0] == self[j3].start
        ):  # e1-e2 path is not present, but splice sites are
            altsplice.setdefault("novel junction", []).append(
                [e[1], e2[0]]
            )  # for example mutually exclusive exons spliced togeter

        logger.debug("check junction %s - %s resulted in %s", e[0], e[1], altsplice)

        return j3, j4, altsplice

    def fuzzy_junction(self, exons: list[tuple[int, int]], size: int):
        """Looks for "fuzzy junctions" in the provided transcript.

        For each intron from "exons", look for introns in the splice graph shifted by less than "size".
        These shifts may be produced by ambigious alignments.

        :param exons: A list of exon tuples representing the transcript
        :type exons: list
        :param size: The maximum size of the fuzzy junction
        :type size: int
        :return: a dict with the intron number as key and the shift as value (assuming size is smaller than introns)
        :rtype: dict"""
        fuzzy = {}
        if size < 1:  # no need to check
            return fuzzy
        j1 = 0
        idx = range(j1, len(self))
        for i, exon1 in enumerate(exons[:-1]):
            exon2 = exons[i + 1]
            try:
                # find j1: first node intersecting size range of e1 end
                if self[j1].end + min(size, exon1[1] - exon1[0]) < exon1[1]:
                    j1 = next(j for j in idx if self[j].end + size >= exon1[1])
            except (StopIteration, IndexError):  # transcript end - we are done
                break
            shift = []
            while j1 < len(self) and self[j1].end - exon1[1] <= min(
                size, exon1[1] - exon1[0]
            ):  # in case there are several nodes starting in the range around e1
                shift_e1 = self[j1].end - exon1[1]
                # print(f'{i} {e1[1]}-{e2[0]} {shift_e1}')
                if shift_e1 == 0:  # no shift required at this intron
                    break
                if any(
                    self[j2].start - exon2[0] == shift_e1
                    for j2 in set(self[j1].suc.values())
                ):
                    shift.append(shift_e1)
                j1 += 1
            else:  # junction not found in sg
                if shift:  # but shifted juction is present
                    fuzzy[i] = sorted(shift, key=abs)[
                        0
                    ]  # if there are several possible shifts, provide the smallest
        return fuzzy

    def find_splice_sites(self, splice_junctions: list[tuple[int, int]]):
        """Checks whether the splice sites of a new transcript are present in the segment graph.

        :param splice_junctions: A list of 2-tuples with the splice site positions
        :return: boolean array indicating whether the splice site is contained or not"""

        sites = np.zeros(len(splice_junctions) * 2, dtype=bool)
        splice_junction_starts = {}
        splice_junction_ends = {}
        for i, splice_junction in enumerate(splice_junctions):
            splice_junction_starts.setdefault(splice_junction[0], []).append(i)
            splice_junction_ends.setdefault(splice_junction[1], []).append(i)

        # check exon ends
        for splice_junction_start, idx in sorted(splice_junction_starts.items()):
            _, node = self._get_node_ending_at(splice_junction_start)
            if node is None:
                continue
            # Check if there is a true splice site behind
            if any(self[s].start > node.end for s in node.suc.values()):
                for i in idx:
                    sites[i * 2] = True

        # check exon starts
        for splice_junction_end, idx in sorted(splice_junction_ends.items()):
            _, node = self._get_node_starting_at(splice_junction_end)
            if node is None:
                continue
            # Check if there is a true splice site in front
            if any(self[p].end < node.start for p in node.pre.values()):
                for i in idx:
                    sites[i * 2 + 1] = True
        return sites

    def get_overlap(self, exons):
        """Compute the exonic overlap of a new transcript with the segment graph.

        :param exons: A list of exon tuples representing the transcript
        :type exons: list
        :return: a tuple: the overlap with the gene, and a list of the overlaps with the transcripts
        """

        ol = 0
        j = 0
        transcript_overlap = [0 for _ in self._pas]
        for e in exons:
            while self[j].end < e[0]:  # no overlap, go on
                j += 1
                if j == len(self):
                    return ol, transcript_overlap
            while self[j].start < e[1]:
                i_end = min(e[1], self[j].end)
                i_start = max(e[0], self[j].start)
                ol += i_end - i_start
                for transcript_id in self[j].suc.keys():
                    transcript_overlap[transcript_id] += i_end - i_start
                for transcript_id, pas in enumerate(self._pas):
                    if pas == j:
                        transcript_overlap[transcript_id] += i_end - i_start
                if self[j].end > e[1]:
                    break
                j += 1
                if j == len(self):
                    return ol, transcript_overlap

        return ol, transcript_overlap

    def get_intron_support_matrix(self, exons):
        """Check the intron support for the provided transcript w.r.t. transcripts from self.

        This is supposed to be helpful for the analysis of novel combinations of known splice sites.


        :param exons: A list of exon positions defining the transcript to check.
        :return: A boolean array of shape (n_transcripts in self)x(len(exons)-1).
            An entry is True iff the intron from "exons" is present in the respective transcript of self.
        """
        node_iter = iter(self)
        ism = np.zeros(
            (len(self._tss), len(exons) - 1), bool
        )  # the intron support matrix
        for intron_nr, (e1, e2) in enumerate(pairwise(exons)):
            try:
                node = next(n for n in node_iter if n.end >= e1[1])
            except StopIteration:
                return ism
            if node.end == e1[1]:
                for transcript_id, suc in node.suc.items():
                    if self[suc].start == e2[0]:
                        ism[transcript_id, intron_nr] = True
        return ism

    def get_exon_support_matrix(self, exons):
        """Check the exon support for the provided transcript w.r.t. transcripts from self.

        This is supposed to be helpful for the analysis of novel combinations of known splice sites.


        :param exons: A list of exon positions defining the transcript to check.
        :return: A boolean array of shape (n_transcripts in self)x(len(exons)-1).
            An entry is True iff the exon from "exons" is fully covered in the respective transcript of self.
            First and last exon are checked to overlap the first and last exon of the ref transcript but do not need to be fully covered
        """
        esm = np.zeros((len(self._tss), len(exons)), bool)  # the intron support matrix
        for transcript_number, tss in enumerate(
            self._tss
        ):  # check overlap of first exon
            for j in range(tss, len(self)):
                if has_overlap(self[j], exons[0]):
                    esm[transcript_number, 0] = True
                elif (
                    self[j].suc.get(transcript_number, None) == j + 1
                    and j - 1 < len(self)
                    and self[j].end == self[j + 1].start
                ):
                    continue
                break
        for transcript_number, pas in enumerate(
            self._pas
        ):  # check overlap of last exon
            for j in range(pas, -1, -1):
                if has_overlap(self[j], exons[-1]):
                    esm[transcript_number, -1] = True
                elif (
                    self[j].pre.get(transcript_number, None) == j - 1
                    and j > 0
                    and self[j].start == self[j - 1].end
                ):
                    continue
                break

        j2 = 0
        for e_nr, e in enumerate(exons[1:-1]):
            j1 = next((j for j in range(j2, len(self)) if self[j].end > e[0]))
            # j1: index of first segment ending after exon start (i.e. first overlapping segment)
            j2 = next(
                (j - 1 for j in range(j1, len(self)) if self[j].start >= e[1]),
                len(self) - 1,
            )
            # j2: index of last segment starting befor exon end (i.e. last overlapping segment)
            if self[j1].start <= e[0] and self[j2].end >= e[1]:
                covered = set.intersection(
                    *(set(self[j].suc) for j in range(j1, j2 + 1))
                )
                if covered:
                    esm[covered, e_nr + 1] = True
        return esm

    def get_exonic_region(self):
        regs = [[self[0].start, self[0].end]]
        for n in self[1:]:
            if n.start == regs[-1][1]:
                regs[-1][1] = n.end
            else:
                regs.append([n.start, n.end])
        return regs

    def get_intersects(self, exons):
        """Computes the splice junction exonic overlap of a new transcript with the segment graph.

        :param exons: A list of exon tuples representing the transcript
        :type exons: list
        :return: the splice junction overlap and exonic overlap"""

        intersect = [0, 0]
        i = j = 0
        while True:
            if self[j][0] == exons[i][0] and any(
                self[k][1] < self[j][0] for k in self[j].pre.values()
            ):
                intersect[
                    0
                ] += 1  # same position and actual splice junction(not just tss or pas and internal junction)
            if self[j][1] == exons[i][1] and any(
                self[k][0] > self[j][1] for k in self[j].suc.values()
            ):
                intersect[0] += 1
            if self[j][1] > exons[i][0] and exons[i][1] > self[j][0]:  # overlap
                intersect[1] += min(self[j][1], exons[i][1]) - max(
                    self[j][0], exons[i][0]
                )
            if exons[i][1] < self[j][1]:
                i += 1
            else:
                j += 1
            if i == len(exons) or j == len(self):
                return intersect

    @deprecated
    def _find_ts_candidates(self, coverage):
        """Computes a metric indicating template switching."""
        for i, gnode in enumerate(self._graph[:-1]):
            if (
                self._graph[i + 1].start == gnode.end
            ):  # jump candidates: introns that start within an exon
                jumps = {
                    idx: n
                    for idx, n in gnode.suc.items()
                    if n > i + 1 and self._graph[n].start == self._graph[n - 1].end
                }
                # find jumps (n>i+1) and check wether they end within an exon begin(jumptarget)==end(node before)
                jump_weight = {}
                for idx, target in jumps.items():
                    jump_weight.setdefault(target, [0, []])
                    jump_weight[target][0] += coverage[:, idx].sum(0)
                    jump_weight[target][1].append(idx)

                for target, (w, idx) in jump_weight.items():
                    long_idx = set(
                        idx for idx, n in gnode.suc.items() if n == i + 1
                    ) & set(
                        idx for idx, n in self[target].pre.items() if n == target - 1
                    )
                    try:
                        longer_weight = coverage[:, list(long_idx)].sum()
                    except IndexError:
                        print(long_idx)
                        raise
                    yield gnode.end, self[target].start, w, longer_weight, idx

    def _is_spliced(self, transcript_id, node_index1, node_index2):
        "checks if transcript is spliced (e.g. has an intron) between nodes ni1 and ni2"
        if any(
            self[i].end < self[i + 1].start for i in range(node_index1, node_index2)
        ):  # all transcripts are spliced
            return True
        if all(transcript_id in self[i].suc for i in range(node_index1, node_index2)):
            return False
        return True

    def _get_next_spliced(self, transcript_id: int, node: int):
        "find the next spliced node for given transcript"
        while node != self._pas[transcript_id]:
            try:
                next_node = self[node].suc[
                    transcript_id
                ]  # raises error if transcript_id not in node.suc
            except KeyError:
                logger.error(
                    "transcript_id %s seems to be not in node %s", transcript_id, node
                )
                raise
            if self[next_node].start > self[node].end:
                return next_node
            node = next_node
        return None

    def _get_exon_end(self, transcript_id: int, node: int):
        "find the end of the exon to which node belongs for given transcript"
        while node != self._pas[transcript_id]:
            try:
                next_node = self[node].suc[
                    transcript_id
                ]  # raises error if transcript_id not in node.suc
            except KeyError:
                logger.error(
                    "transcript_id %s seems to be not in node %s", transcript_id, node
                )
                raise
            if self[next_node].start > self[node].end:
                return node
            node = next_node
        return node

    def _get_exon_end_all(self, node: int):
        "find the end of the exon considering all transcripts"
        while node < len(self) - 1 and self[node].end == self[node + 1].start:
            node += 1
        return node

    def _get_exon_start(self, transcript_id: int, node: int):
        "find the start of the exon to which node belongs for given transcript"
        while node != self._tss[transcript_id]:
            try:
                next_node = self[node].pre[
                    transcript_id
                ]  # raises error if transcript_id not in node.pre
            except KeyError:
                logger.error(
                    "transcript_id %s seems to be not in node %s", transcript_id, node
                )
                raise
            if self[next_node].end < self[node].start:
                return node
            node = next_node
        return node

    def _get_exon_start_all(self, node):
        "find the start of the exon considering all transcripts"
        while node > 0 and self[node - 1].end == self[node].start:
            node -= 1
        return node

    def _find_splice_bubbles_at_position(self, types: list[ASEType], pos):
        """function to refind bubbles at a certain genomic position.
        This turns out to be fundamentally different compared to iterating over all bubbles, hence it is a complete rewrite of the function.
        On the positive site, the functions can validate each other. I tried to reuse the variable names.
        If both functions yield same results, there is a good chance that the complex code is actually right.
        """
        # TODO: format of pos isn't documented anywhere and the intended isn't clear from the code
        # more a comment than a docstring...
        if any(type in ["ES", "3AS", "5AS", "IR", "ME"] for type in types):
            try:
                i, node_A = self._get_node_ending_at(pos[0])
                if len(pos) == 3:
                    middle = [
                        next(
                            idx
                            for idx, node in enumerate(self[i:], i)
                            if node.start > pos[1]
                        )
                    ]
                    j, node_B = self._get_node_starting_at(pos[2], middle[0])
                else:
                    j, node_B = self._get_node_starting_at(pos[-1], i)
                    middle = range(i + 2, j)
            except StopIteration as e:
                raise ValueError(
                    f"cannot find segments at {pos} in segment graph"
                ) from e

            direct: set[int] = set()  # primary
            indirect: dict[ASEType, set[int]] = {
                "ES": set(),
                "3AS": set(),
                "5AS": set(),
                "IR": set(),
            }
            for transcript, node_id in node_A.suc.items():
                if transcript not in node_B.pre:
                    continue
                if node_id == j:
                    direct.add(transcript)
                    continue
                five_prime = self[node_id].start == node_A.end
                three_prime = self[node_B.pre[transcript]].end == node_B.start
                if five_prime and three_prime:
                    type = "IR"
                elif five_prime:
                    type = "5AS" if self.strand == "+" else "3AS"
                elif three_prime:
                    type = "3AS" if self.strand == "+" else "5AS"
                else:
                    type = "ES"
                indirect[type].add(transcript)
            for type in types:
                if type in ["ES", "3AS", "5AS", "IR"] and direct and indirect[type]:
                    yield list(direct), list(indirect[type]), i, j, type
                elif type == "ME" and len(indirect["ES"]) > 2:
                    me: list[ASEvent] = list()
                    seen_alt = set()
                    for middle_idx in middle:
                        # alternative exons before the middle node, primary exons after the middle node (or the middle node itself)
                        alt, prim = set(), set()
                        for transcript in indirect["ES"]:
                            if node_B.pre[transcript] < middle_idx:
                                alt.add(transcript)
                            elif node_A.suc[transcript] >= middle_idx:
                                prim.add(transcript)
                        # make sure there is at least one new alt transcript with this middle node.
                        if prim and alt - seen_alt:
                            me.append((list(prim), list(alt), i, j, "ME"))
                            seen_alt.update(alt)
                    seen_prim = set()
                    for me_event in reversed(me):
                        # report only if there is a new transcript in prim with respect to middle nodes to the right
                        if me_event[0] - seen_prim:
                            yield me_event
                            seen_prim.update(me_event[0])
        if any(type in ["TSS", "PAS"] for type in types):
            try:
                i, _ = next((idx, n) for idx, n in enumerate(self) if n.start >= pos[0])
                j, _ = next(
                    ((idx, n) for idx, n in enumerate(self[i:], i) if n.end >= pos[-1]),
                    (len(self) - 1, self[-1]),
                )
            except StopIteration as e:
                raise ValueError(
                    f"cannot find segments at {pos} in segment graph"
                ) from e

            alt_types = ["TSS", "PAS"] if self.strand == "+" else ["PAS", "TSS"]
            # TODO: Second condition is always false, because node_B starts at pos[-1]
            if any(type == alt_types[0] for type in types) and node_B.end == pos[-1]:
                alt = {
                    transcript
                    for transcript, tss in enumerate(self._tss)
                    if i <= tss <= j and self._get_exon_end(transcript, tss) == j
                }
                if alt:  # find compatible alternatives: end after tss /start before pas
                    prim = [
                        transcript
                        for transcript, pas in enumerate(self._pas)
                        if transcript not in alt and pas > j
                    ]  # prim={transcript for transcript in range(len(self._tss)) if transcript not in alt}
                    if prim:
                        yield list(prim), list(alt), i, j, alt_types[0]
            # TODO: Second condition is always false, because node_A ends at pos[0]
            if any(type == alt_types[1] for type in types) and node_A.start == pos[0]:
                alt = {
                    transcript
                    for transcript, pas in enumerate(self._pas)
                    if i <= pas <= j and self._get_exon_start(transcript, pas) == i
                }
                if alt:
                    prim = [
                        transcript
                        for transcript, tss in enumerate(self._tss)
                        if transcript not in alt and tss < i
                    ]
                    if prim:
                        yield list(prim), list(alt), i, j, alt_types[1]

    def find_splice_bubbles(
        self, types: Optional[str | list[ASEType]] = None, pos=None
    ):
        """Searches for alternative paths in the segment graph ("bubbles").

        Bubbles are defined as combinations of nodes x_s and x_e with more than one path from x_s to x_e.

        :param types: A tuple with event types to find. Valid types are ('ES', '3AS', '5AS', 'IR', 'ME', 'TSS', 'PAS').
            If ommited, all types are considered
        :param pos: If specified, restrict the search on specific position.
            This is useful to find the supporting transcripts for a given type if the position is known.

        :return: Tuple with 1) transcript indices of primary (e.g. most direct) paths and 2) alternative paths respectively,
            as well as 3) start and 4) end node ids and 5) type of alternative event
            ('ES', '3AS', '5AS', 'IR', 'ME', 'TSS', 'PAS')"""

        if types is None:
            types: list[ASEType] = ("ES", "3AS", "5AS", "IR", "ME", "TSS", "PAS")
        elif isinstance(types, str):
            types = (types,)
        alt_types: list[ASEType] = (
            ("ES", "5AS", "3AS", "IR", "ME", "PAS", "TSS")
            if self.strand == "-"
            else ("ES", "3AS", "5AS", "IR", "ME", "TSS", "PAS")
        )

        if pos is not None:
            for prim, alt, i, j, alt_type in self._find_splice_bubbles_at_position(
                types, pos
            ):
                yield list(prim), list(alt), i, j, alt_type
            return
        if any(type in types for type in ("ES", "3AS", "5AS", "IR", "ME")):
            # list of spliced and unspliced transcripts joining in B
            inB_sets: list[tuple[set[int], set[int]]] = [(set(), set())]
            # node_matrix=self.get_node_matrix()
            for i, node_B in enumerate(self[1:]):
                inB_sets.append((set(), set()))
                unspliced = self[i].end == node_B.start
                for transcript_id, node_id in node_B.pre.items():
                    inB_sets[i + 1][unspliced and node_id == i].add(transcript_id)
            for i, node_A in enumerate(self):
                # target nodes for junctions from node A ordered by intron size
                junctions = sorted(list(set(node_A.suc.values())))
                if len(junctions) < 2:
                    continue  # no alternative
                # transcripts supporting the different junctions
                outA_sets: dict[int, set[int]] = {}
                for transcript_id, node_id in node_A.suc.items():
                    outA_sets.setdefault(node_id, set()).add(transcript_id)
                unspliced = node_A.end == self[junctions[0]].start
                alternative: tuple[set[int], set[int]] = (
                    ({}, outA_sets[junctions[0]])
                    if unspliced
                    else (outA_sets[junctions[0]], {})
                )
                # node_C_dict aims to avoid recalculation of node_C for ME events
                # transcript_id -> node at start of 2nd exon C for transcript_id such that there is one exon (B) (and both flanking introns) between node_A and C; None if transcript ends
                node_C_dict: dict[int, int | None] = {}
                # ensure that only ME events with novel transcript_id are reported
                me_alt_seen = set()
                logger.debug(
                    "checking node %s: %s (%s)",
                    i,
                    node_A,
                    list(zip(junctions, [outA_sets[j] for j in junctions])),
                )
                # start from second, as first does not have an alternative
                for j_idx, junction in enumerate(junctions[1:], 1):
                    # check that transcripts extend beyond node_B
                    alternative = [
                        {
                            transcript_id
                            for transcript_id in alternative[i]
                            if self._pas[transcript_id] > junction
                        }
                        for i in range(2)
                    ]
                    logger.debug(alternative)
                    # alternative transcript sets for the 4 types
                    found = [
                        trL1.intersection(trL2)
                        for trL1 in alternative
                        for trL2 in inB_sets[junction]
                    ]
                    # 5th type: mutually exclusive (outdated handling of ME for reference)
                    # found.append(set.union(*alternative)-inB_sets[junction][0]-inB_sets[junction][1])
                    logger.debug(
                        "checking junction %s (transcript_id=%s) and found %s at B=%s",
                        junction,
                        outA_sets[junction],
                        found,
                        inB_sets[junction],
                    )
                    for alt_type_id, alt in enumerate(found):
                        if alt_types[alt_type_id] in types and alt:
                            yield list(outA_sets[junction]), list(
                                alt
                            ), i, junction, alt_types[alt_type_id]
                    # me_alt=set.union(*alternative)-inB_sets[junction][0]-inB_sets[junction][1] #search 5th type: mutually exclusive
                    if "ME" in types:
                        # search 5th type: mutually exclusive - needs to be spliced
                        me_alt = (
                            alternative[0]
                            - inB_sets[junction][0]
                            - inB_sets[junction][1]
                        )
                        # there is at least one novel alternative transcript
                        if me_alt - me_alt_seen:
                            # for ME we need to find (potentially more than one) node_C where the alternatives rejoin
                            # find node_C for all me_alt
                            for transcript_id in me_alt:
                                node_C_dict.setdefault(
                                    transcript_id,
                                    self._get_next_spliced(
                                        transcript_id, node_A.suc[transcript_id]
                                    ),
                                )
                                # transcript end in node_B, no node_C
                                if node_C_dict[transcript_id] is None:
                                    # those are not of interest for ME
                                    me_alt_seen.add(transcript_id)
                            # dict of node_C indices to sets of node_A-intron-node_B-intron-node_C transcripts, where node_B >= junction
                            inC_sets: dict[int, set[int]] = {}
                            # all primary junctions
                            for node_B_i in junctions[j_idx:]:
                                # primary transcripts
                                for transcript_id in outA_sets[node_B_i]:
                                    # find node_C
                                    node_C_dict.setdefault(
                                        transcript_id,
                                        self._get_next_spliced(transcript_id, node_B_i),
                                    )
                                    if node_C_dict[transcript_id] is None:
                                        continue
                                    # first, all primary transcript_id/nCs from junction added
                                    if node_B_i == junction:
                                        inC_sets.setdefault(
                                            node_C_dict[transcript_id], set()
                                        ).add(transcript_id)
                                    # then add primary transcript_id that also rejoin at any of the junction nC
                                    elif node_C_dict[transcript_id] in inC_sets:
                                        inC_sets[node_C_dict[transcript_id]].add(
                                            transcript_id
                                        )
                                # no node_C for any of the junction primary transcript_id - no need to check the other primaries
                                if not inC_sets:
                                    break
                            for node_C_i, me_prim in sorted(inC_sets.items()):
                                found_alt = {
                                    transcript_id
                                    for transcript_id in me_alt
                                    if node_C_dict[transcript_id] == node_C_i
                                }
                                # ensure, there is a new alternative
                                if found_alt - me_alt_seen:
                                    yield (
                                        list(me_prim),
                                        list(found_alt),
                                        i,
                                        node_C_i,
                                        "ME",
                                    )
                                    # me_alt=me_alt-found_alt
                                    me_alt_seen.update(found_alt)
                    # now transcripts supporting junction join the alternatives
                    alternative[0].update(outA_sets[junction])
        if "TSS" in types or "PAS" in types:
            yield from self._find_start_end_events(types)

    def _find_start_end_events(
        self, types: list[ASEType]
    ) -> Generator[ASEvent, None, None]:
        """Searches for alternative TSS/PAS in the segment graph.

        All transcripts sharing the same first/ last node in the splice graph are summarized.
        All pairs of TSS/PAS are returned. The primary set is the set with the smaller coordinate,
        the alternative the one with the larger coordinate.

        :return: Tuple with 1) transcript ids sharing common start exon and 2) alternative transcript ids respectively,
            as well as 3) start and 4) end node ids of the exon and 5) type of alternative event ("TSS" or "PAS")
        """
        tss: dict[int, set[int]] = {}
        pas: dict[int, set[int]] = {}
        # tss_start: dict[int, int] = {}
        # pas_end: dict[int, int] = {}
        for transcript_id, (start1, end1) in enumerate(zip(self._tss, self._pas)):
            tss.setdefault(start1, set()).add(transcript_id)
            pas.setdefault(end1, set()).add(transcript_id)
        alt_types: list[ASEType] = (
            ["PAS", "TSS"] if self.strand == "-" else ["TSS", "PAS"]
        )
        if alt_types[0] in types:
            for (prim_node_id, prim_set), (
                alt_node_id,
                alt_set,
            ) in itertools.combinations(
                sorted(tss.items(), key=lambda item: item[0]), 2
            ):
                yield (
                    list(prim_set),
                    list(alt_set),
                    prim_node_id,
                    alt_node_id,
                    alt_types[0],
                )
        if alt_types[1] in types:
            for (prim_node_id, prim_set), (
                alt_node_id,
                alt_set,
            ) in itertools.combinations(
                sorted(pas.items(), key=lambda item: item[0]), 2
            ):
                yield (
                    list(prim_set),
                    list(alt_set),
                    prim_node_id,
                    alt_node_id,
                    alt_types[1],
                )

    def is_exonic(self, position):
        """Checks whether the position is within an exon.

        :param position: The genomic position to check.
        :return: True, if the position overlaps with an exon, else False."""
        for node in self:
            if node[0] <= position and node[1] >= position:
                return True
        return False

    def _get_all_exons(self, nodeX, nodeY, transcript):
        "get all exonic regions between (including) nodeX to nodeY for transcripts transcript"
        # TODO: add option to extend first and last exons
        node = max(nodeX, self._tss[transcript])  # if tss>nodeX start there
        if (
            transcript not in self[node].pre and self._tss[transcript] != node
        ):  # nodeX is not part of transcript
            # find first node in transcript after nodeX but before nodeY
            for i in range(node, nodeY + 1):
                if transcript in self[node].suc:
                    node = i
                    break
            else:
                return []
        if node > nodeY:
            return []
        exons = [[self[node].start, self[node].end]]
        while node < nodeY:
            try:
                node = self[node].suc[transcript]
            except KeyError:  # end of transcript before nodeY was reached
                break
            if self[node].start == exons[-1][1]:
                exons[-1][1] = self[node].end
            else:
                exons.append([self[node].start, self[node].end])
        return [tuple(e) for e in exons]

    def __getitem__(self, key: int):
        return self._graph[key]

    def __iter__(self):
        return iter(self._graph)

    def __len__(self):
        return len(self._graph)

    def events_dist(self, event1: ASEvent, event2: ASEvent):
        """
        returns the distance (in nucleotides) between two Alternative Splicing Events.

        :param event1: event obtained from .find_splice_bubbles()
        :param event2: event obtained from .find_splice_bubbles()
        """

        # the event begins at the beginning of the first exon and ends at the end of the last exon
        e1_coor = [
            self[event1[2]].start,
            self[event1[3]].end,
        ]  # starting and ending coordinates of event 1
        e2_coor = [
            self[event2[2]].start,
            self[event2[3]].end,
        ]  # starting and ending coordinates of event 2

        return _interval_dist(e1_coor, e2_coor)

    def _get_node_starting_at(
        self, coordinate: int, start_index=0
    ) -> tuple[int, SegGraphNode]:
        """
        return the node in the splice graph starting at the given coordinate.
        """
        for i, node in enumerate(self[start_index:], start_index):
            if node.start == coordinate:
                return i, node
            if node.start > coordinate:
                return -1, None
        return -1, None

    def _get_node_ending_at(
        self, coordinate: int, start_index=0
    ) -> tuple[int, SegGraphNode]:
        """
        return the node in the splice graph ending at the given coordinate.
        """
        for i, node in enumerate(self[start_index:], start_index):
            if node.end == coordinate:
                return i, node
            if node.end > coordinate:
                return -1, None
        return -1, None

    def _get_event_coordinate(self, event: ASEvent):
        if event[4] == "TSS":
            return (self[event[2]].start, self[event[3]].start)
        elif event[4] == "PAS":
            return (self[event[2]].end, self[event[3]].end)
        else:
            return (self[event[2]].end, self[event[3]].start)


class SegGraphNode(tuple):
    """A node in a segment graph represents an exonic segment."""

    def __new__(cls, start, end, pre=None, suc=None):
        if pre is None:
            pre = dict()
        if suc is None:
            suc = dict()
        return super(SegGraphNode, cls).__new__(cls, (start, end, pre, suc))

    def __getnewargs__(self):
        return tuple(self)

    @property
    def start(self) -> int:
        """the (genomic 5') start of the segment"""
        return self.__getitem__(0)

    @property
    def end(self) -> int:
        """the (genomic 3') end of the segment"""
        return self.__getitem__(1)

    @property
    def pre(self) -> dict[int, int]:
        """the predecessor segments of the segment (linked nodes upstream)"""
        return self.__getitem__(2)

    @property
    def suc(self) -> dict[int, int]:
        """the successor segments of the segment (linked nodes downstream)"""
        return self.__getitem__(3)


class SpliceGraph:
    """(Experimental) Splice Graph Implementation

    Nodes represent splice sites and are tuples of genomic positions and a "lower" flag.
    The "lower flag" is true, if the splice site is a genomic 5' end of an exon
    Nodes are kept sorted, so iteration over splicegraph returns all nodes in genomic order
    Edges are assessed with SpliceGraph.pre(node, [transcript_number]) and SpliceGraph.suc(node, [transcript_number]) functions.
    If no transcript_number is provided, a dict with all incoming/outgoing edges is returned
    """

    # @experimental

    def __init__(self, is_reverse, graph, fwd_starts, rev_starts):
        self.is_reverse = is_reverse
        self._graph = graph
        self._fwd_starts = fwd_starts
        self._rev_starts = rev_starts

    @classmethod
    def from_transcript_list(cls, exon_lists, strand):
        """Compute the splice graph from a list of transcripts

        :param exon_lists: A list of transcripts, which are lists of exons, which in turn are (start,end) tuples
        :type exon_lists: list
        :param strand: the strand of the gene, either "+" or "-"
        :return: The SpliceGraph object
        :rtype: SpliceGraph"""

        assert strand in "+-", 'strand must be either "+" or "-"'

        graph = SortedDict()
        fwd_starts = [(exons[0][0], True) for exons in exon_lists]  # genomic 5'
        rev_starts = [(exons[-1][1], False) for exons in exon_lists]  # genomic 3'

        for transcript_number, exons in enumerate(exon_lists):
            graph.setdefault((exons[0][0], True), ({}, {}))

            for i, (b1, b2) in enumerate(
                pairwise(pos for exon in exons for pos in exon)
            ):
                graph.setdefault((b2, bool(i % 2)), ({}, {}))
                graph[b2, bool(i % 2)][1][transcript_number] = b1, not bool(
                    (i) % 2
                )  # successor
                graph[b1, not bool(i % 2)][0][transcript_number] = b2, bool(
                    (1) % 2
                )  # predesessor
        sg = cls(strand == "-", graph, fwd_starts, rev_starts)
        return sg

    def __iter__(self):
        return self._graph.__iter__()

    def __len__(self):
        return self._graph.__len__()

    @experimental
    def add(self, exons) -> None:
        """
        Add one transcript to the existing graph.

        :param exons: A list of exon tuples representing the transcript to add
        :type exons: list
        """

        transcript_number = len(self._fwd_starts)
        self._fwd_starts.append((exons[0][0], True))  # genomic 5'
        self._rev_starts.append((exons[-1][1], False))  # genomic 3'
        self._graph.setdefault((exons[0][0], True), ({}, {}))
        for i, (b1, b2) in enumerate(pairwise(pos for exon in exons for pos in exon)):
            self._graph.setdefault((b2, bool(i % 2)), ({}, {}))
            self._graph[b2, bool(i % 2)][1][transcript_number] = b1, not bool(
                (i) % 2
            )  # successor
            self._graph[b1, not bool(i % 2)][0][transcript_number] = b2, bool((1) % 2)

    def suc(self, node, transcript_number=None) -> Union[int, dict]:
        """get index of successor node (next genomic upstream node) of transcript, or, if transcript_number is omitted, a dict with successors for all transcripts.

        :param node: index of the originating node
        :type node: int
        :param transcript_number: index of the transcript (optional)
        :type transcript_number: int"""
        edges = self._graph[node][0]
        if transcript_number is None:
            return edges
        return edges[transcript_number]

    def pre(self, node, transcript_number=None) -> Union[int, dict]:
        """get index of predesessor node (next genomic downstream node) of transcript, or, if transcript_number is omitted, a dict with predesessors for all transcripts.

        :param node: index of the originating node
        :type node: int
        :param transcript_number: index of the transcript (optional)
        :type transcript_number: int"""

        edges = self._graph[node][1]
        if transcript_number is None:
            return edges
        return edges[transcript_number]
