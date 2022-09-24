import re

from Bio import SeqIO, pairwise2
from Bio.Align import substitution_matrices
from sklearn import cluster


class FastaOperator:
    def operate(self, fasta):
        pass

    @property
    def name(self):
        return str(self.__class__.__name__.lower())


class SequenceClusterer(FastaOperator):
    def __init__(self, num_clusters=None, matrix="blosum62", gap_open=-10, gap_extend=-0.5):
        self.num_clusters = num_clusters
        self.matrix = matrix
        self.gap_open = gap_open
        self.gap_extend = gap_extend

    def _prepare_scores(self, sequences):
        num_seqs = len(sequences)
        scores = [[0 for i in range(num_seqs)] for j in range(num_seqs)]
        matrix = substitution_matrices.load(self.matrix)
        for i in range(0, num_seqs):
            for j in range(0, num_seqs):
                a = pairwise2.align.globalds(sequences[i].seq, sequences[j].seq,
                                             matrix, self.gap_open, self.gap_extend)
                (s1, s2, score, start, end) = a[0]
                scores[i][j] = score
                if not j % 10:
                    print(j)
        return scores

    def cluster(self, sequences):
        scores = self._prepare_scores(sequences)
        num_clusters = self.num_clusters or int(0.1 * len(sequences))
        kmeans = cluster.KMeans(num_clusters)
        results = kmeans.fit(scores)
        return results

    def operate(self, fasta):
        return self.cluster(fasta.sequences)


class FastaProcessor:
    def __init__(self, path):
        self.path = path
        self._operators = []
        self._sequences = []
        self._results = {}

    @property
    def sequences(self):
        if self._sequences:
            return self._sequences
        records = list(SeqIO.parse(self.path, "fasta"))
        self._sequences = records
        return self._sequences

    def add_operator(self, op):
        self._operators.append(op)

    def operate_on(self):
        for op in self._operators:
            result = op.operate(self)
            self._results[op.name()] = result
        return self._results


class CdHitAnalyzer:
    def __init__(self, db_clstr_path, db_path=None):
        self.db_path = db_path
        self.db_clstr_path = db_clstr_path

    def _parse_line(self, line):
        length = int(re.search('[1-9]+(?=aa)', line).group(0))
        id_ = re.search('>([^\s]+)', line).group(1)
        id_ = id_.strip('...')
        try:
            pr = float(line.split(' ')[-1].strip('%\n'))
        except Exception:
            pr = None
        return length, id_, pr

    def get_clusters(self, max_cluster_similarity=0.6, size=0.1):
        data = {}
        with open(self.db_clstr_path, 'r') as ww:
            st_read = False
            curr_cl = None
            for line in ww:
                if line.startswith('>'):
                    st_read = True
                    curr_cl = next(ww)
                    ll, id_, sim = self._parse_line(curr_cl)
                    curr_cl = id_
                    data[id_] = []
                elif st_read:
                    ll, id_, sim = self._parse_line(line)
                    if sim / 100 <= max_cluster_similarity:
                        data[curr_cl].append(id_)

        for key, value in data.items():
            data[key] = value[:int(len(value) * size)]
        return data

    def prepare_test_set(self, max_size=0.1, sim=0.6, path='test_set.fasta',
                         seq_dict=None):
        clstr = self.get_clusters(sim, max_size)
        seq_dict = seq_dict or {}
        with open(path, 'w') as pp:
            for key, value in clstr.items():
                s_name = seq_dict.get(key, 'cluster_rep')
                pp.write(f'>{key}\n{s_name}\n')
                for c in value:
                    s_name = seq_dict.get(key, 'cluster_child')
                    pp.write(f'>{c}\n{s_name}\n')


def prepare_test_set():
    seq_dict = SeqIO.parse('ma_predictor/src/samcc_seq_0.fasta', 'fasta')
    seq_dict = {seq.id: str(seq.seq) for seq in seq_dict}
    w = CdHitAnalyzer('ma_predictor/data/input/cd_hit.clstr')
    w.prepare_test_set(seq_dict=seq_dict)
