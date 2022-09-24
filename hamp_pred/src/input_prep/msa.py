import numpy as np
from Bio import AlignIO


class Msa:
    def __init__(self, path, fmt='fasta'):
        self.path = path
        self.fmt = fmt
        self._align = AlignIO.read(self.path, self.fmt)

    @property
    def matrix(self):
        data = []
        for seq in self._align:
            data.append(list(seq.seq))
        return np.array(data)

    @property
    def sequences(self):
        sequences = []
        for seq in self._align:
            sequences.append(str(seq.seq).replace('~', ''))
        return sequences

    def apply_results_on(self, results):
        app = []
        shp = results[0][-1].shape[-1]
        for n, row in enumerate(self.matrix):
            c = 0
            r_app = []
            for pos in row:
                if pos != '~':
                    r_app.append(results[n][c])
                    c += 1
                else:
                    r_app.append([np.nan] * shp)
            app.append(r_app)
        return np.array(app)
