import numpy as np


class Metrics:
    @staticmethod
    def mse(tr, exp, ignore=None):
        sm = 0
        ignore = ignore or set()
        for t, e in zip(tr, exp):
            t = t if isinstance(t, (float, int, np.int64, np.float64, np.int32, np.float32)) else t[0]
            e = e if isinstance(e, (float, int, np.int64, np.float64, np.int32, np.float32)) else e[0]
            if ignore and (t in ignore or e in ignore):
                continue
            sm += (t - e) ** 2
        res = sm / len(exp)
        return res if isinstance(res, float) else res[0]

    @staticmethod
    def f1(tr, exp, sep=10000):
        unk_tr, unk_pred = 0, 0
        common = 0
        for t, e in zip(tr, exp):
            if t != sep:
                unk_tr += 1
            if e != sep:
                unk_pred += 1
            if t != sep and e != sep:
                common += 1
        if unk_tr == 0 or unk_pred == 0 or common == 0:
            return 0
        rec = common / unk_tr
        prec = common / unk_pred
        f1 = 2 / ((1 / rec) + (1 / prec))
        return f1

    @staticmethod
    def mse_f1(tr, exp, sep=10000, limit=0.8):
        if Metrics.f1(tr, exp, sep=sep) < limit:
            return None
        return Metrics.mse(tr, exp, ignore={sep})
