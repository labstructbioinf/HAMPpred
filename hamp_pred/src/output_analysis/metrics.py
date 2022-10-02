class Metrics:
    @staticmethod
    def mse(tr, exp, ignore=None):
        sm = 0
        ignore = ignore or set()
        for t, e in zip(tr, exp):
            if ignore and (t in ignore or e in ignore):
                continue
            sm += (t - e) ** 2
        res = sm / len(exp)
        return res if isinstance(res, float) else res[0]
