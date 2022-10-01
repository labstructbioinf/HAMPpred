class Metrics:
    @staticmethod
    def mse(tr, exp):
        sm = 0
        for t, e in zip(tr, exp):
            sm += (t - e) ** 2
        res = sm / len(exp)
        return res if isinstance(res, float) else res[0]
