from json import JSONEncoder

import numpy as np


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)


def are_arrays_with_data(arrs):
    expr = [arr.any() if arr is not None else False for arr in arrs]
    return all(expr)
