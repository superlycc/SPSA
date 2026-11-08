"""Connected components of the detection-to-identity support graph."""

import numpy as np


def components(allowed):
    remaining = set(range(allowed.shape[0]))
    while remaining:
        ds, ts = {remaining.pop()}, set()
        while True:
            new_ts = set(np.where(allowed[list(ds)].any(axis=0))[0])
            new_ds = set(np.where(allowed[:, list(new_ts)].any(axis=1))[0]) if new_ts else ds
            if new_ds == ds and new_ts == ts:
                break
            ds, ts = new_ds, new_ts
        remaining -= ds
        yield sorted(ds), sorted(ts)
