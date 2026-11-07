import os
import numpy as np
from tqdm import tqdm

SRC = "/home/christina/Documents/pillow/processed-datasets/w60/SLEEP-EDFX_harmonized/npz/sedf20"
DST = SRC + "_npy"
os.makedirs(DST, exist_ok=True)

for f in tqdm(sorted(os.listdir(SRC))):
    if not f.endswith(".npz"):
        continue
    name = f[:-4]
    try:
        data = np.load(os.path.join(SRC, f))
        x, y = data["x"], data["y"]
    except Exception as err:
        print(f"SKIPPED {f}: {err}")
        continue
    keep = y < 5
    np.save(os.path.join(DST, name + "_x.npy"), x[keep].astype(np.float32))
    np.save(os.path.join(DST, name + "_y.npy"), y[keep])