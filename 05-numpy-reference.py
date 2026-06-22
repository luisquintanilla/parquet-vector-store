"""Same-box BLAS reference for 05-blas-comparable.cs.

Reads the EXACT float32 matrix and query that the .NET sample dumped, runs Max Woolf's
fast_dot_product (numpy `query @ matrix.T`, which dispatches to BLAS / OpenBLAS), and times
it on the same machine. This removes the hardware confound: .NET vs numpy on identical data
and identical silicon.

Run 05-blas-comparable.cs first (it writes bench_*.f32), then:  python3 05-numpy-reference.py
"""
import sys, time, glob
import numpy as np

D = 768
mfile = sorted(glob.glob("bench_*x*.f32"))
if not mfile:
    sys.exit("No bench_*x*.f32 found. Run 05-blas-comparable.cs first.")
mfile = mfile[0]
N = int(mfile.split("_")[1].split("x")[0])
D = int(mfile.split("x")[1].split(".")[0])

m = np.fromfile(mfile, dtype=np.float32).reshape(N, D)
q = np.fromfile(f"bench_query_{D}.f32", dtype=np.float32)

def fast_dot_product(query, matrix, k=10):
    dots = query @ matrix.T
    idx = np.argpartition(dots, -k)[-k:]
    idx = idx[np.argsort(dots[idx])[::-1]]
    return idx, dots[idx]

for _ in range(5):
    fast_dot_product(q, m)

reps = 200
t = time.perf_counter()
for _ in range(reps):
    idx, score = fast_dot_product(q, m)
dt = (time.perf_counter() - t) / reps * 1000.0

gb = N * D * 4 / 1e9
print(f"numpy scenario: {N:,} x {D} float32, single query, top-10")
print(f"numpy @ (BLAS): {dt:.3f} ms/query | {gb/(dt/1000.0):.1f} GB/s | {N/(dt/1000.0)/1e6:.1f} M vec/s")

try:
    cfg = np.show_config(mode="dicts")
    blas = cfg.get("Build Dependencies", {}).get("blas", {})
    print(f"BLAS backend: {blas.get('name','?')} {blas.get('version','')}")
except Exception:
    pass

print("top-10 idx:", sorted(idx.tolist()))
