# parquet-vector-store

A spike in a handful of files: **embeddings live in a Parquet file, top-k is a SIMD scan over a
contiguous float matrix**, and the scan generalizes to a **batched GEMM over `Tensor<float>`**
that .NET does not ship today. No vector database, no native dependency beyond a managed
Parquet reader. The "no vector DB" pattern from Max Woolf
([The Best Way to Use Text Embeddings Portably is With Parquet and Polars](https://minimaxir.com/2025/02/embeddings-parquet/)),
rebuilt in C# on .NET 10 with `System.Numerics.Tensors.TensorPrimitives`, shown as a drop-in
`Microsoft.Extensions.VectorData` (MEVD) collection, then pushed to the compute-bound regime.

## Why

Below roughly 100k vectors, an HNSW index plus a database server is often more moving parts
than the problem needs. Brute force over a flat file is exact (no recall cliff), trivial to
ship, and fast enough when the math is SIMD and the data is one contiguous array. The open
question this spike answers: can .NET do this with in-box primitives, and does it slot under
the abstraction .NET already ships (MEVD) so an app can start on a file and graduate to a real
index later without rewriting?

## Files

- **`01-bruteforce.cs`** is the perf proof. Generates N x D embeddings into Parquet, loads
  them into one normalized `float[]`, runs brute-force top-k cosine (single-thread vs
  parallel), prints latency, GB/s, and throughput.
- **`02-mevd-provider.cs`** is the drop-in. A complete `VectorStoreCollection<TKey, TRecord>`
  subclass backed by the same Parquet matrix, served through MEVD's `SearchAsync`, with a real
  write path (`UpsertAsync` / `DeleteAsync` / `EnsureCollection*`). App code touches only the
  abstraction. See [Write path](#write-path-02) for the Parquet-honest write model.
- **`03-batched-gemm.cs`** is the kernel proof. The batched GEMM that
  `System.Numerics.Tensors` does not ship, written against `Tensor<float>` / `TensorSpan<float>`
  (zero-copy). Synthetic in-memory data, no Parquet, so the kernel path is AOT-clean and the
  measurement is isolated from I/O. Shows the memory-bound to compute-bound crossover.
- **`04-parquet-tensor-gemm.cs`** is the cohesive showcase. 01 + 03 end to end on real loaded
  data: Parquet -> contiguous `float[]` -> zero-copy `Tensor<float>` -> batched GEMM search,
  returning real top-k neighbors.
- **`05-blas-comparable.cs`** + **`05-numpy-reference.py`** answer "can pure .NET match BLAS
  here?" Replicates Woolf's single-query scenario (32,254 x 768) in managed .NET at three
  precisions (float32, fp16, int8 via AVX-512BW), dumps the identical data, and compares against
  numpy (OpenBLAS) on the same machine.
- **`06-inmemory-comparable.cs`** answers "is this any faster than the vector stores .NET ships
  for local use?" Runs this repo's contiguous + parallel scan against the two MEVD stores the
  ecosystem ships for no-cloud use — `InMemoryCollection`
  (`Microsoft.SemanticKernel.Connectors.InMemory`) and `SqliteVec`
  (`Microsoft.SemanticKernel.Connectors.SqliteVec`, the AI Chat template's local default) — on the
  same 50k vectors in the same process. See [vs the stores .NET ships](#vs-stores-06).

## Run

```bash
# .NET 10 SDK required (file-based apps + the directives below)
dotnet run -c Release 01-bruteforce.cs            # defaults: 50000 x 768
dotnet run -c Release 01-bruteforce.cs -- 100000 1024
dotnet run -c Release 02-mevd-provider.cs         # reuses the parquet from step 1
dotnet run -c Release 03-batched-gemm.cs          # synthetic, AOT-clean kernel
dotnet run -c Release 04-parquet-tensor-gemm.cs   # end to end on the parquet vectors
dotnet run -c Release 05-blas-comparable.cs       # Woolf scenario, .NET vs BLAS, 3 precisions
python3 05-numpy-reference.py                     # same-box numpy/OpenBLAS reference (run 05 first)
dotnet run -c Release 06-inmemory-comparable.cs   # this repo vs MEVD InMemoryCollection + SqliteVec (reuses step 1 parquet)
```

## Measured: single-query scan (50,000 x 768 float32, ~147 MB, 16-core x86)

| Path | top-10 latency | throughput |
|---|---|---|
| single-thread `TensorPrimitives.Dot` | ~8.7 ms | ~17.6 GB/s |
| parallel (`Partitioner` + `Parallel.ForEach`) | ~3.7 ms | ~41.2 GB/s |

Through the MEVD `SearchAsync` abstraction: same neighbors, warm query ~5.8 ms.

<a id="write-path-02"></a>
## Write path (02): honest to Parquet, not OLTP

`02` is a complete MEVD collection: alongside read/search it implements `UpsertAsync`,
`DeleteAsync`, and `EnsureCollection*`. The write model is shaped by what Parquet *is* — a
columnar **bulk** format whose row groups are immutable — not papered over:

- **Inserts of new keys append a new row group** (`ParquetOptions { Append = true }`): cheap,
  O(batch), the idiomatic incremental Parquet write.
- **Updates of existing keys and deletes compact via a full rewrite**: a row inside an existing
  row group cannot be edited or removed in place, so the file is rewritten from the in-memory
  source of truth.
- **Mutations apply in memory immediately** (read-your-writes for `GetAsync`/`SearchAsync`) and
  **persist at flush boundaries**: call `FlushAsync` explicitly, or rely on the best-effort flush
  in `Dispose`. `FlushAsync` returns which mode it ran (`Append` vs `Rewrite`) so the trade-off is
  visible. The demo in `02` runs the round-trip on its own small file (never the 50k benchmark):

  ```
  EnsureCollectionExists -> file created
  Upsert 5 new keys + Flush -> Append (inserts append a row group)
  Update key 3 + Delete key 5 + Flush -> Rewrite (mutations force a rewrite)
  ```

This mirrors how real vector stores layer an LSM / segment-merge tier on top of immutable files;
`02` exposes the floor so the cost of a mutation is explicit. Two notes: a write invalidates the
packed search matrix, which is repacked lazily on the next search; and because `Dispose` is
synchronous, the auto-flush is pumped there (a production provider would prefer `IAsyncDisposable`).

**Writes are deliberately not benchmarked.** The perf comparison in this repo (Max Woolf / numpy)
is a single-query *read* scenario, so the write path is here for completeness and correctness of
the abstraction, not for a throughput number.

## How this compares to numpy / BLAS (05, same machine)

The pattern comes from Max Woolf ([post](https://minimaxir.com/2025/02/embeddings-parquet/)),
whose published number is a **single query** via numpy `query @ matrix.T` (which dispatches to
BLAS): 32,254 x 768 float32, top-3, **1.08 ms on an M3 Pro MacBook Pro**, i.e. ~90 GB/s. That is
a *different machine* (M3 Pro, LPDDR5 ~150 GB/s membw), so it is not a clean comparison to our
x86 box. 05 removes that confound: it runs the **same data through numpy/OpenBLAS and through
.NET on the same machine** (i9-11950H, AVX-512BW, ~51 GB/s DDR4).

Single query, 32,254 x 768, top-10, same box, same data:

| Impl | precision | per-query | GB/s | vs numpy | recall@10 |
|---|---|---|---|---|---|
| numpy `@` (OpenBLAS sgemv) | float32 | ~3.9 ms | ~25 | 1.00x | 100% |
| .NET parallel `TensorPrimitives.Dot` | float32 | ~2.7 ms | ~37 | **~1.45x faster** | 100% |
| .NET `Half` (convert + dot) | fp16 | ~4.3 ms | ~11 | ~0.9x (loss) | 100% |
| .NET AVX-512BW int8 madd | int8 | ~0.95 ms | ~25 (of int8) | **~4.1x faster** | ~90% |

The findings, stated honestly:

1. **float32 single query is bandwidth-bound, so .NET ties or slightly beats BLAS on this shape.**
   There is almost no compute to optimize; it is a memory-streaming contest. Our parallel scan
   saturates ~37 GB/s (about 72% of the box's ~51 GB/s ceiling) while OpenBLAS's single-query
   sgemv path reaches ~25 GB/s here, so .NET comes out ~1.45x ahead. The earlier "~2.2x slower
   than numpy" was the M3 Pro's faster *memory*, not numpy being smarter. On equal hardware the
   float32 story is a tie-to-win.
2. **int8 is the differentiated win: ~4.1x vs numpy float32 at ~90% recall@10.** Since the wall is
   bytes moved, quantizing to int8 reads a quarter of the DRAM and uses a hardware int8
   multiply-add (`Avx512BW.MultiplyAddAdjacent`, vpmaddubsw + vpmaddwd). Pure managed, no native
   dependency, with the recall cost measured, not hidden. This is the honest "beats numpy
   float32" result, and it is the kind of thing a Python dev would drop into C or Rust for.
3. **fp16 is a wash-to-loss in managed .NET today.** Storing as `Half` halves the DRAM bytes, but
   there is no fused `Half` dot, so you pay a separate `ConvertToSingle` pass that is not
   bandwidth-bound and eats the saving. Use int8, not fp16, until a fused half-precision dot
   exists. (A real .NET gap, noted below.)

Two .NET gaps this surfaced: (a) **no managed `Avx512Vnni`** class is exposed (only the VEX
`AvxVnni`, which is unsupported on Tiger Lake), so int8 uses AVX-512BW multiply-add; that is fine
here because the workload is bandwidth-bound, not compute-bound, so VNNI's extra MAC density would
not help. (b) **no fused `Half` dot** in `TensorPrimitives`, which is why fp16 underdelivers.

The 04 batched figure (Q=128 -> ~0.67 ms/query) is a *different question* from this single-query
comparison: batching is a lever both stacks exploit (numpy via BLAS GEMM), so do not read it as
beating numpy. The defensible single-query claim is the one measured here: **on equal hardware,
managed .NET ties-to-beats BLAS at float32 and beats float32-numpy ~4x at int8, with no native
dependency.**

<a name="vs-stores-06"></a>
## How this compares to the stores .NET ships (06): in-memory + SqliteVec

For local, no-cloud use the .NET AI ecosystem ships two MEVD stores, and `06` runs this repo
against **both** on the **same 50k x 768 vectors in the same process**, single-query top-10:

- **`InMemoryCollection`** (`Microsoft.SemanticKernel.Connectors.InMemory`) — the no-database,
  in-process store. Closest sibling to this repo: exact brute force over float vectors via
  `System.Numerics.Tensors`, records held as objects in a `ConcurrentDictionary`.
- **`SqliteVec`** (`Microsoft.SemanticKernel.Connectors.SqliteVec`) — the AI Chat Web template's
  actual *local* default. A single portable `.db` file with the bundled native `sqlite-vec`
  extension; vectors live on disk, search is brute force in C.

All three are **exact** brute force (no ANN index), so the neighbors are identical (10/10, same
top hit) — the comparison is layout, threading, and where the bytes live.

| Impl | layout | threads | ingest 50k | per-query | vs in-memory | top-10 agreement |
|---|---|---|---|---|---|---|
| MEVD `SqliteVec` | SQLite file (disk) | native | **~14 min** | ~110 ms | ~0.3x (slower) | 10/10 |
| MEVD `InMemoryCollection` | object dictionary (RAM) | 1 (LINQ) | ~45 ms | ~30 ms | 1.00x | - |
| this repo, single-thread | contiguous matrix (RAM) | 1 | (pack ~matrix) | ~16 ms | **~2x faster** | 10/10 |
| this repo, parallel | contiguous matrix (RAM) | 16 | (pack ~matrix) | ~7 ms | **~4-5x faster** | 10/10 |

Two stories sit in that table.

**vs `InMemoryCollection` (the RAM tier).** Same neighbors, ~2-5x faster, for four structural
reasons in order of impact:

1. **Contiguous vs scattered.** This repo packs every vector into one `float[]` matrix the scan
   streams linearly; `InMemoryCollection` holds each record as an object in a `ConcurrentDictionary`,
   so each vector is a separate heap `float[]` and the scan pointer-chases. Even single-threaded,
   the contiguous layout alone is ~2x.
2. **Parallel vs sequential.** This repo partitions across cores (`Parallel.ForEach`);
   `InMemoryCollection` iterates records with a sequential LINQ `Select`. That is the step to ~4-5x.
3. **Pre-normalized `Dot` vs per-query `CosineSimilarity`.** This repo L2-normalizes once at load,
   so each query is N x `TensorPrimitives.Dot`; `InMemoryCollection` recomputes the stored
   vector's norm every query.
4. **Partial top-k vs full sort.** This repo keeps a partial top-k (~O(N)); `InMemoryCollection`
   `OrderBy`-sorts all N scores (O(N log N)) then `Skip`/`Take`.

The honest trade-off is **memory**, not correctness: `InMemoryCollection` references the record
vectors in place and adds only ~9 MB of dictionary/wrapper overhead, while this repo allocates an
*additional* ~146 MB contiguous matrix. So this is "spend a packed copy to buy locality + a
streaming parallel scan."

**vs `SqliteVec` (the on-disk tier).** This is a different trade, not just a slower scan. SqliteVec
buys **durability and portability** (one file you can copy, back up, and reopen) and **SQL metadata
filtering** alongside the vector search — things the in-RAM stores do not give you. It pays for that
on **both** axes here: per-query is ~3-4x slower than `InMemoryCollection` and ~15x slower than this
repo's parallel scan (it crosses the managed→native boundary and parses SQL per query), and
**ingest is dramatically more expensive** — ~15 ms per vector, so building the 50k store took ~14
minutes versus ~45 ms for `InMemoryCollection` and a sub-second matrix pack here. `06` caches the
built `.db` so only the first run pays that cost. The takeaway is not "SqliteVec is bad" — it is the
right tool when the corpus outgrows RAM or you need a durable, filterable single file — but for a
fits-in-RAM corpus you are paying a large read- and write-latency tax for durability you may not
need yet. That is exactly the tier boundary the next section is about.

And neither in-box store changes the answer: this repo returns the same neighbors faster, which is
what `02` packages behind MEVD so an app can pick a tier without a rewrite.

## Where this leaves us: the story, the .NET gaps, and a plan

We started with "just Parquet + a dot product" and ended with a measured map of the **local vector
search tiers** in .NET. Here is the story it tells.

**The tiers (and when each is right).**

- **Tier 0 — in-memory `float[]` + dot product.** A few thousand to ~100k vectors, no database.
  Load once, scan with `TensorPrimitives`. This is Woolf's
  [Parquet + dot-product](https://minimaxir.com/2025/02/embeddings-parquet/) point, and `01`/`05`
  show .NET does it at numpy/BLAS-class speed (and ~4x that with int8). The in-box
  `InMemoryCollection` lives here too — and leaves ~2-5x on the table by not packing/parallelizing.
- **Tier 1 — a single-file, on-disk DB.** Vectors outgrow comfortable RAM, or you want durability,
  portability, and metadata filtering in one file, but you do not want to run a server. This is
  `sqlite-vec`, which Woolf calls out as the natural next step. `06` shows it works in .NET today
  via `SqliteVec` — with a real read- and write-latency cost.
- **Tier 2 — a dedicated vector DB.** Millions of vectors, ANN indexes, horizontal scale,
  concurrent writers (Qdrant, Azure AI Search, etc.). The AI Chat template wires these up too.

`Microsoft.Extensions.VectorData` (MEVD) is the **seam** that lets one app move across these tiers
without rewriting query code — and `02` shows you can implement that seam over nothing but a Parquet
file, no native dependency, with a Parquet-honest write path. That is the whole arc: the simplest
durable artifact (a Parquet file) can be a first-class, swappable vector store.

**What we showed, concretely.**

- A pure-managed contiguous + parallel scan **ties or beats numpy/OpenBLAS** at float32 and is
  **~4x faster at int8** on the same machine, single-query (`05`).
- It is **~2-5x faster than the in-box `InMemoryCollection`** and returns identical neighbors
  (`06`), and competitive-to-far-faster than native `SqliteVec` on reads.
- A complete **read + write MEVD provider over Parquet** (`02`), with honest bulk semantics
  (append for inserts, compaction for updates/deletes).
- The **single-query → batched-GEMM crossover** (`03`/`04`): batching turns a bandwidth-bound scan
  into a compute-bound GEMM and drops per-query latency ~10x — with a kernel .NET does not ship.

**The gaps this surfaced in .NET** (consolidated; see [Honest findings](#honest-findings) for the
per-file detail):

1. **No batched GEMM / GEMV in `System.Numerics.Tensors`.** `TensorPrimitives` has `Dot` and
   `CosineSimilarity` but no matmul (by charter), so the one operation that turns brute-force search
   compute-bound is hand-rolled here.
2. **No fused `Half` (fp16) dot.** fp16 should halve the bandwidth that bounds single-query search,
   but there is no fused half-precision reduction, so it pays a separate convert pass and the win
   evaporates.
3. **No managed AVX-512 VNNI surface.** int8 is the biggest measured win (~4x), reached via
   `Avx512BW.MultiplyAddAdjacent`; the dedicated VNNI path is not exposed in managed code.
4. **Row-by-row Parquet load.** `ParquetSerializer.DeserializeAsync<T>` materializes 50k POCOs by
   reflection (~6 s); there is no first-class columnar bulk-read straight into a contiguous buffer.
5. **`TensorSpan<T>` ergonomics under parallelism.** It is a `ref struct`, so it cannot be captured
   in a `Parallel` lambda — you must pass the `Tensor<T>` and re-acquire the span per partition.
6. **File-based apps disable dynamic code by default.** `dotnet run app.cs` sets
   `IsDynamicCodeSupported=false`, breaking `Expression.Compile()` paths until you set
   `#:property DynamicCodeSupport=true`.
7. **The in-box in-memory store leaves ~2-5x on the table.** Scattered object layout + sequential
   LINQ + per-query norm recompute + full sort, where a packed parallel partial-top-k scan is free.
8. **Connector portability rough edges.** `SqliteVec` accepts only `CosineDistance` (not
   `CosineSimilarity`), rejects key `0` (vec0 rowid), and has very slow ingest — papercuts an app
   moving across tiers hits.

**A plan to fill them.**

- **Runtime asks:** a fused `Half` dot reduction; an exposed managed VNNI path for int8; and a
  tiled GEMM/GEMV helper (or a documented row-stationary recipe) so the compute-bound regime is not
  hand-rolled per project.
- **Tensor ergonomics:** a parallel-friendly way to map a contiguous tensor's rows across cores
  without re-acquiring spans by hand.
- **Data path:** a columnar Parquet bulk-load that fills a contiguous `float[]`/`Tensor<float>`
  directly, skipping per-row POCO materialization.
- **A first-class Parquet-backed MEVD provider:** productize `02` (columnar load + the write path
  here + optional fp16/int8 quantization) so Tier 0/1 has a no-dependency, durable, swappable store.
- **A faster in-box brute force:** apply the packing + parallel partial-top-k here to
  `InMemoryCollection` (or ship it alongside) so the default store is not ~2-5x off.
- **Use this repo as the evidence:** every claim above is a runnable file with numbers on commodity
  hardware, so the asks are concrete and measured rather than aspirational.

The honest boundary: none of this replaces ANN/IVF at millions of vectors. It owns the
**small-to-mid, no-dependency floor** — and the seam to graduate upward when you outgrow it.

## Measured: batched GEMM over `Tensor<float>` (03, synthetic, AOT-clean kernel)

A single query (GEMV) is bandwidth-bound and cannot beat the scan above. Batching Q queries
into one GEMM reads the 147 MB matrix once and reuses each row across all Q queries, crossing
from memory-bound to compute-bound. Per-query latency drops ~10x and plateaus as compute takes
over (GFLOP/s climbs and flattens):

| Q (batch) | baseline/query (GEMV) | batched/query (GEMM) | speedup | GFLOP/s |
|---|---|---|---|---|
| 1 | ~4.5 ms | ~4.2 ms | 1.1x | ~18 |
| 8 | ~4.1 ms | ~0.73 ms | 5.6x | ~106 |
| 32 | ~4.3 ms | ~0.45 ms | 9.6x | ~172 |
| 128 | ~4.0 ms | ~0.43 ms | 9.4x | ~178 |

The ~178 GFLOP/s plateau is roughly 15-20% of fp32 peak: the headroom a tiled micro-kernel and
int8/fp16 would reclaim toward BLAS. See the next table for the same kernel end to end on real
Parquet vectors.

## Measured: end-to-end on real Parquet vectors (04)

Same batched GEMM, now on vectors loaded from the Parquet file and wrapped zero-copy as
`Tensor<float>`. The startup load (Parquet -> `Tensor<float>`) is a one-time ~7.6 s and is NOT
AOT-clean (`ParquetSerializer`); the steady-state search kernel below IS AOT-clean. The
crossover holds on the real data, a touch slower than 03 because it carries the load's cache/GC
pressure:

| Q (batch) | baseline/query (GEMV) | batched/query (GEMM) | speedup | GFLOP/s |
|---|---|---|---|---|
| 1 | ~5.4 ms | ~4.6 ms | 1.2x | ~17 |
| 8 | ~5.4 ms | ~0.96 ms | 5.7x | ~80 |
| 32 | ~5.1 ms | ~0.71 ms | 7.2x | ~109 |
| 128 | ~5.2 ms | ~0.67 ms | 7.8x | ~115 |

Returns correct top-k (query = vector #123 finds itself at cosine 1.0).

## Honest findings

1. **The wall is memory bandwidth, not compute.** Parallel hits ~41 GB/s and stops scaling
   (2.3x on 16 cores, not 16x) because each query streams the whole 147 MB matrix. The
   lever to sub-millisecond is fewer bytes: fp16 halves it, int8 quarters it. That is a
   numerics/quantization story, not a "need a database" story.
2. **The load path is the slow part, and it is fixable.** `ParquetSerializer.DeserializeAsync<T>`
   (row-by-row POCO reflection) takes ~6 s for 50k rows. A real provider should bulk-read the
   columns into the matrix directly, not materialize 50k objects. This is the concrete gap a
   first-class provider would close.
3. **File-based apps disable dynamic code by default.** `dotnet run app.cs` sets
   `IsDynamicCodeSupported=false`, so Parquet.Net's `Expression.Compile()` falls back to the
   interpreter and throws. Fix: `#:property DynamicCodeSupport=true` (used in both files).
4. **float32 random vectors do not compress**, so on-disk ~= in-memory here. Real embeddings
   compress somewhat; quantization is still the real size lever.
5. **The win above single-query needs a batched GEMM, which .NET does not ship.**
   `System.Numerics.Tensors` has `Tensor<T>` and `TensorPrimitives` but no matmul (by charter).
   A single query is bandwidth-bound, so the only way to change the physics is to batch and
   amortize the matrix read. The ~30-line row-stationary GEMM here does that, pure managed and
   AOT-clean, no native dependency. That is the differentiated claim here: BLAS-class
   batched throughput with no native dep. We are at ~15-20% of fp32 peak, so a tiled kernel and
   quantization are the next lever, not the abstraction.
6. **`Tensor<float>` is a usable surface for this, with one wrinkle.** `Tensor.Create(float[], [N,D])`
   wraps an existing buffer with no copy; `GetSpan([i,0], D)` returns a contiguous row as
   `ReadOnlySpan<float>` that feeds `TensorPrimitives.Dot` directly (no marshalling tax). The
   wrinkle: `TensorSpan<T>` is a ref struct, so it cannot be captured by a `Parallel` lambda;
   pass the `Tensor<float>` (a class) into each partition and re-acquire the span inside.
7. **The in-box in-memory store leaves ~2-5x on the table.** `InMemoryCollection`
   (`Microsoft.SemanticKernel.Connectors.InMemory`) holds records as objects in a
   `ConcurrentDictionary`, scans them with a sequential LINQ `Select`, recomputes each stored
   vector's norm per query, and `OrderBy`-sorts all N scores. Packing into one contiguous matrix,
   pre-normalizing, scanning in parallel, and keeping a partial top-k returns the *same* neighbors
   ~2-5x faster (`06`). The cost is an extra ~146 MB packed copy, not correctness.
8. **The on-disk tier (SqliteVec) is a real read- and write-latency trade, plus papercuts.**
   `SqliteVec` (`Microsoft.SemanticKernel.Connectors.SqliteVec`) buys durability, a single portable
   file, and SQL metadata filtering, but on this box per-query is ~3-4x slower than `InMemoryCollection`
   (~15x slower than this repo's parallel scan), and **ingest is ~15 ms/vector — ~14 minutes for
   50k** (`06` caches the built `.db`). Interop papercuts: it accepts only `CosineDistance` (not
   `CosineSimilarity`), and its `vec0` table rejects key `0` (rowid 0), so 0-based ids need a +1
   shift. Same neighbors throughout (10/10) — the cost is latency and ergonomics, not the answer.

## What this is not

Not an ANN index (no HNSW/IVF), not an OLTP row store (the write path is bulk: append for
inserts, compaction for updates/deletes), not a database, not a tuned BLAS (the GEMM is a
teaching/floor kernel, not MKL parity). It is the floor: the simplest thing that
works, measured honestly, expressed through the abstraction .NET ships, then pushed to the
compute-bound regime to show where a managed kernel changes the physics.

## License

MIT. See [`LICENSE`](LICENSE).
