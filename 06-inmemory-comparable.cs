#:package Microsoft.SemanticKernel.Connectors.InMemory@1.74.0-preview
#:package Microsoft.SemanticKernel.Connectors.SqliteVec@1.74.0-preview
#:package Parquet.Net@6.0.3
#:package System.Numerics.Tensors@10.0.9
#:property DynamicCodeSupport=true

// Is the Parquet-backed contiguous + parallel scan in this repo any faster than the
// vector stores .NET ships for local, no-cloud use? This measures THREE on the SAME data
// and SAME machine, single-query, like 05 does for numpy:
//
//   1. this repo            — Parquet → one contiguous, L2-normalized float[] + parallel Dot.
//   2. InMemoryCollection   — Microsoft.SemanticKernel.Connectors.InMemory: the canonical
//                             no-database, in-process MEVD store (records as objects in a dict).
//   3. SqliteVec            — Microsoft.SemanticKernel.Connectors.SqliteVec: the AI Chat Web
//                             template's actual LOCAL default. A single-file, on-disk SQLite DB
//                             with the native sqlite-vec extension (brute force in C, no ANN).
//
// All three are EXACT brute force over float vectors (no ANN index), so neighbor ids should
// agree; the comparison is about layout, threading, and where the bytes live.
//
// Why this repo and InMemoryCollection differ, structurally:
//   * Layout: this repo packs every vector into ONE contiguous, L2-normalized float[]
//     matrix; InMemoryCollection keeps each record as an object in a ConcurrentDictionary,
//     so each vector is a separate heap float[] (pointer chasing, poor locality).
//   * Metric: this repo pre-normalizes once at load, so each query is N x TensorPrimitives.Dot;
//     InMemoryCollection calls TensorPrimitives per record, recomputing the stored vector's
//     norm on every query.
//   * Parallelism: this repo partitions the scan across cores (Parallel.ForEach);
//     InMemoryCollection iterates records sequentially via LINQ Select (single thread).
//   * Top-k: this repo keeps a partial top-k (~O(N)); InMemoryCollection OrderBy-sorts all
//     N scores (O(N log N)) then Skip/Take.
//
// SqliteVec is a different tier: vectors live on DISK in a single .db file, so it trades raw
// in-RAM speed for durability, a single portable file, and SQL metadata filtering. It crosses
// the managed → native boundary and parses SQL per query, so expect it to be slower than the
// in-RAM scans but to "just work" without holding everything in memory.
//
// All three use DistanceFunction.CosineDistance (SqliteVec only supports CosineDistance, not
// CosineSimilarity; distance ascending == similarity descending, so ranking is identical).
//
// Run:  dotnet run 06-inmemory-comparable.cs        (reads embeddings_50000x768.parquet)

using System.Diagnostics;
using System.Numerics.Tensors;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using Microsoft.SemanticKernel.Connectors.SqliteVec;
using Parquet.Serialization;

string path = args.Length > 0 ? args[0] : "embeddings_50000x768.parquet";
if (!File.Exists(path))
{
    Console.WriteLine($"Missing {path}. Run `dotnet run 01-bruteforce.cs` first to generate it.");
    return;
}

const int Top = 10;
const int Iters = 25;
const int QueryRow = 2925;

// ── Load the shared data once ─────────────────────────────────────────────────
var loadSw = Stopwatch.StartNew();
var data = (await ParquetSerializer.DeserializeAsync<Doc>(path)).Data.ToArray();
loadSw.Stop();
int n = data.Length;
int d = data[0].Embedding.Length;
Console.WriteLine($"Loaded {n:N0} x {d} from {path} in {loadSw.Elapsed.TotalSeconds:F1} s ({n * (long)d * 4 / (1024 * 1024)} MB of float32)\n");

var query = data[QueryRow].Embedding;

// ── Build A: this repo — one contiguous, L2-normalized float[] matrix ──────────
long memBefore = GcBytes();
var matrix = new float[(long)n * d];
for (int i = 0; i < n; i++)
{
    var v = data[i].Embedding;
    float norm = MathF.Sqrt(TensorPrimitives.Dot(v, v));
    if (norm > 1e-12f) TensorPrimitives.Divide(v, norm, matrix.AsSpan(i * d, d));
    else v.CopyTo(matrix, i * d);
}
long matrixBytes = GcBytes() - memBefore;

// ── Build B: the MEVD in-memory store ─────────────────────────────────────────
memBefore = GcBytes();
var inMemory = new InMemoryCollection<int, Doc>("docs");
await inMemory.EnsureCollectionExistsAsync();
var buildSw = Stopwatch.StartNew();
await inMemory.UpsertAsync(data);
buildSw.Stop();
double inMemIngestMs = buildSw.Elapsed.TotalMilliseconds;
long inMemoryBytes = GcBytes() - memBefore;
Console.WriteLine($"Ingest: InMemoryCollection.UpsertAsync({n:N0}) took {inMemIngestMs:F0} ms");
Console.WriteLine($"Memory: both in-RAM stores keep the {n:N0} record objects (the ~{Mb((long)n * d * 4)} of vectors). On top of that,");
Console.WriteLine($"  InMemoryCollection adds ~{Mb(inMemoryBytes)} (dictionary + wrappers, vectors referenced in place);");
Console.WriteLine($"  this repo allocates an additional contiguous matrix ~{Mb(matrixBytes)} to buy locality + a parallel scan.\n");

// ── Build C: SqliteVec — single-file, on-disk, native sqlite-vec ──────────────
// Ingest is SLOW (~15 ms/vector here → ~14 min for 50k), so a fully-built db is CACHED on
// disk: the first run pays the write cost, later runs reuse vec-store-bench.db and skip ingest.
// A ".done" marker (written only after a complete ingest) guards against reusing a partial db.
// Delete vec-store-bench.db to force a fresh, timed ingest.
string dbPath = "vec-store-bench.db";
string donePath = dbPath + ".done";
bool cached = File.Exists(dbPath) && File.Exists(donePath);
if (!cached) { File.Delete(dbPath); File.Delete(donePath); }   // clean slate BEFORE opening (avoids pool/readonly race)

var sqlite = new SqliteCollection<int, Doc>($"Data Source={dbPath}", "docs");
await sqlite.EnsureCollectionExistsAsync();
double sqliteIngestMs;
if (cached)
{
    sqliteIngestMs = double.NaN;
    Console.WriteLine($"Ingest: SqliteVec — reusing cached {dbPath} ({Mb(new FileInfo(dbPath).Length)} on disk).");
    Console.WriteLine($"  (delete the file to force a fresh ingest; first build is ~15 ms/vector → ~14 min for {n:N0}.)\n");
}
else
{
    // sqlite-vec's vec0 virtual table reserves rowid 0, so a record with key 0 throws a UNIQUE
    // constraint. The parquet ids start at 0, so shift every key +1 for the SqliteVec path and
    // map back on read. (Embedding arrays are shared read-only; only the key changes.)
    var sqliteData = Array.ConvertAll(data, d => new Doc { Id = d.Id + 1, Text = d.Text, Embedding = d.Embedding });
    buildSw.Restart();
    // Chunk the upsert: SqliteVec binds every record in one statement, and 50k records
    // blows past SQLite's bound-parameter limit ("too many SQL variables").
    const int BatchSize = 200;
    for (int off = 0; off < n; off += BatchSize)
        await sqlite.UpsertAsync(sqliteData.Skip(off).Take(BatchSize));
    buildSw.Stop();
    sqliteIngestMs = buildSw.Elapsed.TotalMilliseconds;
    File.WriteAllText(donePath, "ok");          // mark the cache complete
    Console.WriteLine($"Ingest: SqliteVec.UpsertAsync({n:N0}) took {sqliteIngestMs:F0} ms → {Mb(new FileInfo(dbPath).Length)} on disk ({dbPath})");
    Console.WriteLine($"  (writes every vector to a SQLite table; vectors live on disk, not in the managed heap.)\n");
}

// ── Measure single-query top-10 (warm) ────────────────────────────────────────
var (idsParallel, msParallel) = await Measure("this repo: contiguous + parallel Dot", () => Task.FromResult(ScanParallel(matrix, n, d, query, Top)));
var (idsSingle, msSingle) = await Measure("this repo: contiguous single-thread Dot", () => Task.FromResult(ScanSingle(matrix, n, d, query, Top)));
var (idsInMem, msInMem) = await Measure("MEVD InMemoryCollection.SearchAsync", async () =>
{
    var ids = new List<int>(Top);
    await foreach (var r in inMemory.SearchAsync(query, Top))
        ids.Add(r.Record.Id);
    return ids.ToArray();
});
var (idsSqlite, msSqlite) = await Measure("MEVD SqliteVec.SearchAsync (on disk)", async () =>
{
    var ids = new List<int>(Top);
    await foreach (var r in sqlite.SearchAsync(query, Top))
        ids.Add(r.Record.Id - 1);   // undo the +1 key shift to compare against parquet ids
    return ids.ToArray();
});

string sqliteIngestCell = double.IsNaN(sqliteIngestMs) ? "cached" : $"{sqliteIngestMs / 1000,5:F0} s";
Console.WriteLine($"\nSingle-query top-{Top}, {n:N0} x {d}, same machine, warm:\n");
Console.WriteLine($"| Impl | layout | threads | ingest | per-query | vs in-memory |");
Console.WriteLine($"|---|---|---|---|---|---|");
Console.WriteLine($"| MEVD InMemoryCollection | object dict (RAM) | 1 (LINQ) | {inMemIngestMs,5:F0} ms | {msInMem,6:F2} ms | 1.00x |");
Console.WriteLine($"| MEVD SqliteVec | SQLite file (disk) | native | {sqliteIngestCell} | {msSqlite,6:F2} ms | {msInMem / msSqlite,4:F2}x |");
Console.WriteLine($"| this repo, single-thread | contiguous (RAM) | 1 | {"—",5} | {msSingle,6:F2} ms | {msInMem / msSingle,4:F2}x |");
Console.WriteLine($"| this repo, parallel | contiguous (RAM) | {Environment.ProcessorCount} | {"—",5} | {msParallel,6:F2} ms | {msInMem / msParallel,4:F2}x |");

Console.WriteLine($"\nNeighbor agreement (top-{Top} ids, all exact brute force):");
Console.WriteLine($"  in-memory vs parallel : {Overlap(idsInMem, idsParallel)}/{Top}   (top hit {idsInMem[0]} == {idsParallel[0]}: {idsInMem[0] == idsParallel[0]})");
Console.WriteLine($"  sqlitevec vs parallel : {Overlap(idsSqlite, idsParallel)}/{Top}   (top hit {idsSqlite[0]} == {idsParallel[0]}: {idsSqlite[0] == idsParallel[0]})");
Console.WriteLine($"  single    vs parallel : {Overlap(idsSingle, idsParallel)}/{Top}");

// Keep the on-disk benchmark database cached so later runs skip the slow ingest.
// (vec-store-bench.db is gitignored; delete it to force a fresh, timed ingest.)
// The parquet file is left untouched.
sqlite.Dispose();

// ── Helpers ───────────────────────────────────────────────────────────────────
static long GcBytes() => GC.GetTotalMemory(forceFullCollection: true);
static string Mb(long bytes) => $"{bytes / (1024.0 * 1024.0):F0} MB";
static int Overlap(int[] a, int[] b) => a.Count(b.Contains);

static async Task<(int[] ids, double ms)> Measure(string label, Func<Task<int[]>> run)
{
    var ids = await run();              // warm-up
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < Iters; i++) await run();
    sw.Stop();
    double ms = sw.Elapsed.TotalMilliseconds / Iters;
    Console.WriteLine($"  {label,-42} {ms,7:F2} ms/query");
    return (ids, ms);
}

static int[] ScanSingle(float[] matrix, int n, int d, float[] queryVec, int top)
{
    var q = Normalize(queryVec);
    var scores = new float[n];
    var qs = (ReadOnlySpan<float>)q;
    for (int i = 0; i < n; i++)
        scores[i] = TensorPrimitives.Dot(matrix.AsSpan(i * d, d), qs);
    return TopK(scores, top);
}

static int[] ScanParallel(float[] matrix, int n, int d, float[] queryVec, int top)
{
    var q = Normalize(queryVec);
    var scores = new float[n];
    Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, n), range =>
    {
        var qs = (ReadOnlySpan<float>)q;
        for (int i = range.Item1; i < range.Item2; i++)
            scores[i] = TensorPrimitives.Dot(matrix.AsSpan(i * d, d), qs);
    });
    return TopK(scores, top);
}

static float[] Normalize(float[] v)
{
    var copy = v.ToArray();
    float norm = MathF.Sqrt(TensorPrimitives.Dot(copy, copy));
    if (norm > 1e-12f) TensorPrimitives.Divide(copy, norm, copy);
    return copy;
}

static int[] TopK(float[] scores, int k)
{
    k = Math.Min(k, scores.Length);
    var idx = new int[k];
    var val = new float[k];
    Array.Fill(val, float.NegativeInfinity);
    for (int i = 0; i < scores.Length; i++)
    {
        float s = scores[i];
        if (s <= val[k - 1]) continue;
        int p = k - 1;
        while (p > 0 && val[p - 1] < s) { val[p] = val[p - 1]; idx[p] = idx[p - 1]; p--; }
        val[p] = s; idx[p] = i;
    }
    return idx;
}

public sealed class Doc
{
    [VectorStoreKey]
    public int Id { get; set; }

    [VectorStoreData]
    public string Text { get; set; } = "";

    [VectorStoreVector(768, DistanceFunction = DistanceFunction.CosineDistance)]
    public float[] Embedding { get; set; } = Array.Empty<float>();
}
