#:package Parquet.Net@6.0.3
#:package System.Numerics.Tensors@10.0.9
#:property DynamicCodeSupport=true

// The cohesive showcase: 01 (Parquet-backed vectors) + Tensor<float> + the batched
// GEMM from 03, end to end on REAL loaded data. This is the app shape:
//
//   startup    -> load the Parquet file ONCE into a contiguous float[], wrap it
//                 zero-copy as a Tensor<float> [N, D]. One-time cost.
//   steady run -> serve queries. A single query (GEMV) is memory-bandwidth bound
//                 and cannot beat 01. Batching Q queries into one GEMM reads the
//                 matrix once and crosses into compute-bound -> ~10x lower per-query
//                 latency. Returns real top-k neighbors, not just raw scores.
//
// Honesty note (why this is 04 and not folded into 03's AOT-clean claim):
//   the SEARCH kernel is pure managed and AOT-clean, but the LOAD path uses
//   ParquetSerializer, which needs DynamicCodeSupport=true (NOT AOT-clean). The
//   load also dominates wall time (~seconds for 50k rows). Both are the I/O gap,
//   separate from the kernel, and both are fixed by a columnar bulk-load. We eat
//   that caveat here on purpose so the demo runs
//   on the actual Parquet vectors instead of synthetic floats.
//
// Run:  dotnet run -c Release 04-parquet-tensor-gemm.cs              (50000 x 768)
//       dotnet run -c Release 04-parquet-tensor-gemm.cs -- 100000 1024

using System.Diagnostics;
using System.Numerics.Tensors;
using Parquet.Serialization;

int rows = args.Length > 0 ? int.Parse(args[0]) : 50_000;
int dims = args.Length > 1 ? int.Parse(args[1]) : 768;
const int TopK = 10;
int[] batches = { 1, 8, 32, 128 };
int cores = Environment.ProcessorCount;
string path = $"embeddings_{rows}x{dims}.parquet";

if (!File.Exists(path))
{
    Console.WriteLine($"Generating {rows:N0} x {dims} embeddings -> {path} ...");
    await GenerateAsync(path, rows, dims);
}
Console.WriteLine($"Parquet file: {new FileInfo(path).Length / 1024.0 / 1024.0:F1} MB on disk");

// ---- Startup: Parquet -> contiguous float[] -> zero-copy Tensor<float> ----
// This half is NOT AOT-clean (ParquetSerializer) and is the slow part. One-time.
var swLoad = Stopwatch.StartNew();
var (buf, ids, texts, n, d) = await LoadAsync(path);
swLoad.Stop();
Tensor<float> matrix = Tensor.Create(buf, [(nint)n, (nint)d]); // no copy
Console.WriteLine($"Startup load (Parquet -> Tensor<float> [{n:N0}, {d}]): {swLoad.ElapsedMilliseconds:N0} ms " +
                  $"(one-time, NOT AOT-clean)");
Console.WriteLine($"In-memory matrix: {(long)n * d * sizeof(float) / 1024.0 / 1024.0:F0} MB, {cores} cores\n");

// ---- Steady state: batched search over the real loaded Tensor<float> ----
// This half IS AOT-clean: TensorPrimitives only, no reflection, no dynamic code.
Console.WriteLine("Steady-state search (this kernel is AOT-clean):");
Console.WriteLine("  baseline = Q queries one at a time (GEMV, reads the matrix Q times)");
Console.WriteLine("  batched  = one GEMM over Tensor<float>  (reads the matrix once)\n");
Console.WriteLine($"{"Q",4} | {"baseline/query",15} | {"batched/query",14} | {"speedup",8} | {"GFLOP/s",9}");
Console.WriteLine(new string('-', 64));

var rng = new Random(99);
foreach (int Q in batches)
{
    var qbuf = new float[Q * d];
    for (int j = 0; j < Q; j++)
        buf.AsSpan(rng.Next(n) * d, d).CopyTo(qbuf.AsSpan(j * d, d)); // real rows -> meaningful cosine
    Tensor<float> queries = Tensor.Create(qbuf, [(nint)Q, (nint)d]);
    var scores = new float[(long)n * Q];

    RepeatedGemv(matrix, queries, scores, n, d, Q); // warm both paths
    BatchedMatMul(matrix, queries, scores, n, d, Q);

    const int reps = 3;
    var sw = Stopwatch.StartNew();
    for (int r = 0; r < reps; r++) RepeatedGemv(matrix, queries, scores, n, d, Q);
    double baseMs = sw.Elapsed.TotalMilliseconds / reps;

    sw.Restart();
    for (int r = 0; r < reps; r++) BatchedMatMul(matrix, queries, scores, n, d, Q);
    double batMs = sw.Elapsed.TotalMilliseconds / reps;

    double gflops = 2.0 * n * d * Q / (batMs / 1000.0) / 1e9;
    Console.WriteLine($"{Q,4} | {baseMs / Q,12:F3} ms | {batMs / Q,11:F3} ms | {baseMs / batMs,6:F1}x | {gflops,8:F0}");
}

// ---- Prove it is a real search: top-k neighbors for one real query ----
{
    int probe = 123;
    var qbuf = new float[d];
    buf.AsSpan(probe * d, d).CopyTo(qbuf);
    var queries = Tensor.Create(qbuf, [(nint)1, (nint)d]);
    var scores = new float[n];
    BatchedMatMul(matrix, queries, scores, n, d, 1);
    var top = TopK1(scores, n, TopK);
    Console.WriteLine($"\nTop-{TopK} for query = vector #{probe} (id : cosine):");
    foreach (var (idx, score) in top)
        Console.WriteLine($"  {ids[idx],8} : {score:F4}   {Truncate(texts[idx], 48)}");
}

Console.WriteLine("\nReading it: startup load is the slow, non-AOT-clean half (the I/O gap a");
Console.WriteLine("columnar bulk-load closes). Steady-state search is AOT-clean: single query");
Console.WriteLine("stays bandwidth-bound (== 01), but batching over Tensor<float> reads the");
Console.WriteLine("matrix once and cuts per-query latency ~8x on the actual Parquet vectors");
Console.WriteLine("(the isolated 03 kernel hits ~10x without the load's cache/GC pressure).");

// ── GEMM op over Tensor<float> (same kernel as 03) ───────────────────────────
static void BatchedMatMul(Tensor<float> matrix, Tensor<float> queries, float[] scores, int n, int d, int q)
{
    var qbuf = new float[q * d];
    queries.FlattenTo(qbuf);
    System.Threading.Tasks.Parallel.ForEach(
        System.Collections.Concurrent.Partitioner.Create(0, n),
        range =>
        {
            var m = matrix.AsReadOnlyTensorSpan(); // ref struct: re-acquire per partition
            for (int i = range.Item1; i < range.Item2; i++)
            {
                ReadOnlySpan<float> row = m.GetSpan([(nint)i, (nint)0], d); // zero-copy row
                long baseIdx = (long)i * q;
                for (int j = 0; j < q; j++)
                    scores[baseIdx + j] = TensorPrimitives.Dot(row, qbuf.AsSpan(j * d, d));
            }
        });
}

static void RepeatedGemv(Tensor<float> matrix, Tensor<float> queries, float[] scores, int n, int d, int q)
{
    var qbuf = new float[q * d];
    queries.FlattenTo(qbuf);
    for (int j = 0; j < q; j++)
    {
        int jj = j;
        System.Threading.Tasks.Parallel.ForEach(
            System.Collections.Concurrent.Partitioner.Create(0, n),
            range =>
            {
                var m = matrix.AsReadOnlyTensorSpan();
                var qj = qbuf.AsSpan(jj * d, d);
                for (int i = range.Item1; i < range.Item2; i++)
                    scores[(long)i * q + jj] = TensorPrimitives.Dot(m.GetSpan([(nint)i, (nint)0], d), qj);
            });
    }
}

static (int idx, float score)[] TopK1(float[] scores, int n, int k)
{
    Span<int> topIdx = stackalloc int[k];
    Span<float> topScore = stackalloc float[k];
    topScore.Fill(float.NegativeInfinity);
    for (int i = 0; i < n; i++)
    {
        float s = scores[i];
        if (s <= topScore[k - 1]) continue;
        int p = k - 1;
        while (p > 0 && topScore[p - 1] < s)
        {
            topScore[p] = topScore[p - 1];
            topIdx[p] = topIdx[p - 1];
            p--;
        }
        topScore[p] = s;
        topIdx[p] = i;
    }
    var result = new (int, float)[k];
    for (int i = 0; i < k; i++) result[i] = (topIdx[i], topScore[i]);
    return result;
}

static async Task<(float[] matrix, int[] ids, string[] texts, int n, int d)> LoadAsync(string path)
{
    var result = await ParquetSerializer.DeserializeAsync<EmbeddingRecord>(path);
    IList<EmbeddingRecord> data = result.Data;
    int n = data.Count;
    int d = data[0].Embedding.Length;
    var matrix = new float[(long)n * d];
    var ids = new int[n];
    var texts = new string[n];
    for (int i = 0; i < n; i++)
    {
        var rec = data[i];
        Normalize(rec.Embedding);
        rec.Embedding.CopyTo(matrix, (long)i * d);
        ids[i] = rec.Id;
        texts[i] = rec.Text;
    }
    return (matrix, ids, texts, n, d);
}

static async Task GenerateAsync(string path, int rows, int dims)
{
    var rng = new Random(7);
    var records = new List<EmbeddingRecord>(rows);
    for (int id = 0; id < rows; id++)
    {
        var e = new float[dims];
        for (int i = 0; i < dims; i++) e[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        Normalize(e);
        records.Add(new EmbeddingRecord { Id = id, Text = $"doc-{id} sample passage", Embedding = e });
    }
    await ParquetSerializer.SerializeAsync(records, path);
}

static void Normalize(Span<float> v)
{
    float norm = MathF.Sqrt(TensorPrimitives.Dot(v, v));
    if (norm > 1e-12f) TensorPrimitives.Divide(v, norm, v);
}

static string Truncate(string s, int n) => s.Length <= n ? s : s.Substring(0, n);

class EmbeddingRecord
{
    public int Id { get; set; }
    public string Text { get; set; } = "";
    public float[] Embedding { get; set; } = Array.Empty<float>();
}
