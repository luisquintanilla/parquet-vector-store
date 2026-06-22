#:package Parquet.Net@6.0.3
#:package System.Numerics.Tensors@10.0.9
#:property DynamicCodeSupport=true

// Parquet-backed brute-force vector search: the "no vector DB" pattern.
// Replicates Max Woolf's result (embeddings live in a Parquet
// file; top-k is a SIMD dot-product scan over a contiguous float matrix) in C#,
// using only Parquet.Net + System.Numerics.Tensors.TensorPrimitives. No engine,
// no database, no native dependency beyond the managed Parquet reader.
//
// Run:  dotnet run 01-bruteforce.cs                 (defaults: 50000 x 768)
//       dotnet run 01-bruteforce.cs -- 100000 1024  (rows dims)

using System.Diagnostics;
using System.Numerics.Tensors;
using Parquet.Serialization;

int rows = args.Length > 0 ? int.Parse(args[0]) : 50_000;
int dims = args.Length > 1 ? int.Parse(args[1]) : 768;
const int TopK = 10;
const int QueryReps = 200;
string path = $"embeddings_{rows}x{dims}.parquet";

if (!File.Exists(path))
{
    Console.WriteLine($"Generating {rows:N0} x {dims} embeddings -> {path} ...");
    await GenerateAsync(path, rows, dims);
}

long fileBytes = new FileInfo(path).Length;
Console.WriteLine($"Parquet file: {fileBytes / 1024.0 / 1024.0:F1} MB on disk");

// ---- Load: Parquet -> one contiguous, L2-normalized float[] matrix ----
long memBefore = GC.GetTotalMemory(true);
var swLoad = Stopwatch.StartNew();
var (matrix, ids, texts, n, d) = await LoadAsync(path);
swLoad.Stop();
long memAfter = GC.GetTotalMemory(false);

Console.WriteLine($"Loaded {n:N0} x {d} into contiguous float[] in {swLoad.ElapsedMilliseconds:N0} ms");
Console.WriteLine($"In-memory matrix: {(long)n * d * sizeof(float) / 1024.0 / 1024.0:F1} MB " +
                  $"(process delta ~{(memAfter - memBefore) / 1024.0 / 1024.0:F1} MB)");

// ---- Brute-force top-k cosine (== dot, vectors are normalized) ----
var rng = new Random(42);
var scores = new float[n];
var query = new float[d];
int cores = Environment.ProcessorCount;

// Warm up JIT / TensorPrimitives codegen on both paths.
matrix.AsSpan(0, d).CopyTo(query);
_ = Search(matrix, query, n, d, scores, TopK, parallel: false);
_ = Search(matrix, query, n, d, scores, TopK, parallel: true);

(int idx, float score)[] last = Array.Empty<(int, float)>();
double Bench(bool parallel)
{
    var rng2 = new Random(99);
    double total = 0;
    for (int q = 0; q < QueryReps; q++)
    {
        int seed = rng2.Next(n);
        matrix.AsSpan(seed * d, d).CopyTo(query);
        var sw = Stopwatch.StartNew();
        last = Search(matrix, query, n, d, scores, TopK, parallel);
        sw.Stop();
        total += sw.Elapsed.TotalMilliseconds;
    }
    return total / QueryReps;
}

double single = Bench(parallel: false);
double multi = Bench(parallel: true);
double bytesPerQuery = (long)n * d * sizeof(float);

Console.WriteLine();
Console.WriteLine($"Brute-force top-{TopK} over {n:N0} vectors ({cores} cores available):");
Console.WriteLine($"  single-thread: {single,7:F3} ms   {bytesPerQuery / single / 1e6:F1} GB/s   {n / (single / 1000.0) / 1e6:F1} M vec/s");
Console.WriteLine($"  parallel:      {multi,7:F3} ms   {bytesPerQuery / multi / 1e6:F1} GB/s   {n / (multi / 1000.0) / 1e6:F1} M vec/s");
Console.WriteLine($"  speedup:       {single / multi:F1}x");
Console.WriteLine();
Console.WriteLine($"Last query top-{TopK} (id : score):");
foreach (var (idx, score) in last)
    Console.WriteLine($"  {ids[idx],8} : {score:F4}   {Truncate(texts[idx], 48)}");

// ---------- helpers ----------

static (int idx, float score)[] Search(float[] matrix, float[] query, int n, int d, float[] scores, int k, bool parallel)
{
    if (parallel)
    {
        System.Threading.Tasks.Parallel.ForEach(
            System.Collections.Concurrent.Partitioner.Create(0, n),
            range =>
            {
                var q = (ReadOnlySpan<float>)query;
                for (int i = range.Item1; i < range.Item2; i++)
                    scores[i] = TensorPrimitives.Dot(matrix.AsSpan(i * d, d), q);
            });
    }
    else
    {
        var q = (ReadOnlySpan<float>)query;
        for (int i = 0; i < n; i++)
            scores[i] = TensorPrimitives.Dot(matrix.AsSpan(i * d, d), q);
    }

    // Top-k selection over the score vector (cheap relative to the scan).
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
    var matrix = new float[n * d];
    var ids = new int[n];
    var texts = new string[n];
    for (int i = 0; i < n; i++)
    {
        var rec = data[i];
        Normalize(rec.Embedding);
        rec.Embedding.CopyTo(matrix, i * d);
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
        RandomUnit(rng, e);
        records.Add(new EmbeddingRecord { Id = id, Text = $"doc-{id} sample passage", Embedding = e });
    }
    await ParquetSerializer.SerializeAsync(records, path);
}

static void RandomUnit(Random rng, float[] v)
{
    for (int i = 0; i < v.Length; i++)
        v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
    Normalize(v);
}

static void Normalize(float[] v)
{
    float norm = MathF.Sqrt(TensorPrimitives.Dot(v, v));
    if (norm > 1e-12f)
        TensorPrimitives.Divide(v, norm, v);
}

static string Truncate(string s, int n) => s.Length <= n ? s : s.Substring(0, n);

class EmbeddingRecord
{
    public int Id { get; set; }
    public string Text { get; set; } = "";
    public float[] Embedding { get; set; } = Array.Empty<float>();
}
