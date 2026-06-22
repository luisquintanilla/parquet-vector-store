#:package System.Numerics.Tensors@10.0.9

// Batched matrix multiply (GEMM) expressed as an operation over Tensor<float>,
// the thing System.Numerics.Tensors does NOT ship today. The point of 03 is two-fold:
//
//   1. Perf: a single query (GEMV) is memory-bandwidth bound, so it cannot beat
//      01. Batching Q queries into a GEMM reads the matrix ONCE and reuses each
//      row across all Q queries, crossing from memory-bound to compute-bound.
//      That is the only place a matmul changes the physics.
//
//   2. Usability: the operation is written against Tensor<float> / TensorSpan<float>
//      (zero-copy over a flat buffer), so this also answers "is Tensor<T> a usable
//      surface for this workload, or does it fight us?" Findings are noted inline.
//
// Pure managed, no native dependency, no dynamic-code directive (AOT-clean path):
// the matrix is generated in memory so there is no Parquet serializer here.
//
// Run:  dotnet run -c Release 03-batched-gemm.cs            (50000 x 768)
//       dotnet run -c Release 03-batched-gemm.cs -- 50000 768

using System.Diagnostics;
using System.Numerics.Tensors;

int N = args.Length > 0 ? int.Parse(args[0]) : 50_000;
int D = args.Length > 1 ? int.Parse(args[1]) : 768;
int[] batches = { 1, 8, 32, 128 };
int cores = Environment.ProcessorCount;
var rng = new Random(7);

Console.WriteLine($"Generating {N:N0} x {D} normalized vectors in memory (AOT-clean, no Parquet)...");
var buf = new float[N * D];
for (int i = 0; i < N; i++)
{
    var row = buf.AsSpan(i * D, D);
    for (int k = 0; k < D; k++) row[k] = (float)(rng.NextDouble() * 2 - 1);
    Normalize(row);
}

// Zero-copy: wrap the existing flat float[] as a Tensor<float> of shape [N, D].
Tensor<float> matrix = Tensor.Create(buf, [(nint)N, (nint)D]);

Console.WriteLine($"Matrix: {(long)N * D * 4 / 1024.0 / 1024.0:F0} MB in one contiguous buffer, {cores} cores\n");
Console.WriteLine("Baseline = run Q queries one at a time (reads the matrix Q times).");
Console.WriteLine("Batched  = one GEMM over Tensor<float> (reads the matrix once).\n");
Console.WriteLine($"{"Q",4} | {"baseline/query",15} | {"batched/query",14} | {"speedup",8} | {"GFLOP/s",9} | {"eff GB/s",9}");
Console.WriteLine(new string('-', 78));

foreach (int Q in batches)
{
    var qbuf = new float[Q * D];
    for (int j = 0; j < Q; j++)
        buf.AsSpan(rng.Next(N) * D, D).CopyTo(qbuf.AsSpan(j * D, D)); // real rows -> meaningful cosine
    Tensor<float> queries = Tensor.Create(qbuf, [(nint)Q, (nint)D]);
    var scores = new float[N * Q];

    RepeatedGemv(matrix, queries, scores, N, D, Q); // warm up both paths
    BatchedMatMul(matrix, queries, scores, N, D, Q);

    const int reps = 3;
    var sw = Stopwatch.StartNew();
    for (int r = 0; r < reps; r++) RepeatedGemv(matrix, queries, scores, N, D, Q);
    double baseMs = sw.Elapsed.TotalMilliseconds / reps;

    sw.Restart();
    for (int r = 0; r < reps; r++) BatchedMatMul(matrix, queries, scores, N, D, Q);
    double batMs = sw.Elapsed.TotalMilliseconds / reps;

    double gflops = 2.0 * N * D * Q / (batMs / 1000.0) / 1e9;
    double effGB = (double)N * D * 4 / (batMs / 1000.0) / 1e9; // matrix streamed once
    Console.WriteLine($"{Q,4} | {baseMs / Q,12:F3} ms | {batMs / Q,11:F3} ms | {baseMs / batMs,6:F1}x | {gflops,8:F0} | {effGB,8:F1}");
}

// Correctness: batched scores must match a direct dot product.
{
    int Q = 4;
    var qbuf = new float[Q * D];
    for (int j = 0; j < Q; j++) buf.AsSpan(j * D, D).CopyTo(qbuf.AsSpan(j * D, D));
    var queries = Tensor.Create(qbuf, [(nint)Q, (nint)D]);
    var scores = new float[N * Q];
    BatchedMatMul(matrix, queries, scores, N, D, Q);
    float direct = TensorPrimitives.Dot(buf.AsSpan(123 * D, D), qbuf.AsSpan(2 * D, D));
    float batched = scores[123 * Q + 2];
    Console.WriteLine($"\nCorrectness: batched[123,2]={batched:F6} vs direct Dot={direct:F6} -> " +
                      (MathF.Abs(batched - direct) < 1e-4f ? "match" : "MISMATCH"));
}

Console.WriteLine("\nReading it: Q=1 is memory-bandwidth bound (a matmul cannot beat 01). Batching");
Console.WriteLine("amortizes the single matrix read across the batch and crosses into compute-bound");
Console.WriteLine("(GFLOP/s climbs and plateaus), cutting per-query latency ~10x. All on the");
Console.WriteLine("Tensor<float> surface, pure managed, no native dependency. The plateau is the");
Console.WriteLine("headroom a tuned micro-kernel (and int8/fp16) would reclaim toward BLAS peak.");

// ── The GEMM op, written against Tensor<float> ───────────────────────────────
//
// Usability findings (the answer to "is Tensor<T> usable here?"):
//  - Tensor.Create(float[], [N,D]) wraps an existing buffer with no copy. Good.
//  - TensorSpan<T> is a ref struct, so it cannot be captured by a Parallel lambda.
//    We pass the Tensor<float> (a class) into each partition and re-acquire the
//    span inside. That works, but it is the one wrinkle worth calling out.
//  - ReadOnlyTensorSpan.GetSpan([i,0], D) returns a contiguous row as
//    ReadOnlySpan<float> with no copy, which feeds TensorPrimitives.Dot directly.
//    So the hot path stays on the Tensor surface with no marshalling tax.

static void BatchedMatMul(Tensor<float> matrix, Tensor<float> queries, float[] scores, int n, int d, int q)
{
    var qbuf = new float[q * d];
    queries.FlattenTo(qbuf); // small operand; flatten once so the inner loop uses raw spans
    System.Threading.Tasks.Parallel.ForEach(
        System.Collections.Concurrent.Partitioner.Create(0, n),
        range =>
        {
            var m = matrix.AsReadOnlyTensorSpan(); // re-acquired per partition (ref struct)
            for (int i = range.Item1; i < range.Item2; i++)
            {
                ReadOnlySpan<float> row = m.GetSpan([(nint)i, (nint)0], d); // zero-copy row
                int baseIdx = i * q;
                for (int j = 0; j < q; j++)
                    scores[baseIdx + j] = TensorPrimitives.Dot(row, qbuf.AsSpan(j * d, d));
            }
        });
}

// Baseline: the same kernel, but one query at a time. Streams the matrix q times.
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
                    scores[i * q + jj] = TensorPrimitives.Dot(m.GetSpan([(nint)i, (nint)0], d), qj);
            });
    }
}

static void Normalize(Span<float> v)
{
    float norm = MathF.Sqrt(TensorPrimitives.Dot(v, v));
    if (norm > 1e-12f) TensorPrimitives.Divide(v, norm, v);
}
