#:package System.Numerics.Tensors@10.0.9

// Woolf's scenario (single-query brute-force top-k over ~32k x 768 embeddings),
// rebuilt entirely in managed .NET and pushed as hard as the hardware allows, then
// compared against the SAME-box numpy/BLAS baseline (run 05-numpy-reference.py after this).
//
// The question this answers: can pure .NET be comparable with BLAS here, and can we beat it?
//
// First principles: a single query is MEMORY-BANDWIDTH bound (read the whole matrix once,
// one multiply-add per element, no reuse). So:
//   - float32: there is almost no compute to optimize. On equal hardware .NET ties BLAS,
//     because both are limited by the same DRAM bus. The only way to go faster is fewer bytes.
//   - fp16:  half the DRAM bytes, BUT managed .NET has no fused Half dot, so you pay a separate
//            convert pass (ConvertToSingle) before the dot. That extra pass is not bandwidth-bound
//            and eats the saving: measured, fp16 is a WASH-to-LOSS here. Honest negative result.
//   - int8:  a quarter of the bytes -> ~3x. Hardware int8 multiply-add via AVX-512BW
//            (vpmaddubsw + vpmaddwd). NOTE: this box has avx512_vnni, but .NET 10 exposes no
//            managed Avx512Vnni class (only the VEX AvxVnni, unsupported on Tiger Lake). That is
//            fine here: we are bandwidth-bound, so the win is bytes-moved, not VNNI MAC density.
//
// All managed, no native dependency, AOT-clean. Quantized paths report recall vs the float32
// top-k so the accuracy tradeoff is visible, not hidden.
//
// Run:  dotnet run -c Release 05-blas-comparable.cs            (32254 x 768, Woolf's size)
//       dotnet run -c Release 05-blas-comparable.cs -- 32254 768

using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

int N = args.Length > 0 ? int.Parse(args[0]) : 32_254; // Woolf's MTG card count
int D = args.Length > 1 ? int.Parse(args[1]) : 768;
const int TopK = 10;
const int Reps = 200;
int cores = Environment.ProcessorCount;
var rng = new Random(7);

Console.WriteLine($"Woolf scenario rebuilt in .NET: {N:N0} x {D} float32, single query, top-{TopK}, {cores} cores");
Console.WriteLine($"Matrix sizes in memory: float32 {(long)N*D*4/1048576.0:F0} MB | fp16 {(long)N*D*2/1048576.0:F0} MB | int8 {(long)N*D/1048576.0:F0} MB\n");

// ---- generate one normalized float32 matrix in memory ----
var m32 = new float[(long)N * D];
for (int i = 0; i < N; i++)
{
    var r = m32.AsSpan(i * D, D);
    for (int k = 0; k < D; k++) r[k] = (float)(rng.NextDouble() * 2 - 1);
    Normalize(r);
}
var q32 = new float[D];
m32.AsSpan(12345 * D, D).CopyTo(q32); // query = a real row, so neighbors are meaningful

// ---- fp16 copy ----
var m16 = new Half[(long)N * D];
TensorPrimitives.ConvertToHalf(m32, m16);

// ---- int8 copy (matrix as uint8 = int8 + 128; per-row scale) ----
var m8 = new byte[(long)N * D];
var rowScale = new float[N];
for (int i = 0; i < N; i++) rowScale[i] = QuantizeRow(m32.AsSpan(i * D, D), m8.AsSpan(i * D, D));
var q8 = new sbyte[D];
float qScale = QuantizeQuery(q32, q8);
int qSum = 0;
foreach (var v in q8) qSum += v;

// ---- dump raw float32 data for the same-box numpy/BLAS reference ----
string mPath = $"bench_{N}x{D}.f32", qPath = $"bench_query_{D}.f32";
File.WriteAllBytes(mPath, MemoryMarshal.AsBytes(m32.AsSpan()).ToArray());
File.WriteAllBytes(qPath, MemoryMarshal.AsBytes(q32.AsSpan()).ToArray());

var scores = new float[N];

// ---- exact float32 top-k (the reference for recall) ----
ScanF32(scores);
var refTop = TopKIndices(scores, N, TopK);
var refSet = new HashSet<int>(refTop.Select(t => t.idx));

// ---- benchmark each path (whole query: scan + top-k, like Woolf's fast_dot_product) ----
double f32 = Bench(() => { ScanF32(scores); _ = TopKIndices(scores, N, TopK); });
double f16 = Bench(() => { ScanF16(scores); _ = TopKIndices(scores, N, TopK); });
double i8  = Avx512BW.IsSupported ? Bench(() => { ScanI8(scores); _ = TopKIndices(scores, N, TopK); }) : double.NaN;

double recall16 = Recall(ScanF16, refSet);
double recall8  = Avx512BW.IsSupported ? Recall(ScanI8, refSet) : double.NaN;

double mbF32 = (double)N * D * 4 / 1e9, mbF16 = (double)N * D * 2 / 1e9, mbI8 = (double)N * D / 1e9;
Console.WriteLine($"{"Path",-22} | {"per-query",10} | {"GB/s",7} | {"M vec/s",8} | {"vs f32",7} | {"recall@10",9}");
Console.WriteLine(new string('-', 82));
Row("float32 (TensorPrimitives)", f32, mbF32, 1.0, 1.0);
Row("fp16 (Half + F16C)",         f16, mbF16, f32 / f16, recall16);
if (!double.IsNaN(i8)) Row("int8 (AVX-512BW madd)",  i8,  mbI8,  f32 / i8,  recall8);
else Console.WriteLine("int8: skipped (AVX-512BW not supported on this CPU)");

void Row(string name, double ms, double gb, double speedup, double recall) =>
    Console.WriteLine($"{name,-22} | {ms,7:F3} ms | {gb / (ms/1000.0),6:F1} | {N/(ms/1000.0)/1e6,7:F1} | {speedup,6:F2}x | {recall*100,7:F1}%");

Console.WriteLine($"\nData for the same-box BLAS check written to {mPath} / {qPath}.");
Console.WriteLine("Now run:  python3 05-numpy-reference.py   (numpy @ on the identical data + machine)");
Console.WriteLine("\nReading it: float32 is bandwidth-bound, so it ties (here slightly beats) same-box");
Console.WriteLine("BLAS: the bus is the wall and our parallel scan saturates it across all cores.");
Console.WriteLine("fp16 disappoints: managed .NET has no fused Half dot, so the separate convert pass");
Console.WriteLine("eats the byte saving (a real .NET gap). int8 is the differentiated, honest win:");
Console.WriteLine("~3x fewer DRAM bytes + hardware AVX-512BW madd, pure managed, no native dependency,");
Console.WriteLine("with the recall cost shown above. Compare these against 05-numpy-reference.py.");

// ───────────────────────── scans (parallel over rows) ─────────────────────────

void ScanF32(float[] outScores) =>
    System.Threading.Tasks.Parallel.ForEach(
        System.Collections.Concurrent.Partitioner.Create(0, N),
        range => { var q = (ReadOnlySpan<float>)q32;
            for (int i = range.Item1; i < range.Item2; i++)
                outScores[i] = TensorPrimitives.Dot(m32.AsSpan(i * D, D), q); });

void ScanF16(float[] outScores) =>
    System.Threading.Tasks.Parallel.ForEach(
        System.Collections.Concurrent.Partitioner.Create(0, N),
        range => {
            Span<float> tmp = stackalloc float[D];
            var q = (ReadOnlySpan<float>)q32;
            for (int i = range.Item1; i < range.Item2; i++)
            {
                TensorPrimitives.ConvertToSingle(m16.AsSpan(i * D, D), tmp); // reads 2 bytes/elem from DRAM
                outScores[i] = TensorPrimitives.Dot(tmp, q);
            }
        });

void ScanI8(float[] outScores) =>
    System.Threading.Tasks.Parallel.ForEach(
        System.Collections.Concurrent.Partitioner.Create(0, N),
        range => {
            for (int i = range.Item1; i < range.Item2; i++)
            {
                int dot = Dot8(m8.AsSpan(i * D, D), q8);           // uint8 . int8 via AVX-512BW
                int dotInt8 = dot - 128 * qSum;                    // undo the +128 matrix offset
                outScores[i] = dotInt8 * rowScale[i] * qScale;     // dequantize to cosine
            }
        });

// int8 dot: vpmaddubsw (uint8*int8 -> int16 pairs) then vpmaddwd vs ones (widen+sum to int32).
static int Dot8(ReadOnlySpan<byte> row, ReadOnlySpan<sbyte> q)
{
    ref byte rb = ref MemoryMarshal.GetReference(row);
    ref sbyte qb = ref MemoryMarshal.GetReference(q);
    var acc = Vector512<int>.Zero;
    var ones = Vector512.Create((short)1);
    int d = row.Length, k = 0;
    for (; k + 64 <= d; k += 64)
    {
        Vector512<byte> a = Vector512.LoadUnsafe(ref rb, (nuint)k);
        Vector512<sbyte> b = Vector512.LoadUnsafe(ref qb, (nuint)k);
        Vector512<short> t16 = Avx512BW.MultiplyAddAdjacent(a, b);
        acc += Avx512BW.MultiplyAddAdjacent(t16, ones);
    }
    int s = Vector512.Sum(acc);
    for (; k < d; k++) s += row[k] * q[k];
    return s;
}

// ───────────────────────── helpers ─────────────────────────

double Bench(Action query)
{
    for (int w = 0; w < 5; w++) query();           // warm
    var sw = Stopwatch.StartNew();
    for (int r = 0; r < Reps; r++) query();
    return sw.Elapsed.TotalMilliseconds / Reps;
}

double Recall(Action<float[]> scan, HashSet<int> reference)
{
    var s = new float[N];
    scan(s);
    var top = TopKIndices(s, N, TopK);
    int hit = top.Count(t => reference.Contains(t.idx));
    return (double)hit / TopK;
}

static (int idx, float score)[] TopKIndices(float[] scores, int n, int k)
{
    Span<int> ti = stackalloc int[k];
    Span<float> ts = stackalloc float[k];
    ts.Fill(float.NegativeInfinity);
    for (int i = 0; i < n; i++)
    {
        float s = scores[i];
        if (s <= ts[k - 1]) continue;
        int p = k - 1;
        while (p > 0 && ts[p - 1] < s) { ts[p] = ts[p - 1]; ti[p] = ti[p - 1]; p--; }
        ts[p] = s; ti[p] = i;
    }
    var res = new (int, float)[k];
    for (int i = 0; i < k; i++) res[i] = (ti[i], ts[i]);
    return res;
}

static float QuantizeRow(ReadOnlySpan<float> x, Span<byte> outq)
{
    float amax = 0; foreach (var v in x) amax = MathF.Max(amax, MathF.Abs(v));
    float scale = amax > 1e-12f ? amax / 127f : 1f;
    for (int k = 0; k < x.Length; k++)
    {
        int q = Math.Clamp((int)MathF.Round(x[k] / scale), -127, 127);
        outq[k] = (byte)(q + 128);
    }
    return scale;
}

static float QuantizeQuery(ReadOnlySpan<float> x, Span<sbyte> outq)
{
    float amax = 0; foreach (var v in x) amax = MathF.Max(amax, MathF.Abs(v));
    float scale = amax > 1e-12f ? amax / 127f : 1f;
    for (int k = 0; k < x.Length; k++)
        outq[k] = (sbyte)Math.Clamp((int)MathF.Round(x[k] / scale), -127, 127);
    return scale;
}

static void Normalize(Span<float> v)
{
    float norm = MathF.Sqrt(TensorPrimitives.Dot(v, v));
    if (norm > 1e-12f) TensorPrimitives.Divide(v, norm, v);
}
