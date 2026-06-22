#:package Microsoft.Extensions.VectorData.Abstractions@10.7.0
#:package Parquet.Net@6.0.3
#:package System.Numerics.Tensors@10.0.9
#:property DynamicCodeSupport=true

// A complete Microsoft.Extensions.VectorData (MEVD) collection backed by a Parquet
// file: brute-force top-k cosine read (the same parallel TensorPrimitives scan as
// 01-bruteforce.cs) PLUS a real write path (Upsert / Delete / EnsureCollection*).
// The point: an app written against the MEVD abstraction can run on a flat Parquet
// file with no database and no native dependency, then swap to a managed HNSW or a
// hosted store later by changing one line, not the app.
//
// Write model — honest to Parquet, which is a columnar BULK format, not an OLTP row
// store. Row groups are immutable, so:
//   * inserts of NEW keys      -> append a new row group   (cheap, O(batch))
//   * updates of existing keys -> compaction (full rewrite) because the on-disk row
//   * deletes                  -> compaction (full rewrite)   cannot be edited in place
// Mutations apply in memory immediately (read-your-writes for Get/Search) and persist
// at flush boundaries: call FlushAsync explicitly, or rely on the best-effort flush in
// Dispose. This mirrors how real vector stores layer an LSM/segment-merge tier on top
// of immutable files; here we expose the floor so the trade-off is visible, not hidden.
//
// Writes are deliberately NOT benchmarked: the perf comparison in this repo (Max Woolf /
// numpy) is a single-query READ scenario. The write path is here for completeness and
// correctness of the abstraction, not for a throughput number.
//
// Run:  dotnet run 02-mevd-provider.cs           (reads embeddings_50000x768.parquet
//                                                  produced by 01-bruteforce.cs, then
//                                                  runs a write round-trip on its own file)

using System.Diagnostics;
using System.Numerics.Tensors;
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.VectorData;
using Parquet;
using Parquet.Serialization;

// ── Part 1: read / search on the 50k benchmark file (unchanged) ───────────────
string path = args.Length > 0 ? args[0] : "embeddings_50000x768.parquet";
if (File.Exists(path))
{
    VectorStoreCollection<int, Doc> collection =
        new ParquetVectorStoreCollection<int, Doc>("docs", path);

    Console.WriteLine($"CollectionExists: {await collection.CollectionExistsAsync()}");

    // Use a real stored vector as the query so neighbors are meaningful.
    Doc seed = (await collection.GetAsync(2925))!;
    ReadOnlyMemory<float> query = seed.Embedding;

    var sw = Stopwatch.StartNew();
    var hits = new List<VectorSearchResult<Doc>>();
    await foreach (var r in collection.SearchAsync(query, top: 5))
        hits.Add(r);
    sw.Stop();

    Console.WriteLine($"\nSearchAsync top-5 in {sw.Elapsed.TotalMilliseconds:F2} ms (cold scan; store loaded on first access above):");
    foreach (var hit in hits)
        Console.WriteLine($"  {hit.Record.Id,8} : {hit.Score:F4}   {hit.Record.Text}");

    // Warm query (store already loaded) to show steady-state latency.
    sw.Restart();
    await foreach (var _ in collection.SearchAsync(query, top: 5)) { }
    sw.Stop();
    Console.WriteLine($"\nWarm SearchAsync top-5: {sw.Elapsed.TotalMilliseconds:F2} ms");
}
else
{
    Console.WriteLine($"(Skipping read demo: {path} not found. Run `dotnet run 01-bruteforce.cs` first.)");
}

// ── Part 2: write round-trip on a small, separate file (not the benchmark file) ─
await WriteDemoAsync();

static async Task WriteDemoAsync()
{
    const string demoPath = "mevd-write-demo.parquet";
    Console.WriteLine("\n────────────────────────────────────────────────────────────");
    Console.WriteLine("Write round-trip demo (Parquet append vs compaction)");
    Console.WriteLine("────────────────────────────────────────────────────────────");

    // Fresh start: a brand-new, empty collection.
    var store = new ParquetVectorStoreCollection<int, Doc>("docs-write", demoPath);
    await store.EnsureCollectionDeletedAsync();
    await store.EnsureCollectionExistsAsync();
    Console.WriteLine($"EnsureCollectionExists -> file created, CollectionExists={await store.CollectionExistsAsync()}");

    // Insert a small batch of new keys -> should persist as an APPENDED row group.
    var batch = Enumerable.Range(1, 5).Select(i => MakeDoc(i, $"doc-{i}", seed: i)).ToList();
    await store.UpsertAsync(batch);
    var mode = await store.FlushAsync();
    Console.WriteLine($"Upsert 5 new keys + Flush -> {mode} (inserts append a row group)");

    // Reload from a fresh instance to prove durability + read-your-writes across instances.
    var reload = new ParquetVectorStoreCollection<int, Doc>("docs-write", demoPath);
    var got = await reload.GetAsync(3);
    var neighbors = new List<VectorSearchResult<Doc>>();
    await foreach (var r in reload.SearchAsync(got!.Embedding, top: 3))
        neighbors.Add(r);
    Console.WriteLine($"Reload -> Get(3)='{got.Text}', Search top-3 ids=[{string.Join(",", neighbors.Select(n => n.Record.Id))}]");

    // Update an existing key and delete another -> on-disk rows are now stale,
    // so the next flush must COMPACT (full rewrite), not append.
    await reload.UpsertAsync(MakeDoc(3, "doc-3-UPDATED", seed: 99));
    await reload.DeleteAsync(5);
    mode = await reload.FlushAsync();
    Console.WriteLine($"Update key 3 + Delete key 5 + Flush -> {mode} (mutations force a rewrite)");

    // Reload again and confirm the update + delete persisted.
    var final = new ParquetVectorStoreCollection<int, Doc>("docs-write", demoPath);
    var updated = await final.GetAsync(3);
    var deleted = await final.GetAsync(5);
    int count = 0;
    await foreach (var _ in final.SearchAsync(updated!.Embedding, top: 100)) count++;
    Console.WriteLine($"Reload -> Get(3)='{updated.Text}', Get(5)={(deleted is null ? "null (deleted)" : deleted.Text)}, total rows={count}");

    File.Delete(demoPath);
    Console.WriteLine("(demo file cleaned up)");
}

static Doc MakeDoc(int id, string text, int seed)
{
    var rng = new Random(seed);
    var v = new float[768];
    for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
    return new Doc { Id = id, Text = text, Embedding = v };
}

// ─────────────────────────────────────────────────────────────────────────────
// The record model: plain MEVD attributes. Parquet column names match property
// names (Id, Text, Embedding), so Parquet.Net's serializer maps them directly.
// ─────────────────────────────────────────────────────────────────────────────
public sealed class Doc
{
    [VectorStoreKey]
    public int Id { get; set; }

    [VectorStoreData]
    public string Text { get; set; } = "";

    [VectorStoreVector(768, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public float[] Embedding { get; set; } = Array.Empty<float>();
}

// The persistence action a flush performed, surfaced so callers can see which
// Parquet write mode ran.
public enum ParquetWriteMode
{
    None,        // nothing pending
    Append,      // new keys written as an appended row group
    Rewrite,     // full file rewrite (first write, or compaction for update/delete)
}

// ─────────────────────────────────────────────────────────────────────────────
// The provider. Complete read + write MEVD collection over a Parquet file.
//   read  : load once into a contiguous, L2-normalized float[] matrix; serve
//           SearchAsync via a parallel TensorPrimitives.Dot scan.
//   write : mutate an in-memory dictionary (read-your-writes); persist at flush
//           boundaries — append a row group for pure inserts, full rewrite to
//           compact updates/deletes (Parquet row groups are immutable).
// ─────────────────────────────────────────────────────────────────────────────
public sealed class ParquetVectorStoreCollection<TKey, TRecord>
    : VectorStoreCollection<TKey, TRecord>
    where TKey : notnull
    where TRecord : class, new()
{
    private readonly string _name;
    private readonly string _path;
    private readonly PropertyInfo _keyProp;
    private readonly PropertyInfo _vectorProp;
    private readonly VectorStoreCollectionMetadata _metadata;
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private readonly SemaphoreSlim _mutateLock = new(1, 1);

    private readonly Dictionary<TKey, TRecord> _store = new();
    private readonly List<TRecord> _pendingAppends = new(); // new keys not yet on disk
    private bool _needsCompaction;                          // an update/delete made disk stale
    private bool _loaded;

    private float[]? _matrix;   // contiguous, normalized, row-major [n * dim]
    private TRecord[]? _matrixRecords; // record order aligned to _matrix rows
    private bool _matrixDirty = true;
    private int _dim;

    public ParquetVectorStoreCollection(string name, string parquetFilePath)
    {
        _name = name;
        _path = parquetFilePath;
        _keyProp = FindAttributed(typeof(VectorStoreKeyAttribute), "[VectorStoreKey]");
        _vectorProp = FindAttributed(typeof(VectorStoreVectorAttribute), "[VectorStoreVector]");
        _dim = _vectorProp.GetCustomAttribute<VectorStoreVectorAttribute>()?.Dimensions ?? 0;
        _metadata = new VectorStoreCollectionMetadata
        {
            VectorStoreSystemName = "parquet",
            CollectionName = name,
        };
    }

    public override string Name => _name;

    public override Task<bool> CollectionExistsAsync(CancellationToken cancellationToken = default)
        => Task.FromResult(File.Exists(_path));

    public override async Task<TRecord?> GetAsync(
        TKey key, RecordRetrievalOptions? options = default, CancellationToken cancellationToken = default)
    {
        await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
        await _mutateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try { return _store.TryGetValue(key, out var r) ? r : null; }
        finally { _mutateLock.Release(); }
    }

    public override async IAsyncEnumerable<VectorSearchResult<TRecord>> SearchAsync<TInput>(
        TInput searchValue, int top, VectorSearchOptions<TRecord>? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ReadOnlyMemory<float> q = searchValue switch
        {
            ReadOnlyMemory<float> rom => rom,
            float[] arr => arr,
            _ => throw new NotSupportedException(
                $"Query type {typeof(TInput).Name} not supported; pass ReadOnlyMemory<float> or float[].")
        };

        var (records, matrix, d) = await SnapshotAsync(cancellationToken).ConfigureAwait(false);
        int n = records.Length;

        var query = NormalizedCopy(q.Span, d);
        var scores = new float[n];
        System.Threading.Tasks.Parallel.ForEach(
            System.Collections.Concurrent.Partitioner.Create(0, n),
            range =>
            {
                var qs = (ReadOnlySpan<float>)query;
                for (int i = range.Item1; i < range.Item2; i++)
                    scores[i] = TensorPrimitives.Dot(matrix.AsSpan(i * d, d), qs);
            });

        int skip = options?.Skip ?? 0;
        double? threshold = options?.ScoreThreshold;
        foreach (var (idx, score) in SelectTopK(scores, n, skip + top))
        {
            if (skip-- > 0) continue;
            if (threshold is double t && score < t) yield break;
            cancellationToken.ThrowIfCancellationRequested();
            yield return new VectorSearchResult<TRecord>(records[idx], score);
        }
    }

    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceType == typeof(VectorStoreCollectionMetadata)) return _metadata;
        if (serviceKey is null && serviceType.IsInstanceOfType(this)) return this;
        return null;
    }

    // ── Write path ───────────────────────────────────────────────────────────
    public override async Task UpsertAsync(TRecord record, CancellationToken cancellationToken = default)
    {
        await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
        await _mutateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try { UpsertCore(record); }
        finally { _mutateLock.Release(); }
    }

    public override async Task UpsertAsync(IEnumerable<TRecord> records, CancellationToken cancellationToken = default)
    {
        await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
        await _mutateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try { foreach (var r in records) UpsertCore(r); }
        finally { _mutateLock.Release(); }
    }

    public override async Task DeleteAsync(TKey key, CancellationToken cancellationToken = default)
    {
        await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
        await _mutateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try { DeleteCore(key); }
        finally { _mutateLock.Release(); }
    }

    public override async Task DeleteAsync(IEnumerable<TKey> keys, CancellationToken cancellationToken = default)
    {
        await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
        await _mutateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try { foreach (var k in keys) DeleteCore(k); }
        finally { _mutateLock.Release(); }
    }

    public override async Task EnsureCollectionExistsAsync(CancellationToken cancellationToken = default)
    {
        await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
        if (File.Exists(_path)) return;
        await _mutateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try { await WriteFullAsync(cancellationToken).ConfigureAwait(false); } // creates schema file (empty is valid)
        finally { _mutateLock.Release(); }
    }

    public override async Task EnsureCollectionDeletedAsync(CancellationToken cancellationToken = default)
    {
        await _mutateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (File.Exists(_path)) File.Delete(_path);
            _store.Clear();
            _pendingAppends.Clear();
            _needsCompaction = false;
            _matrix = Array.Empty<float>();
            _matrixRecords = Array.Empty<TRecord>();
            _matrixDirty = false;
            _loaded = true; // nothing to reload; the file is gone
        }
        finally { _mutateLock.Release(); }
    }

    // Persist pending mutations. Returns which Parquet write mode ran.
    public async Task<ParquetWriteMode> FlushAsync(CancellationToken cancellationToken = default)
    {
        await EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
        await _mutateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_needsCompaction || !File.Exists(_path))
            {
                // Update/delete made the on-disk row groups stale (or there is no file
                // yet): rewrite the whole collection from the in-memory source of truth.
                if (_pendingAppends.Count == 0 && !_needsCompaction && _store.Count == 0)
                {
                    // Nothing to do and no file expected.
                    return ParquetWriteMode.None;
                }
                await WriteFullAsync(cancellationToken).ConfigureAwait(false);
                _pendingAppends.Clear();
                _needsCompaction = false;
                return ParquetWriteMode.Rewrite;
            }

            if (_pendingAppends.Count > 0)
            {
                // Pure inserts against an existing file: append a new row group.
                await ParquetSerializer.SerializeAsync(
                    _pendingAppends.ToArray(), _path,
                    new ParquetOptions { Append = true }, cancellationToken: cancellationToken)
                    .ConfigureAwait(false);
                _pendingAppends.Clear();
                return ParquetWriteMode.Append;
            }

            return ParquetWriteMode.None;
        }
        finally { _mutateLock.Release(); }
    }

    public override IAsyncEnumerable<TRecord> GetAsync(
        System.Linq.Expressions.Expression<Func<TRecord, bool>> filter, int top,
        FilteredRecordRetrievalOptions<TRecord>? options = null, CancellationToken ct = default)
        => throw new NotSupportedException("Filtered scan not supported; use SearchAsync.");

    // ── Write helpers (must hold _mutateLock) ─────────────────────────────────
    private void UpsertCore(TRecord record)
    {
        var key = (TKey)_keyProp.GetValue(record)!;
        ValidateVector(record);
        if (_store.ContainsKey(key))
        {
            _store[key] = record;     // in-place update -> on-disk row is now stale
            _needsCompaction = true;
        }
        else
        {
            _store[key] = record;
            _pendingAppends.Add(record);
        }
        _matrixDirty = true;
    }

    private void DeleteCore(TKey key)
    {
        if (_store.Remove(key))       // a removed row cannot be edited out of a row group
            _needsCompaction = true;
        _matrixDirty = true;
    }

    private Task WriteFullAsync(CancellationToken ct) =>
        ParquetSerializer.SerializeAsync(
            _store.Values.ToArray(), _path,
            new ParquetOptions { Append = false }, cancellationToken: ct);

    // ── Loading + matrix packing ──────────────────────────────────────────────
    private async Task EnsureLoadedAsync(CancellationToken ct)
    {
        if (_loaded) return;
        await _loadLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (_loaded) return;

            if (File.Exists(_path))
            {
                var result = await ParquetSerializer.DeserializeAsync<TRecord>(_path, cancellationToken: ct)
                                                     .ConfigureAwait(false);
                foreach (var r in result.Data)
                    _store[(TKey)_keyProp.GetValue(r)!] = r;
                if (_dim == 0 && _store.Count > 0)
                    _dim = ((float[])_vectorProp.GetValue(_store.Values.First())!).Length;
            }

            _matrixDirty = true;
            _loaded = true;
        }
        finally { _loadLock.Release(); }
    }

    // Ensure loaded + matrix packed, then return a stable snapshot for a search.
    private async Task<(TRecord[] records, float[] matrix, int dim)> SnapshotAsync(CancellationToken ct)
    {
        await EnsureLoadedAsync(ct).ConfigureAwait(false);
        await _mutateLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (_matrixDirty || _matrix is null || _matrixRecords is null)
                RebuildMatrix();
            return (_matrixRecords!, _matrix!, _dim);
        }
        finally { _mutateLock.Release(); }
    }

    // A write invalidates the packed matrix; repack from the in-memory store. For the
    // tiny write-demo collection this is trivial; the 50k read path never hits it
    // because it performs no writes.
    private void RebuildMatrix()
    {
        var records = _store.Values.ToArray();
        int n = records.Length;
        int d = _dim;
        var matrix = new float[n * d];
        for (int i = 0; i < n; i++)
        {
            var v = (float[])_vectorProp.GetValue(records[i])!;
            float norm = MathF.Sqrt(TensorPrimitives.Dot(v, v));
            if (norm > 1e-12f) TensorPrimitives.Divide(v, norm, matrix.AsSpan(i * d, d));
            else v.CopyTo(matrix, i * d);
        }
        _matrix = matrix;
        _matrixRecords = records;
        _matrixDirty = false;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────
    private void ValidateVector(TRecord record)
    {
        var v = (float[])_vectorProp.GetValue(record)!;
        if (_dim == 0) _dim = v.Length;
        if (v.Length != _dim)
            throw new ArgumentException($"Record vector has {v.Length} dims; collection has {_dim}.");
    }

    private float[] NormalizedCopy(ReadOnlySpan<float> v, int expectedDim)
    {
        if (v.Length != expectedDim)
            throw new ArgumentException($"Query has {v.Length} dims; collection has {expectedDim}.");
        var copy = v.ToArray();
        float norm = MathF.Sqrt(TensorPrimitives.Dot(copy, copy));
        if (norm > 1e-12f) TensorPrimitives.Divide(copy, norm, copy);
        return copy;
    }

    private static IEnumerable<(int idx, float score)> SelectTopK(float[] scores, int n, int k)
    {
        k = Math.Min(k, n);
        var topIdx = new int[k];
        var topScore = new float[k];
        Array.Fill(topScore, float.NegativeInfinity);
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
        for (int i = 0; i < k; i++) yield return (topIdx[i], topScore[i]);
    }

    private static PropertyInfo FindAttributed(Type attr, string label)
    {
        var props = typeof(TRecord).GetProperties()
            .Where(p => p.GetCustomAttributes(attr, true).Length > 0).ToArray();
        return props.Length == 1
            ? props[0]
            : throw new InvalidOperationException(
                $"{typeof(TRecord).Name} must have exactly one {label} property (found {props.Length}).");
    }

    // Best-effort flush of pending writes. Dispose(bool) is synchronous, so the async
    // flush is pumped here; a real provider would prefer IAsyncDisposable.
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            try { FlushAsync().GetAwaiter().GetResult(); }
            catch { /* best-effort: never throw from Dispose */ }
        }
        _store.Clear();
        _matrix = null;
        _matrixRecords = null;
    }
}
