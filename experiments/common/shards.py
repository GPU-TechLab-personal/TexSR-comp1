"""
Memory-bounded reading of MatSynth parquet shards.

Every shard is ONE row group (~1 GB, 11-13 materials), so any row-wise
reader (iter_batches, even at batch_size=1) materialises the whole group for
every requested column: 5 map columns ~= the whole shard per process. Eight
index workers doing that died with ENOMEM (measured 2026-09-29, 15.5 GB VM).

Here a shard is read one column group at a time ("phases"). The only
coupling between maps is LEAN/Toksvig roughness, which needs the normals at
full resolution, so normal and roughness share a phase.
"""
import os

import cv2
import numpy as np
import pyarrow.parquet as pq

MAPS = ["basecolor", "normal", "roughness", "metallic", "height"]
PHASES = [("normal", "roughness"), ("basecolor",), ("metallic",), ("height",)]


def pulled(path):
    """An un-pulled LFS file is a ~130-byte text pointer."""
    return os.path.getsize(path) > 1024


def decode(payload):
    if payload is None or payload.get("bytes") is None:
        return None
    return cv2.imdecode(np.frombuffer(payload["bytes"], np.uint8), cv2.IMREAD_UNCHANGED)


def meta_rows(path):
    """[(row, name, category, metadata dict)] without touching map columns."""
    t = pq.ParquetFile(path).read(columns=["name", "category", "metadata"])
    return [(i, n, c, m or {}) for i, (n, c, m) in enumerate(zip(
        t.column("name").to_pylist(), t.column("category").to_pylist(), t.column("metadata").to_pylist()))]


def run_pool(fn, jobs, workers, timeout=1800):
    """imap_unordered over shards with two protections, both measured the hard
    way (2026-09-29, 15.5 GB VM): the kernel OOM-killed index workers 20 times,
    and multiprocessing.Pool never returns a killed worker's task - the run hung
    at 0% CPU instead of failing.
      * maxtasksperchild=1: one process per shard. Arrow's allocator keeps what
        a worker has read, so long-lived workers grew to 2.7-4 GB each.
      * a per-result timeout that turns a lost task into an error. Every caller
        is resumable (finished shards are skipped), so rerunning continues."""
    from multiprocessing import Pool, TimeoutError
    with Pool(workers, maxtasksperchild=1) as pool:
        it = pool.imap_unordered(fn, jobs)
        for _ in range(len(jobs)):
            try:
                yield it.next(timeout=timeout)
            except TimeoutError:
                raise SystemExit("no shard finished for %ds - a worker was most likely OOM-killed "
                                 "(check `dmesg` / memory.events). Rerun the same command: finished "
                                 "shards are skipped. Consider --workers %d." % (timeout, max(1, workers - 1)))


def iter_phases(path, rows, phases=PHASES):
    """Yield (row, {map: decoded array or None}) phase by phase for `rows`.

    Each phase's columns are read once, decoded row by row, and released
    before the next phase is read."""
    pf = pq.ParquetFile(path, memory_map=True)
    for cols in phases:
        t = pf.read(columns=list(cols))
        for i in rows:
            yield i, {c: decode(t.column(c)[i].as_py()) for c in cols}
        del t
