# resource_tracker semaphore warning — root cause and fix plan

## Symptom

```
UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects...
```

## Root cause

Not a docling bug — it's a cleanup warning from Python's
`multiprocessing.resource_tracker`, and it's a symptom of how the process
died, not a bug in the ingestion code.

Docling's PDF pipeline spins up worker processes (torch/OCR backends
underneath), which allocate POSIX semaphores in `/dev/shm`. The log shows
`Stopping reloader process [3879]` right in the middle of
`preprocess ... pages=[25]` — uvicorn's `--reload` watcher killed the
parent process while the worker pool was still active. The child processes
never got a clean shutdown, so the semaphore was never unlinked. The
resource tracker unlinked it on its own and printed the warning. One leaked
semaphore, already cleaned up — nothing corrupted.

**Why it keeps happening:** running a long docling conversion inside a
`--reload` dev server. Any file save mid-ingest triggers a hard restart of
the parent process, and this warning fires every time.

## Fixes

1. **Move ingestion out of the reload-enabled server (the real fix).**
   Run it as a standalone script or a background worker (rq/celery/arq)
   that the API just enqueues to. A 25-page doc at 0.5–4s/page is minutes
   of work and shouldn't live in a dev-server request path.

2. **If ingestion must stay in the app**, add a FastAPI `lifespan`
   shutdown handler that disposes the `DocumentConverter` / shuts down its
   executor, so teardown is orderly instead of an abrupt kill.

3. **Reduce spawned worker pools:**
   ```bash
   export TOKENIZERS_PARALLELISM=false
   export OMP_NUM_THREADS=1   # or a small number
   ```
   This often makes the warning disappear entirely.

4. **Verify nothing is actually accumulating:**
   ```bash
   ls /dev/shm | head
   ```
   Should stay clean between runs.

## Separate issue worth investigating

Per-page preprocess times climb from ~0.1s to 4.1s over the course of the
25-page run. That's usually memory pressure or thread contention building
up over the run, not page complexity. If ingestion feels slow, this is the
thread to pull — not the semaphore warning.
