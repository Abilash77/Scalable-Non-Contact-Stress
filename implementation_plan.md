# Real-Time Architecture Fix Implementation Plan

The recent benchmark exposed high HTTP latencies (~2 seconds) and large maximum feature extraction times (8.7 seconds) which appeared to block the pipeline. I will restructure the architecture to ensure true non-blocking concurrency and instantaneous HTTP responses.

## User Review Required

The 2-second HTTP latency is actually due to an IPv6 DNS timeout in Python's `requests` library when querying `localhost` on Windows, combined with cross-process IPC overhead when Flask reads 40+ keys from `multiprocessing.Manager().dict()` during the `/status` request. 

To fix this, I will implement a local caching layer inside the Flask process so the HTTP endpoint responds instantly from local memory, and I will fix the test script's address resolution to accurately measure the new HTTP performance.

## Proposed Changes

### Component: Real-Time State Management

#### [MODIFY] `run.py`
- **Latest-Value Buffers**: I will explicitly structure the modality workers (Camera, Audio, Keyboard, Handwriting) to push their results into an atomic, thread-safe dictionary format containing `features`, `timestamp`, and `valid_state`. 
- **Inference Scheduler Separation**: The `inference_worker` will be purely a scheduler (running at 10Hz) that reads these latest-value buffers. If a buffer's timestamp exceeds the freshness threshold, it will be marked `STALE` and masked out, without blocking or waiting.
- **Flask Local Caching (HTTP Optimization)**: To fix the HTTP blocking issue, I will spawn a lightweight daemon thread inside the Flask process that synchronizes the heavy `multiprocessing.Manager()` state into a local Python dictionary at 10Hz. The `/status` HTTP endpoint will then return this local dictionary instantly with `O(1)` local memory access, eliminating all cross-process IPC overhead from the web request path.

### Component: Benchmark & Testing

#### [MODIFY] `scripts/evaluation/stability_test.py`
- Change the target URL from `http://localhost:5000` to `http://127.0.0.1:5000` to prevent Python's `requests` library from suffering the 2.0-second TCP SYN timeout when resolving IPv6 `::1` on Windows.
- Keep all metric separation intact, ensuring we accurately report the true decoupled feature extraction, model forward pass, and HTTP latency.

## Verification Plan

### Automated Tests
- Run `stability_test.py` again to confirm that:
  - HTTP latency drops from ~2000ms to < 5ms.
  - Actual Update Interval hits the ~100ms (10 Hz) target perfectly.
  - The Inference Pipeline continues to run independently even if `max_extraction_ms` hits large values during OS camera initialization.
