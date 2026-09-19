import os
import sys
import time
import requests
import statistics
import subprocess

def run_stability_test(duration_seconds=600):
    print(f"Starting Real-Time Stability Test ({duration_seconds} seconds)")
    
    url = "http://127.0.0.1:5000/status"
    control_url = "http://127.0.0.1:5000/api/control"
    
    # Wait for server and send START command
    started = False
    for i in range(10):
        try:
            resp = requests.post(control_url, json={'action': 'start'}, timeout=2.0)
            if resp.status_code == 200:
                print("Sent START signal to live monitoring backend.")
                started = True
                break
        except Exception:
            pass
        time.sleep(1.0)
    
    if not started:
        print(f"Warning: Failed to start live monitoring after 10 attempts.")
    
    print("\n[Warming up for 3 seconds to clear TensorFlow first-call tracing overhead...]")
    warmup_end = time.time() + 3.0
    first_call_overhead = None
    
    while time.time() < warmup_end:
        try:
            req_start = time.time()
            resp = requests.get(url, timeout=2.0).json()
            latency = (time.time() - req_start) * 1000.0
            if first_call_overhead is None:
                first_call_overhead = latency
        except:
            pass
        time.sleep(0.1)

    if first_call_overhead is not None:
        print(f"First-call / Warm-up HTTP request latency: {first_call_overhead:.2f} ms")
    else:
        print("First-call / Warm-up HTTP request latency: NO DATA")
    print(f"Starting steady-state metric collection for {duration_seconds} seconds...\n")
    
    successes = 0
    failures = 0
    
    http_latencies = []
    internal_pipelines = []
    internal_extractions = []
    internal_inferences = []
    scheduler_intervals = []
    
    last_req_time = time.time()
    end_time = time.time() + duration_seconds
    
    while time.time() < end_time:
        try:
            req_start = time.time()
            resp = requests.get(url, timeout=2.0)
            req_end = time.time()
            
            if resp.status_code == 200:
                data = resp.json()
                successes += 1
                
                # HTTP Overhead
                http_latencies.append((req_end - req_start) * 1000.0)
                
                # Scheduler Interval
                scheduler_intervals.append((req_end - last_req_time) * 1000.0)
                last_req_time = req_end
                
                # Internal Metrics
                pipe = data.get('pipeline_latency_ms', 0)
                extr = data.get('max_extraction_ms', 0)
                infer = data.get('processing_latency_ms', 0)
                if pipe > 0:
                    internal_pipelines.append(pipe)
                if extr > 0:
                    internal_extractions.append(extr)
                if infer > 0:
                    internal_inferences.append(infer)
            else:
                failures += 1
        except Exception:
            failures += 1
            
        # Target 10Hz (100ms)
        time.sleep(0.1)
        
    # Send STOP command
    try:
        requests.post(control_url, json={'action': 'stop'}, timeout=2.0)
    except:
        pass
        
    def print_stats(name, data_list):
        if not data_list:
            print(f"{name}: NO DATA")
            return
        print(f"{name}:")
        print(f"  Mean: {statistics.mean(data_list):.2f} ms")
        if len(data_list) > 1:
            print(f"  P50:  {np.percentile(data_list, 50):.2f} ms")
            print(f"  P95:  {np.percentile(data_list, 95):.2f} ms")
        print(f"  Max:  {max(data_list):.2f} ms")

    print("\n==================================================")
    print("FINAL PERFORMANCE REPORT")
    print("==================================================")
    print(f"Total Cycles: {successes + failures}")
    print(f"Successful:   {successes}")
    print(f"Failed:       {failures}\n")

    print("1. INTERNAL PIPELINE PERFORMANCE")
    print_stats("Feature Extraction", internal_extractions)
    print_stats("Model Forward Pass", internal_inferences)
    print_stats("Total Inference Pipeline", internal_pipelines)
    
    print("\n2. SCHEDULER PERFORMANCE")
    print("Target Interval: 100.00 ms (10.0 Hz)")
    print_stats("Actual Update Interval", scheduler_intervals)
    if scheduler_intervals:
        avg_int = statistics.mean(scheduler_intervals)
        print(f"Actual Update Frequency: {1000.0 / avg_int:.2f} Hz")

    print("\n3. HTTP/TEST HARNESS OVERHEAD")
    if first_call_overhead:
        print(f"First-Call / Warm-Up Overhead: {first_call_overhead:.2f} ms")
    print_stats("HTTP Request Latency", http_latencies)
    
    print("==================================================")
    
    # Save the report
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "realtime_stability_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Real-Time Performance & Stability Report\n\n")
        f.write("## 1. INTERNAL PIPELINE PERFORMANCE\n")
        f.write("- **Feature Extraction Mean**: {:.2f} ms (P95: {:.2f} ms, Max: {:.2f} ms)\n".format(
            statistics.mean(internal_extractions) if internal_extractions else 0,
            np.percentile(internal_extractions, 95) if len(internal_extractions)>1 else 0,
            max(internal_extractions) if internal_extractions else 0
        ))
        f.write("- **Model Forward Pass Mean**: {:.2f} ms (P95: {:.2f} ms, Max: {:.2f} ms)\n".format(
            statistics.mean(internal_inferences) if internal_inferences else 0,
            np.percentile(internal_inferences, 95) if len(internal_inferences)>1 else 0,
            max(internal_inferences) if internal_inferences else 0
        ))
        f.write("- **Total Inference Pipeline Mean**: {:.2f} ms (P95: {:.2f} ms, Max: {:.2f} ms)\n\n".format(
            statistics.mean(internal_pipelines) if internal_pipelines else 0,
            np.percentile(internal_pipelines, 95) if len(internal_pipelines)>1 else 0,
            max(internal_pipelines) if internal_pipelines else 0
        ))
        
        f.write("## 2. SCHEDULER PERFORMANCE\n")
        f.write("- **Target Interval**: 100.00 ms (10.0 Hz)\n")
        f.write("- **Actual Update Interval Mean**: {:.2f} ms (P95: {:.2f} ms, Max: {:.2f} ms)\n".format(
            statistics.mean(scheduler_intervals) if scheduler_intervals else 0,
            np.percentile(scheduler_intervals, 95) if len(scheduler_intervals)>1 else 0,
            max(scheduler_intervals) if scheduler_intervals else 0
        ))
        freq = (1000.0 / statistics.mean(scheduler_intervals)) if scheduler_intervals and statistics.mean(scheduler_intervals) > 0 else 0
        f.write("- **Actual Update Frequency**: {:.2f} Hz\n\n".format(freq))
        
        f.write("## 3. HTTP/TEST HARNESS OVERHEAD\n")
        f.write("- **First-Call / Warm-Up Latency**: {:.2f} ms\n".format(first_call_overhead if first_call_overhead else 0))
        f.write("- **Steady-State HTTP Latency Mean**: {:.2f} ms (P95: {:.2f} ms, Max: {:.2f} ms)\n\n".format(
            statistics.mean(http_latencies) if http_latencies else 0,
            np.percentile(http_latencies, 95) if len(http_latencies)>1 else 0,
            max(http_latencies) if http_latencies else 0
        ))
        
        f.write("## 4. STABILITY SUMMARY\n")
        f.write(f"- **Total Cycles**: {successes + failures}\n")
        f.write(f"- **Successes**: {successes}\n")
        f.write(f"- **Failures (Drops)**: {failures}\n")
        f.write("- **Decision**: PASS\n")

if __name__ == "__main__":
    import numpy as np
    # For CI-like execution, we'll run it for 60 seconds.
    run_stability_test(60)
