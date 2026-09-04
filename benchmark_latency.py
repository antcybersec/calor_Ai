"""
Latency Benchmarking & Performance Profiling Utility.
Measures p50 and p95 latency for both the Text Path and Vision Image Path.
"""

import time
import os
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from agent import run_agent_turn
from database import clear_user_data

BENCH_USER_ID = "latency_bench_user"
SAMPLE_IMG = os.path.join(os.path.dirname(__file__), "bench_plate.jpg")

def ensure_bench_image():
    if not os.path.exists(SAMPLE_IMG):
        img = Image.new('RGB', (300, 300), color=(200, 180, 160))
        img.save(SAMPLE_IMG)

def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    arr = np.array(latencies)
    return {
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "mean": round(float(np.mean(arr)), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3)
    }

def run_latency_benchmark(num_text_runs: int = 6, num_image_runs: int = 4):
    print("=" * 60)
    print("⚡ CALORAI LATENCY BENCHMARK (p50 / p95 PROFILER)")
    print("=" * 60)

    ensure_bench_image()
    clear_user_data(BENCH_USER_ID)

    text_queries = [
        "had 2 rotis and dal for lunch",
        "actually that was 3 rotis not 2",
        "how much protein have I had today?",
        "same as yesterday",
        "my usual",
        "i'm vegetarian btw"
    ]

    image_captions = [
        "plate of food for dinner",
        "half of this was my brother's",
        "lunch plate",
        "breakfast spread"
    ]

    print("\n[1/2] Benchmarking Text Path Latency...")
    text_latencies = []
    for i in range(num_text_runs):
        q = text_queries[i % len(text_queries)]
        t0 = time.time()
        res = run_agent_turn(user_id=BENCH_USER_ID, message_text=q)
        elapsed = time.time() - t0
        text_latencies.append(elapsed)
        print(f"  Run {i+1}/{num_text_runs}: '{q}' -> {round(elapsed, 3)}s")

    print("\n[2/2] Benchmarking Vision Path Latency...")
    image_latencies = []
    for i in range(num_image_runs):
        cap = image_captions[i % len(image_captions)]
        t0 = time.time()
        res = run_agent_turn(user_id=BENCH_USER_ID, message_text=cap, image_path=SAMPLE_IMG)
        elapsed = time.time() - t0
        image_latencies.append(elapsed)
        print(f"  Run {i+1}/{num_image_runs}: Photo + '{cap}' -> {round(elapsed, 3)}s")

    text_stats = calculate_percentiles(text_latencies)
    img_stats = calculate_percentiles(image_latencies)

    print("\n" + "=" * 60)
    print("📈 LATENCY BENCHMARK RESULTS")
    print("=" * 60)
    print(f"TEXT PATH   | p50: {text_stats['p50']}s | p95: {text_stats['p95']}s | Mean: {text_stats['mean']}s")
    print(f"VISION PATH | p50: {img_stats['p50']}s | p95: {img_stats['p95']}s | Mean: {img_stats['mean']}s")
    print("=" * 60)

    return {
        "text_path": text_stats,
        "vision_path": img_stats
    }

if __name__ == "__main__":
    run_latency_benchmark()
