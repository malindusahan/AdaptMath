"""Measure topic extractor startup time, per-path latency, and high-throughput statistics."""

from __future__ import annotations

import time
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def measure_system_runtime() -> dict[str, float]:
    # 1. Measure model startup / load time
    t_start = time.perf_counter()
    from src.topic_extraction.topic_extractor import HybridTopicExtractor
    extractor = HybridTopicExtractor()
    load_time = time.perf_counter() - t_start

    # Warmup
    extractor.extract("warmup query for jit and sentence transformer cache")

    # 2. Measure dataset latency over 500 questions
    test_csv = PROJECT_ROOT / "data" / "topic_extraction" / "splits" / "test.csv"
    test_df = pd.read_csv(test_csv)
    sample_texts = test_df["text"].tolist()[:500]

    all_latencies_ms = []
    for text in sample_texts:
        t0 = time.perf_counter()
        extractor.extract(text)
        all_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    # 3. Path-specific latencies
    kw_queries = [
        "Can you explain PEMDAS?",
        "What is Pythagorean theorem?",
        "How to find prime factorization?",
        "Order of operations rules",
        "Explain slope intercept form",
    ] * 20  # 100 calls
    kw_lats = []
    for q in kw_queries:
        t0 = time.perf_counter()
        extractor.extract(q)
        kw_lats.append((time.perf_counter() - t0) * 1000.0)

    sem_queries = [
        "How steep is this line when plotted on a graph?",
        "How do I determine the center value of an ordered list of test scores?",
        "What is the mathematical definition of a circle circumference?",
        "How to calculate the probability of rolling a 6 on a die?",
        "Finding unknown variable in a proportion ratio",
    ] * 20  # 100 calls
    sem_lats = []
    for q in sem_queries:
        t0 = time.perf_counter()
        extractor.extract(q)
        sem_lats.append((time.perf_counter() - t0) * 1000.0)

    tfidf_fallback_queries = [
        "What is the difference between congruent figures and similar figures?",
        "Can you explain how scale factors work in geometry shapes?",
        "How do coordinates behave under geometric dilations?",
    ] * 35  # ~105 calls
    tfidf_lats = []
    for q in tfidf_fallback_queries:
        t0 = time.perf_counter()
        extractor.extract(q)
        tfidf_lats.append((time.perf_counter() - t0) * 1000.0)

    abs_queries = [
        "Can you help me?",
        "I have a question",
        "What should I do next?",
        "Hi there",
        "Thank you",
    ] * 20  # 100 calls
    abs_lats = []
    for q in abs_queries:
        t0 = time.perf_counter()
        extractor.extract(q)
        abs_lats.append((time.perf_counter() - t0) * 1000.0)

    return {
        "model_load_time_sec": float(load_time),
        "avg_prediction_latency_ms": float(np.mean(all_latencies_ms)),
        "median_prediction_latency_ms": float(np.median(all_latencies_ms)),
        "p95_prediction_latency_ms": float(np.percentile(all_latencies_ms, 95)),
        "keyword_latency_ms": float(np.mean(kw_lats)),
        "minilm_latency_ms": float(np.mean(sem_lats)),
        "tfidf_fallback_latency_ms": float(np.mean(tfidf_lats)),
        "abstention_latency_ms": float(np.mean(abs_lats)),
    }


def main():
    metrics = measure_system_runtime()
    print("=" * 60)
    print("PHASE 14 RUNTIME & PERFORMANCE BENCHMARKS")
    print("=" * 60)
    print(f"Model Load Time:            {metrics['model_load_time_sec']:.2f} s")
    print(f"Average Prediction Latency: {metrics['avg_prediction_latency_ms']:.2f} ms")
    print(f"Median Prediction Latency:  {metrics['median_prediction_latency_ms']:.2f} ms")
    print(f"P95 Prediction Latency:     {metrics['p95_prediction_latency_ms']:.2f} ms")
    print(f"Keyword Latency:            {metrics['keyword_latency_ms']:.2f} ms")
    print(f"MiniLM Latency:             {metrics['minilm_latency_ms']:.2f} ms")
    print(f"TF-IDF Fallback Latency:    {metrics['tfidf_fallback_latency_ms']:.2f} ms")
    print(f"Abstention Latency:         {metrics['abstention_latency_ms']:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
