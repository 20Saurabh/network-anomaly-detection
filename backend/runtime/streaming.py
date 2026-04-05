"""
Streaming inference pipeline with concept drift detection.
Simulates real-time anomaly detection with micro-batch processing.
"""
import time
import numpy as np
import torch
from collections import deque
from typing import Optional, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    DEVICE, STREAM_BATCH_SIZE, DRIFT_DELTA, DRIFT_THRESHOLD
)


class ADWINDriftDetector:
    """
    ADWIN (Adaptive Windowing) concept drift detector.
    Simplified implementation for streaming anomaly detection.
    Monitors prediction confidence — drift = distribution shift in model certainty.
    """

    def __init__(self, delta: float = DRIFT_DELTA):
        self.delta = delta
        self.window = deque()
        self.total = 0.0
        self.count = 0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Add new value and check for drift."""
        self.window.append(value)
        self.total += value
        self.count += 1

        if self.count < 30:
            return False

        # Check for drift: compare first half vs second half
        mid = len(self.window) // 2
        window_list = list(self.window)
        first_half = window_list[:mid]
        second_half = window_list[mid:]

        if len(first_half) < 10 or len(second_half) < 10:
            return False

        mean1 = np.mean(first_half)
        mean2 = np.mean(second_half)

        # Simplified drift test
        n1, n2 = len(first_half), len(second_half)
        m = 1.0 / n1 + 1.0 / n2
        epsilon = np.sqrt(0.5 * m * np.log(2.0 / self.delta))

        if abs(mean1 - mean2) > epsilon:
            self.drift_detected = True
            # Reset window
            self.window = deque(second_half)
            self.total = sum(second_half)
            self.count = len(second_half)
            return True

        # Trim if too large
        if len(self.window) > 5000:
            removed = self.window.popleft()
            self.total -= removed
            self.count -= 1

        return False


class PageHinkleyDetector:
    """
    Page-Hinkley test for concept drift detection.
    Complementary to ADWIN — detects gradual drift.
    """

    def __init__(self, threshold: float = DRIFT_THRESHOLD,
                 delta: float = 0.005, alpha: float = 0.9999):
        self.threshold = threshold
        self.delta = delta
        self.alpha = alpha
        self.m_t = 0
        self.M_T = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float) -> bool:
        self.count += 1
        self.sum += value
        mean = self.sum / self.count

        self.m_t = self.alpha * self.m_t + (value - mean - self.delta)
        self.M_T = max(self.M_T, self.m_t)

        if self.M_T - self.m_t > self.threshold:
            # Reset
            self.m_t = 0
            self.M_T = 0
            self.sum = 0
            self.count = 0
            return True
        return False


class StreamingInferencePipeline:
    """
    Micro-batch streaming inference for real-time anomaly detection.
    Simulates real-time processing via CSV replay.
    """

    def __init__(self, model, scaler=None, device=DEVICE,
                 batch_size=STREAM_BATCH_SIZE):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.batch_size = batch_size

        self.adwin = ADWINDriftDetector()
        self.ph = PageHinkleyDetector()

        self.total_processed = 0
        self.total_anomalies = 0
        self.drift_events = []
        self.latencies = []

    def process_batch(self, X_batch: np.ndarray) -> Dict:
        """Process a micro-batch of flows."""
        from models.base import BaseSklearnModel

        start = time.perf_counter()

        if self.scaler:
            X_batch = self.scaler.transform(X_batch)

        if isinstance(self.model, BaseSklearnModel):
            predictions = self.model.predict(X_batch)
            scores = self.model.get_anomaly_scores(X_batch)
        else:
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(X_batch).to(self.device)
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                if self.model.model_type == "unsupervised":
                    scores = ((x - output) ** 2).mean(dim=-1).cpu().numpy()
                    threshold = np.mean(scores) + 2 * np.std(scores)
                    predictions = (scores > threshold).astype(int)
                else:
                    probs = torch.softmax(output, dim=-1)
                    predictions = output.argmax(dim=-1).cpu().numpy()
                    scores = probs[:, 1].cpu().numpy() if probs.shape[1] > 1 else probs[:, 0].cpu().numpy()

        elapsed = time.perf_counter() - start

        # Update drift detectors
        mean_confidence = float(np.mean(scores))
        adwin_drift = self.adwin.update(mean_confidence)
        ph_drift = self.ph.update(mean_confidence)

        n_anomalies = int(predictions.sum())
        self.total_processed += len(X_batch)
        self.total_anomalies += n_anomalies
        self.latencies.append(elapsed)

        result = {
            "batch_size": len(X_batch),
            "anomalies_detected": n_anomalies,
            "mean_score": float(mean_confidence),
            "latency_ms": elapsed * 1000,
            "throughput": len(X_batch) / max(elapsed, 1e-8),
            "drift_adwin": adwin_drift,
            "drift_page_hinkley": ph_drift,
        }

        if adwin_drift or ph_drift:
            self.drift_events.append({
                "sample": self.total_processed,
                "type": "ADWIN" if adwin_drift else "Page-Hinkley",
                "mean_score": float(mean_confidence),
            })

        return result

    def simulate_stream(self, X: np.ndarray, verbose: bool = True) -> Dict:
        """Simulate streaming by replaying data in micro-batches."""
        print(f"\n  [STREAM] Simulating stream: {len(X)} samples, "
              f"batch_size={self.batch_size}")

        all_results = []
        for i in range(0, len(X), self.batch_size):
            batch = X[i:i + self.batch_size]
            result = self.process_batch(batch)
            all_results.append(result)

            if verbose and (i // self.batch_size) % 10 == 0:
                print(f"    Batch {i // self.batch_size}: "
                      f"{result['anomalies_detected']} anomalies | "
                      f"{result['latency_ms']:.2f}ms | "
                      f"{result['throughput']:.0f} samples/sec"
                      f"{' ⚠️ DRIFT' if result['drift_adwin'] or result['drift_page_hinkley'] else ''}")

        summary = {
            "total_processed": self.total_processed,
            "total_anomalies": self.total_anomalies,
            "avg_latency_ms": float(np.mean([r["latency_ms"] for r in all_results])),
            "avg_throughput": float(np.mean([r["throughput"] for r in all_results])),
            "max_latency_ms": float(np.max([r["latency_ms"] for r in all_results])),
            "drift_events": len(self.drift_events),
            "drift_details": self.drift_events,
        }

        print(f"  [STREAM] Complete: {summary['total_processed']} processed, "
              f"{summary['total_anomalies']} anomalies, "
              f"{summary['drift_events']} drift events")

        return summary
