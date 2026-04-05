"""
Statistical significance tests for model comparison.
- Wilcoxon signed-rank (pairwise comparison)
- Friedman test (all models simultaneously)
- Bootstrap confidence intervals
"""
import numpy as np
from scipy import stats
from itertools import combinations
from typing import Dict, List


def wilcoxon_pairwise(
    results: Dict[str, List[float]],
    metric: str = "f1_score",
    alpha: float = 0.05,
) -> Dict:
    """
    Pairwise Wilcoxon signed-rank test between all model pairs.

    Args:
        results: {model_name: [metric_value_run1, ..., metric_value_run5]}
        metric: name of the metric (for reporting)
        alpha: significance level

    Returns:
        Dict with test results for each pair.
    """
    model_names = list(results.keys())
    comparisons = {}

    for m1, m2 in combinations(model_names, 2):
        vals1 = np.array(results[m1])
        vals2 = np.array(results[m2])

        if len(vals1) < 5 or len(vals2) < 5:
            comparisons[f"{m1}_vs_{m2}"] = {
                "status": "insufficient_samples",
                "note": "Need at least 5 runs for Wilcoxon test"
            }
            continue

        try:
            stat, p_value = stats.wilcoxon(vals1, vals2, alternative="two-sided")
            comparisons[f"{m1}_vs_{m2}"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
                "better_model": m1 if np.mean(vals1) > np.mean(vals2) else m2,
                "mean_diff": float(np.mean(vals1) - np.mean(vals2)),
            }
        except Exception as e:
            comparisons[f"{m1}_vs_{m2}"] = {"error": str(e)}

    return {"metric": metric, "alpha": alpha, "comparisons": comparisons}


def friedman_test(
    results: Dict[str, List[float]],
    metric: str = "f1_score",
    alpha: float = 0.05,
) -> Dict:
    """
    Friedman test — non-parametric comparison across all models.
    Tests null hypothesis: all models perform equally.
    """
    model_names = list(results.keys())
    data_matrix = np.array([results[m] for m in model_names])

    if data_matrix.shape[1] < 3:
        return {"error": "Need at least 3 runs for Friedman test"}

    try:
        stat, p_value = stats.friedmanchisquare(*[data_matrix[i] for i in range(len(model_names))])

        return {
            "metric": metric,
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < alpha,
            "alpha": alpha,
            "n_models": len(model_names),
            "n_runs": data_matrix.shape[1],
            "model_means": {m: float(np.mean(results[m])) for m in model_names},
            "model_stds": {m: float(np.std(results[m])) for m in model_names},
            "ranking": sorted(model_names, key=lambda m: np.mean(results[m]), reverse=True),
        }
    except Exception as e:
        return {"error": str(e)}


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Compute bootstrap confidence interval for a metric.
    """
    np.random.seed(seed)
    values = np.array(values)
    n = len(values)

    boot_means = np.array([
        np.mean(np.random.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 + confidence) / 2 * 100

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci_lower": float(np.percentile(boot_means, lower_pct)),
        "ci_upper": float(np.percentile(boot_means, upper_pct)),
        "confidence": confidence,
        "n_samples": n,
    }


def compute_all_statistics(
    multi_run_results: Dict[str, List[Dict]],
    metrics_of_interest: List[str] = None,
) -> Dict:
    """
    Run all statistical tests given multi-run results.

    Args:
        multi_run_results: {model_name: [result_dict_run1, ..., result_dict_run5]}
    """
    if metrics_of_interest is None:
        metrics_of_interest = ["f1_score", "auc_roc", "precision", "recall"]

    all_stats = {}

    for metric in metrics_of_interest:
        # Extract metric values per model
        metric_values = {}
        for model_name, runs in multi_run_results.items():
            values = [r.get(metric, 0) for r in runs if metric in r]
            if values:
                metric_values[model_name] = values

        if len(metric_values) < 2:
            continue

        # Wilcoxon pairwise
        all_stats[f"wilcoxon_{metric}"] = wilcoxon_pairwise(metric_values, metric)

        # Friedman
        all_stats[f"friedman_{metric}"] = friedman_test(metric_values, metric)

        # Bootstrap CI per model
        for model_name, values in metric_values.items():
            all_stats[f"bootstrap_{metric}_{model_name}"] = bootstrap_ci(values)

    return all_stats
