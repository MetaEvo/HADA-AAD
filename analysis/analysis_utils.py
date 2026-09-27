"""Shared utilities for analysis plotting and statistical testing."""
import os
import numpy as np
from scipy import stats


def compute_bootstrap_ci(data, n_bootstrap=1000, ci_level=0.95, random_seed=42):
    """
    Compute median and bootstrap confidence intervals for data.

    Args:
        data: 1D array of values
        n_bootstrap: number of bootstrap iterations
        ci_level: confidence interval level (default 0.95 for 95% CI)
        random_seed: random seed for reproducibility

    Returns:
        tuple: (median, lower_ci, upper_ci)
    """
    data = np.asarray(data)
    median = np.median(data)

    if len(data) == 1:
        # Single sample - no confidence interval
        return median, median, median

    # Vectorized bootstrap
    rng = np.random.RandomState(random_seed)
    alpha = 1 - ci_level

    # Generate all resamples at once for speed
    resample_indices = rng.randint(0, len(data), size=(n_bootstrap, len(data)))
    resamples = data[resample_indices]  # shape: (n_bootstrap, len(data))
    bootstrap_medians = np.median(resamples, axis=1)  # shape: (n_bootstrap,)

    lower_ci = np.percentile(bootstrap_medians, alpha/2 * 100)
    upper_ci = np.percentile(bootstrap_medians, (1 - alpha/2) * 100)

    return median, lower_ci, upper_ci


def save_significance_tests(methods_data, output_file, metadata=None, use_bootstrap=False):
    """
    Perform pairwise significance testing between methods and save results to a file.

    Uses non-parametric tests (Wilcoxon/Mann-Whitney) for bootstrap/median comparisons.
    Uses parametric tests (paired/independent t-test) for mean/std comparisons.

    Args:
        methods_data: dict mapping method name to array of scores
        output_file: path to output file
        metadata: dict with optional keys like 'group', 'domains', 'plotlabel', etc.
        use_bootstrap: whether bootstrap mode is used
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    metadata = metadata or {}

    # Write results
    with open(output_file, 'w') as f:
        f.write(f"Significance Testing Results\n")
        f.write(f"{'=' * 80}\n")

        # Write metadata if provided
        for key, value in metadata.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")

        f.write(f"Method: {'Bootstrap/Median' if use_bootstrap else 'Mean/Std'}\n\n")

        # Pairwise comparisons
        if use_bootstrap:
            f.write(f"Pairwise Comparisons (Non-parametric tests)\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"Note: Using Wilcoxon signed-rank test (paired) or Mann-Whitney U test (unpaired)\n")
            f.write(f"One-sided test: H1 is that first method > second method\n")
            f.write(f"Non-parametric tests appropriate for median-based comparisons\n\n")
        else:
            f.write(f"Pairwise Comparisons (Parametric tests)\n")
            f.write(f"{'-' * 80}\n")
            f.write(f"Note: Using paired t-test (paired) or independent t-test (unpaired)\n")
            f.write(f"One-sided test: H1 is that first method > second method\n")
            f.write(f"Assumes runs are paired (e.g., same random seeds across methods)\n\n")

        method_names = list(methods_data.keys())
        for i, method1 in enumerate(method_names):
            for method2 in method_names[:i] + method_names[i+1:]:
                scores1 = methods_data[method1]
                scores2 = methods_data[method2]

                n1, n2 = len(scores1), len(scores2)

                if use_bootstrap:
                    # Non-parametric tests for bootstrap/median
                    if n1 != n2:
                        # Mann-Whitney U test for unpaired data
                        statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='greater')
                        test_name = "Mann-Whitney U test"

                        median1, ci_lower1, ci_upper1 = compute_bootstrap_ci(scores1)
                        median2, ci_lower2, ci_upper2 = compute_bootstrap_ci(scores2)

                        f.write(f"{method1} vs {method2}\n")
                        f.write(f"  WARNING: Unequal sample sizes (n1={n1}, n2={n2}), using unpaired test\n")
                        f.write(f"  {method1}: median={median1:.4f}, 95% CI=[{ci_lower1:.4f}, {ci_upper1:.4f}], n={n1}\n")
                        f.write(f"  {method2}: median={median2:.4f}, 95% CI=[{ci_lower2:.4f}, {ci_upper2:.4f}], n={n2}\n")
                        f.write(f"  {test_name} statistic: {statistic:.4f}\n")
                        f.write(f"  p-value ({method1} > {method2}): {p_value:.6f}")
                    else:
                        # Wilcoxon signed-rank test for paired data
                        differences = scores1 - scores2

                        if np.all(differences == 0):
                            p_value = 1.0
                            median1, ci_lower1, ci_upper1 = compute_bootstrap_ci(scores1)
                            median2, ci_lower2, ci_upper2 = compute_bootstrap_ci(scores2)
                            f.write(f"{method1} vs {method2}\n")
                            f.write(f"  {method1}: median={median1:.4f}, 95% CI=[{ci_lower1:.4f}, {ci_upper1:.4f}], n={n1}\n")
                            f.write(f"  {method2}: median={median2:.4f}, 95% CI=[{ci_lower2:.4f}, {ci_upper2:.4f}], n={n2}\n")
                            f.write(f"  No difference between paired runs (all differences = 0)\n")
                            f.write(f"  p-value: 1.000000")
                        else:
                            # Wilcoxon signed-rank test
                            statistic, p_value = stats.wilcoxon(scores1, scores2, alternative='greater')

                            median1, ci_lower1, ci_upper1 = compute_bootstrap_ci(scores1)
                            median2, ci_lower2, ci_upper2 = compute_bootstrap_ci(scores2)
                            median_diff = np.median(differences)

                            f.write(f"{method1} vs {method2}\n")
                            f.write(f"  {method1}: median={median1:.4f}, 95% CI=[{ci_lower1:.4f}, {ci_upper1:.4f}], n={n1}\n")
                            f.write(f"  {method2}: median={median2:.4f}, 95% CI=[{ci_lower2:.4f}, {ci_upper2:.4f}], n={n2}\n")
                            f.write(f"  Median difference: {median_diff:.4f}\n")
                            f.write(f"  Wilcoxon statistic: {statistic:.4f}\n")
                            f.write(f"  p-value ({method1} > {method2}): {p_value:.6f}")
                else:
                    # Parametric tests for mean/std
                    if n1 != n2:
                        # Independent t-test for unpaired data
                        statistic, p_value = stats.ttest_ind(scores1, scores2, alternative='greater')
                        test_name = "Independent t-test"

                        mean1, std1 = np.mean(scores1), np.std(scores1)
                        mean2, std2 = np.mean(scores2), np.std(scores2)

                        f.write(f"{method1} vs {method2}\n")
                        f.write(f"  WARNING: Unequal sample sizes (n1={n1}, n2={n2}), using unpaired test\n")
                        f.write(f"  {method1}: mean={mean1:.4f}, std={std1:.4f}, n={n1}\n")
                        f.write(f"  {method2}: mean={mean2:.4f}, std={std2:.4f}, n={n2}\n")
                        f.write(f"  {test_name} statistic: {statistic:.4f}\n")
                        f.write(f"  p-value ({method1} > {method2}): {p_value:.6f}")
                    else:
                        # Paired t-test
                        differences = scores1 - scores2

                        if np.all(differences == 0):
                            p_value = 1.0
                            f.write(f"{method1} vs {method2}\n")
                            f.write(f"  {method1}: mean={np.mean(scores1):.4f}, std={np.std(scores1):.4f}, n={n1}\n")
                            f.write(f"  {method2}: mean={np.mean(scores2):.4f}, std={np.std(scores2):.4f}, n={n2}\n")
                            f.write(f"  No difference between paired runs (all differences = 0)\n")
                            f.write(f"  p-value: 1.000000")
                        elif np.std(differences) == 0:
                            mean_diff = np.mean(differences)
                            f.write(f"{method1} vs {method2}\n")
                            f.write(f"  {method1}: mean={np.mean(scores1):.4f}, std={np.std(scores1):.4f}, n={n1}\n")
                            f.write(f"  {method2}: mean={np.mean(scores2):.4f}, std={np.std(scores2):.4f}, n={n2}\n")
                            f.write(f"  Mean difference: {mean_diff:.4f}\n")
                            f.write(f"  All differences are identical (std=0)\n")
                            if mean_diff > 0:
                                p_value = 0.0
                                f.write(f"  p-value ({method1} > {method2}): 0.000000")
                            else:
                                p_value = 1.0
                                f.write(f"  p-value ({method1} > {method2}): 1.000000")
                        else:
                            statistic, p_value = stats.ttest_rel(scores1, scores2, alternative='greater')

                            mean1, std1 = np.mean(scores1), np.std(scores1)
                            mean2, std2 = np.mean(scores2), np.std(scores2)
                            mean_diff = np.mean(differences)
                            std_diff = np.std(differences, ddof=1)

                            f.write(f"{method1} vs {method2}\n")
                            f.write(f"  {method1}: mean={mean1:.4f}, std={std1:.4f}, n={n1}\n")
                            f.write(f"  {method2}: mean={mean2:.4f}, std={std2:.4f}, n={n2}\n")
                            f.write(f"  Mean difference: {mean_diff:.4f}, std: {std_diff:.4f}\n")
                            f.write(f"  t-statistic: {statistic:.4f}\n")
                            f.write(f"  p-value ({method1} > {method2}): {p_value:.6f}")

                if p_value < 0.001:
                    f.write(" ***")
                elif p_value < 0.01:
                    f.write(" **")
                elif p_value < 0.05:
                    f.write(" *")

                f.write("\n\n")

    print(f"Saved significance tests to: {os.path.abspath(output_file)}")
