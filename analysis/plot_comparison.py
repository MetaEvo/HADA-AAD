import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from utils.gl_utils import get_saved_score, load_archive_data
from utils.domain_utils import has_domain_val_subset
from analysis.analysis_utils import compute_bootstrap_ci, save_significance_tests

# Color mapping for consistent visualization across plots
# Organized by experiment groups in domains_paths_data
METHODS_TO_COLORS = {
    # Baselines (used across experiments)
    "Initial agent": "grey",
    "SFT": "black",
    "Representative Baseline": "black",
    "Default Reward Function": "black",

    # === MULTITASK experiment (ablation study) ===
    "DGM": "#d34d4d",                        # red
    "DGM-custom": "#7e45aa",        # purple
    "DGM-HA w/o open-ended exploration": "#edb301",  # yellow
    "DGM-HA w/o self-improve": "#00b051",       # green
    "DGM-HA": "#0270c0",                        # blue (main method)

    # === TRANSFER_DGM experiment ===
    "Transfer agents from prev DGM": "#4d4d4d",
    "DGM w/o self-improve": "lightcoral",       # red
    "DGM w/o self-improve + transfer": "darkred",  # dark red
    "DGM w/o self-improve + customization": "#7e45aa",  # purple

    # === TRANSFER_HYP experiment ===
    "Transfer agents from prev DGM-HA": "#4d4d4d",
    # "DGM-HA w/o self-improve" - already defined above
    "DGM-HA w/o self-improve + transfer": "#006a31",  # dark green

    # === TRANSFER_CONTINUE experiment ===
    # "DGM-HA" - already defined above
    "DGM-HA + transfer": "#014373",             # dark blue
    "DGM-HA + transfer + from ProofAutoGrader": "#022abf",  # dark blue

    # === PARENT_SELECTION experiment ===
    "DGM-HA (random parent selection)": "#4ea72e",            # apple green
    "DGM-HA (modifiable parent selection)": "#e97132",        # orange
    "DGM-HA (score-child-prop parent selection)": "#0270c0",  # blue (main)

    # === DIFFJUDGES experiment ===
    "DGM-HA (evaled by ProofAutoGrader)": "#0270c0",  # blue (main)
    "DGM-HA (evaled by better grader)": "#17becf",  # cyan
    "DGM-HA + transfer (evaled by ProofAutoGrader)": "#014373",  # dark blue
}
TO_FORMAT = True


def get_method_color(method_name):
    """Get color for a method, supporting substring matching for methods with citations."""
    # First try exact match
    if method_name in METHODS_TO_COLORS:
        return METHODS_TO_COLORS[method_name]
    # Then try substring match (for methods with citations like "Representative Baseline (Yamada et al., 2025)")
    for key, color in METHODS_TO_COLORS.items():
        if key in method_name:
            return color
    # Default fallback
    return None


def get_iterations_and_scores(path, domains):
    exp_dir = path
    archive_path = os.path.join(exp_dir, "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=True).get("archive", [])

    iterations = []
    cummax_scores = []
    max_score = 0

    # Iterate over generations
    for genid in archive_data:
        # Get per-domain scores
        per_domain_scores = []
        for domain in domains:
            split = "val" if has_domain_val_subset(domain) else "train"
            dom_score = get_saved_score(domain, exp_dir, genid, split=split, type="max")
            per_domain_scores.append(dom_score)
        # Compute cumulative max score
        if all(score is not None for score in per_domain_scores):
            score = sum(per_domain_scores) / len(per_domain_scores)
            max_score = max(max_score, score)
        # Save iteration and cumulative max score
        iterations.append(len(cummax_scores))
        cummax_scores.append(max_score)

    return iterations, cummax_scores


def _aggregate_runs_on_grid(runs, grid, use_bootstrap=False, n_bootstrap=1000, ci_level=0.95, random_seed=42):
    """Interpolate each run (xs, ys) onto the provided `grid` and compute mean/std or median/bootstrap CI."""
    interp_list = []
    grid = np.asarray(grid)
    for xs, ys in runs:
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if xs.size == 0:
            interp = np.full(grid.shape, np.nan)
            interp_list.append(interp)
            continue
        # sort (np.interp requires xp to be increasing)
        order = np.argsort(xs)
        xs_s = xs[order]
        ys_s = ys[order]
        # linear interpolation; values outside xp range will be set to nan
        interp = np.interp(grid, xs_s, ys_s, left=np.nan, right=np.nan)
        # ensure points outside the original domain are NaN (safety)
        outside_mask = (grid < xs_s.min()) | (grid > xs_s.max())
        interp[outside_mask] = np.nan
        interp_list.append(interp)

    arr = np.vstack(interp_list)  # shape (n_runs, n_grid)

    if use_bootstrap:
        # Set random seed for reproducibility
        rng = np.random.RandomState(random_seed)

        # Compute median and bootstrap confidence intervals
        median = np.nanmedian(arr, axis=0)
        lower_ci = np.zeros_like(median)
        upper_ci = np.zeros_like(median)

        # Bootstrap for each grid point (vectorized for speed)
        alpha = 1 - ci_level
        for i in range(len(grid)):
            col_data = arr[:, i]
            valid_data = col_data[~np.isnan(col_data)]

            if len(valid_data) == 0:
                lower_ci[i] = np.nan
                upper_ci[i] = np.nan
            elif len(valid_data) == 1:
                # Only one sample, CI equals the value itself
                lower_ci[i] = valid_data[0]
                upper_ci[i] = valid_data[0]
            else:
                # Vectorized bootstrap: generate all resamples at once
                n_samples = len(valid_data)
                resample_indices = rng.randint(0, n_samples, size=(n_bootstrap, n_samples))
                resamples = valid_data[resample_indices]  # shape: (n_bootstrap, n_samples)
                bootstrap_medians = np.median(resamples, axis=1)  # shape: (n_bootstrap,)

                lower_ci[i] = np.percentile(bootstrap_medians, alpha/2 * 100)
                upper_ci[i] = np.percentile(bootstrap_medians, (1 - alpha/2) * 100)

        return grid, median, lower_ci, upper_ci
    else:
        # Original behavior: mean and std
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        return grid, mean, std, None


def main():
    # Toggle this:
    truncate = False  # truncate each plot to the shortest run (fewest iterations)
    bootstrap = True  # bootstrap confidence intervals (instead of mean and std)

    # List of tuples (plotlabel, domains, paths)
    domains_paths_data = [
        (
            "multitask",
            ["paper_review", "genesis_go2walking"],
            {
                "DGM": [
                    "./outputs/generate_20251217_160752_316659",
                    "./outputs/generate_20251222_014818_110274",
                    "./outputs/generate_20251224_092332_847466",
                    "./outputs/generate_20260107_222803_829555",
                    "./outputs/generate_20260108_173932_869435",
                ],
                "DGM-custom": [
                    "./outputs/generate_20251222_001630_573951",
                    "./outputs/generate_20251222_135813_620703",
                    "./outputs/generate_20251223_083031_722293",
                    "./outputs/generate_20260109_142152_186754",
                    "./outputs/generate_20260110_221035_954288",
                ],
                'DGM-HA w/o open-ended exploration': [
                    "./outputs/generate_20251216_135441_274078",
                    "./outputs/generate_20251220_155654_173447",
                    "./outputs/generate_20251225_070547_512414",
                    "./outputs/generate_20260109_040324_329026",
                    "./outputs/generate_20260109_123123_514879",
                ],
                'DGM-HA w/o self-improve': [
                    "./outputs/generate_20251216_135402_613539",
                    "./outputs/generate_20251220_221822_898998",
                    "./outputs/generate_20251225_051018_858136",
                    "./outputs/generate_20260106_034905_301605",
                    "./outputs/generate_20260107_082846_607439",
                ],
                'DGM-HA': [
                    "./outputs/generate_20251216_192315_534288",
                    "./outputs/generate_20251219_105346_856092",
                    "./outputs/generate_20251219_105301_034575",
                    "./outputs/generate_20260106_094059_776217",
                    "./outputs/generate_20260107_131422_514432",
                ],
            }
        ),
        (
            "transfer",
            ["imo_grading"],
            {
                'DGM w/o self-improve': [
                    "./outputs/generate_20251222_171331_439292",
                    "./outputs/generate_20251227_010742_410768",
                    "./outputs/generate_20251227_141754_535006",
                    "./outputs/generate_20260112_062735_241342",
                    "./outputs/generate_20260112_131300_614863",
                ],
                'DGM w/o self-improve + transfer': [
                    "./outputs/generate_20251224_152317_450789",
                    "./outputs/generate_20251224_020523_132375",
                    "./outputs/generate_20251227_163414_796845",
                    "./outputs/generate_20260115_225140_990066",
                    "./outputs/generate_20260116_140424_976054",
                ],
                'DGM-HA w/o self-improve': [
                    "./outputs/generate_20251213_121122_148456",
                    "./outputs/generate_20251222_054321_890255",
                    "./outputs/generate_20251227_010438_125288",
                    "./outputs/generate_20260110_205156_808106",
                    "./outputs/generate_20260111_031001_998552",
                ],
                'DGM-HA w/o self-improve + transfer': [
                    "./outputs/generate_20251218_145224_922879",
                    "./outputs/generate_20251223_020515_080718",
                    "./outputs/generate_20251227_084459_138135",
                    "./outputs/generate_20260109_083654_392255",
                    "./outputs/generate_20260109_230653_107263",
                ],
            }
        ),
        (
            "transfer_continue",
            ["imo_grading"],
            {
                'DGM-HA': [
                    "./outputs/generate_20251229_075804_943053",
                    "./outputs/generate_20251212_141649_449529",
                    "./outputs/generate_20260108_103314_613227",
                    "./outputs/generate_20260117_061311_656062",
                    "./outputs/generate_20260117_211309_085168",
                ],
                'DGM-HA + transfer': [
                    "./outputs/generate_20260102_155816_312992",
                    "./outputs/generate_20260106_054744_957436",
                    "./outputs/generate_20260107_110639_657836",
                    "./outputs/generate_20260118_205530_595326",
                    "./outputs/generate_20260120_085406_048093",
                ],
                'DGM-HA + transfer + from ProofAutoGrader': [
                    "./outputs/generate_20260123_145202_608321",
                ],
            }
        ),
        (
            "parent_selection",
            ["paper_review", "genesis_go2walking"],
            {
                'DGM-HA (random parent selection)': [
                    "./outputs/generate_20260102_205330_890479",
                    "./outputs/generate_20260104_020830_626152",
                    "./outputs/generate_20260104_202105_355863",
                    "./outputs/generate_20260112_004139_012590",
                    "./outputs/generate_20260112_141932_885462",
                ],
                'DGM-HA (modifiable parent selection)': [
                    "./outputs/generate_20260102_205202_948855",
                    "./outputs/generate_20260103_233257_612218",
                    "./outputs/generate_20260105_060817_740419",
                    "./outputs/generate_20260110_035020_918239",
                    "./outputs/generate_20260110_222351_578109",
                ],
                'DGM-HA (score-child-prop parent selection)': [
                    "./outputs/generate_20251216_192315_534288",
                    "./outputs/generate_20251219_105346_856092",
                    "./outputs/generate_20251219_105301_034575",
                    "./outputs/generate_20260106_094059_776217",
                    "./outputs/generate_20260107_131422_514432",
                ],
            }
        ),
        (
            "diffjudges",
            ["imo_proof"],
            {
                'DGM-HA (evaled by ProofAutoGrader)': [
                    "./outputs/generate_20251230_160325_797106",
                    "./outputs/generate_20260114_105147_024606",
                    "./outputs/generate_20260115_092504_277336",  # long run
                    "./outputs/generate_20260117_060742_071671",
                    "./outputs/generate_20260118_171616_481195",
                ],
                'DGM-HA (evaled by better grader)': [
                    "./outputs/generate_20260127_180608_608129",
                    "./outputs/generate_20260128_144246_512696",
                    "./outputs/generate_20260129_102231_169514",
                    "./outputs/generate_20260130_122105_300424",
                    "./outputs/generate_20260130_192919_884973",
                ],
            }
        ),
    ]

    # Make outputs folder
    os.makedirs('./analysis/outputs', exist_ok=True)

    for plotlabel, domains, paths in domains_paths_data:
        print(f"Comparing on domains: {domains}")

        # ----------------
        # Collect runs and build a common grid
        # ----------------
        runs_by_label = {}
        all_xs = []
        per_run_max_x = []  # for truncation

        for label, run_paths in paths.items():
            runs = []
            for p in run_paths:
                xs, ys = get_iterations_and_scores(p, domains=domains)
                runs.append((xs, ys))
                if len(xs) > 0:
                    xs_arr = np.asarray(xs)
                    all_xs.append(xs_arr)
                    per_run_max_x.append(xs_arr.max())
            runs_by_label[label] = runs

        if len(all_xs) == 0:
            print("No data found for this plot; skipping.")
            continue

        # create a sorted, unique grid from all x values
        grid = np.unique(np.concatenate(all_xs))

        # If truncating, restrict the grid to the shortest run (fewest iterations)
        if truncate and len(per_run_max_x) > 0:
            max_common_x = int(np.min(per_run_max_x))  # inclusive
            grid = grid[grid <= max_common_x]

        # ----------------
        # Save significance tests
        # ----------------
        # Extract best found score for each method (final value in cummax_scores)
        best_scores_by_label = {}
        for label, runs in runs_by_label.items():
            if len(runs) == 0:
                continue
            best_scores = []
            for xs, ys in runs:
                if len(ys) > 0:
                    best_scores.append(ys[-1])
            if len(best_scores) > 0:
                best_scores_by_label[label] = np.array(best_scores)

        # Save significance tests using shared function
        output_file = os.path.join('./analysis/outputs', f"significance_{plotlabel}_{'+'.join(domains)}.txt")
        metadata = {
            'domains': ', '.join(domains),
            'plot_label': plotlabel
        }
        save_significance_tests(best_scores_by_label, output_file, metadata, use_bootstrap=bootstrap)

        # ----------------
        # Plot
        # ----------------
        plt.figure(figsize=(10, 6))

        for label, runs in runs_by_label.items():
            if len(runs) == 0:
                print(f"Skipping label {label}: no runs.")
                continue

            result = _aggregate_runs_on_grid(runs, grid, use_bootstrap=bootstrap)
            g, central, param, _ = result

            # When all values are NaN for a label, skip plotting
            if np.all(np.isnan(central)):
                print(f"Label '{label}' has no overlapping data on the grid; skipping.")
                continue

            color = get_method_color(label)
            plt.plot(g, central, label=label, linewidth=2, color=color)

            # Add dot at first point for ProofAutoGrader transfer in transfer_continue plot
            if plotlabel == "transfer_continue" and label == "DGM-HA + transfer + from ProofAutoGrader":
                # Find first non-NaN point
                valid_mask = ~np.isnan(central)
                if np.any(valid_mask):
                    first_idx = np.argmax(valid_mask)
                    plt.scatter([g[first_idx]], [central[first_idx]], color=color, s=100, zorder=5, label="ProofAutoGrader (Luong et al., 2025)")

            # Shade confidence region
            if bootstrap:
                # param is lower_ci, result[3] is upper_ci
                lower_ci = param
                upper_ci = result[3]
                plt.fill_between(g, lower_ci, upper_ci, alpha=0.25, color=color)
            else:
                # param is std, compute mean +/- std
                std = param
                lower = central - std
                upper = central + std
                plt.fill_between(g, lower, upper, alpha=0.25, color=color)

        if not TO_FORMAT:
            plt.xlabel('Iterations', fontsize=18)
            plt.ylabel('Score of Best Agent', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.5)
        plt.tight_layout()

        # Save PNG
        out_file = f"./analysis/outputs/comparison_{plotlabel}_{'+'.join(domains)}.png"
        plt.savefig(out_file, dpi=300)
        print(f"Saved plot to: {os.path.abspath(out_file)}")

        # Save PDF and SVG with transparent background
        pdfs_dir = "./analysis/outputs/pdfs"
        os.makedirs(pdfs_dir, exist_ok=True)
        base_name = f"comparison_{plotlabel}_{'+'.join(domains)}"
        plt.savefig(os.path.join(pdfs_dir, f"{base_name}.pdf"), transparent=True)
        plt.savefig(os.path.join(pdfs_dir, f"{base_name}.svg"), transparent=True)

        plt.close()


if __name__ == "__main__":
    main()
