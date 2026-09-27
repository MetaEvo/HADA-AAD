import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from analysis.analysis_utils import compute_bootstrap_ci, save_significance_tests
from analysis.plot_comparison import get_method_color

# Custom y-axis limits for each domain
DOMAIN_YLIM = {
    "paper_review": 0.8,
    "genesis_go2walking": 1.0,
    "genesis_go2walkback": 1.0,
    "genesis_go2hop": 0.5,
    "imo_grading": 0.7,
    "imo_proof": 1.0,
}
TO_FORMAT = False


def wrap_label(label):
    """Wrap long labels to multiple lines at semantic breakpoints."""
    # Define smart breakpoints in order of priority
    breakpoints = [
        (" w/o ", "\nw/o "),           # "DGM-HA w/o X" -> "DGM-HA\nw/o X"
        (" + ", "\n+ "),               # "X + Y" -> "X\n+ Y"
        (" (", "\n("),                 # "DGM-HA (description)" -> "DGM-HA\n(description)"
    ]

    result = label
    for pattern, replacement in breakpoints:
        if pattern in result:
            result = result.replace(pattern, replacement, 1)  # Only replace first occurrence

    # For remaining long segments, wrap at spaces if still too long
    lines = result.split("\n")
    wrapped_lines = []
    for line in lines:
        if len(line) > 20:
            # Try to break at a space near the middle
            words = line.split(" ")
            current = ""
            for word in words:
                if len(current) + len(word) + 1 <= 18:
                    current = current + " " + word if current else word
                else:
                    if current:
                        wrapped_lines.append(current)
                    current = word
            if current:
                wrapped_lines.append(current)
        else:
            wrapped_lines.append(line)

    return "\n".join(wrapped_lines)


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=1, help="How many top agents per run to include")
    args = parser.parse_args()
    n = args.top_n
    bootstrap = True  # bootstrap confidence intervals (instead of mean and std)

    # Provided data
    static_methods = ["SFT", "Representative Baseline", "Default Reward Function"]
    data = [
        (
            "multitask",
            ["paper_review"],
            {
                "Initial agent": [
                    [0.0],
                ],
                "DGM": [
                    [0.0],  # genid initial
                    [0.51],  # genid 27
                    [0.0],  # genid initial
                    [0.0],  # genid initial
                    [0.0],  # genid 53, timing out
                ],
                "DGM-custom": [
                    [0.59],  # genid 92
                    [0.57],  # genid 55
                    [0.64],  # genid 86
                    [0.65],  # genid 72
                    [0.57],  # genid 65
                ],
                "DGM-HA w/o open-ended exploration": [
                    [0.0],  # genid initial
                    [0.0],  # genid initial
                    [0.56],  # genid 100
                    [0.0],  # genid initial
                    [0.0],  # genid initial
                ],
                "DGM-HA w/o self-improve": [
                    [0.0],  # genid initial
                    [0.0],  # genid initial
                    [0.0],  # genid initial
                    [0.0],  # genid initial
                    [0.13],  # genid 48
                ],
                "DGM-HA": [
                    [0.68],  # genid 48
                    [0.71],  # genid 49
                    [0.59],  # genid 62
                    [0.75],  # genid 78
                    [0.75],  # genid 51
                ],
                "Representative Baseline (Yamada et al., 2025)": [
                    [0.63],
                ],
            },
        ),
        (
            "multitask",
            ["genesis_go2walkback"],
            {
                "Initial agent": [
                    [0.0],
                ],
                "DGM": [
                    [0.059],  # genid 83
                    [0.0],  # genid 89
                    [0.112],  # genid 44
                    [0.571],  # genid 84
                    [0.0],  # genid 12
                ],
                "DGM-custom": [
                    [0.0],  # genid 60
                    [0.0],  # genid 79
                    [0.721],  # genid 45
                    [0.402],  # genid 87
                    [0.0],  # genid 86
                ],
                "DGM-HA w/o open-ended exploration": [
                    [0.057],  # genid 3
                    [0.812],  # genid 54
                    [0.0],  # genid 1
                    [0.0],  # genid 44
                    [0.135],  # genid 4
                ],
                "DGM-HA w/o self-improve": [
                    [0.605],  # genid 63
                    [0.341],  # genid 97
                    [0.738],  # genid 62
                    [0.641],  # genid 31
                    [0.133],  # genid 34
                ],
                "DGM-HA": [
                    [0.798],  # genid 79
                    [0.613],  # genid 91
                    [0.811],  # genid 12
                    [0.807],  # genid 39
                    [0.797],  # genid 89
                ],
                "Default Reward Function": [
                    [0.757],
                ],
            },
        ),
        (
            "multitask",
            ["genesis_go2hop"],
            {
                "Initial agent": [
                    [0.060],
                ],
                "DGM": [
                    [0.0],  # genid 83
                    [0.0],  # genid 89
                    [0.0],  # genid 44
                    [0.090],  # genid 84
                    [0.058],  # genid 12
                ],
                "DGM-custom": [
                    [0.363],  # genid 60
                    [0.348],  # genid 79
                    [0.306],  # genid 45
                    [0.385],  # genid 87
                    [0.305],  # genid 86
                ],
                "DGM-HA w/o open-ended exploration": [
                    [0.0],  # genid 3
                    [0.348],  # genid 54
                    [0.116],  # genid 1
                    [0.0],  # genid 44
                    [0.232],  # genid 4
                ],
                "DGM-HA w/o self-improve": [
                    [0.204],  # genid 63
                    [0.348],  # genid 97
                    [0.213],  # genid 62
                    [0.263],  # genid 31
                    [0.180],  # genid 34
                ],
                "DGM-HA": [
                    [0.399],  # genid 79
                    [0.368],  # genid 91
                    [0.372],  # genid 12
                    [0.355],  # genid 39
                    [0.436],  # genid 89
                ],
                "Default Reward Function (standing tall)": [
                    [0.348],
                ],
            },
        ),
        (
            # same agent on all tasks
            "multitask",
            ["paper_review", "genesis_go2walkback"],
            {
                "Initial agent": [
                    [(0.0+0.0/2)],
                ],
                "DGM": [
                    [(0.0+0.059)/2],  # genid 83
                    [(0.0+0.0)/2],  # genid 89
                    [(0.0+0.112)/2],  # genid 44
                    [(0.0+0.571)/2],  # genid 84
                    [(0.0+0.0)/2],  # genid 12
                ],
                "DGM-custom": [
                    [(0.52+0.0)/2],  # genid 60
                    [(0.5+0.0)/2],  # genid 79
                    [(0.64+0.304)/2],  # genid 86
                    [(0.5+0.402)/2],  # genid 87
                    [(0.52+0.0)/2],  # genid 86
                ],
                "DGM-HA w/o open-ended exploration": [
                    [(0.0+0.057)/2],  # genid 3
                    [(0.0+0.812)/2],  # genid 54
                    [(0.56+0.0)/2],  # genid 100
                    [(0.0+0.0)/2],  # genid 44
                    [(0.0+0.134)/2],  # genid 4
                ],
                "DGM-HA w/o self-improve": [
                    [(0.0+0.605)/2],  # genid 63
                    [(0.0+0.341)/2],  # genid 97
                    [(0.0+0.738)/2],  # genid 62
                    [(0.0+0.641)/2],  # genid 31
                    [(0.0+0.133)/2],  # genid 34
                ],
                "DGM-HA": [
                    [(0.65+0.803)/2],  # genid 89
                    [(0.47+0.613)/2],  # genid 91
                    [(0.59+0.648)/2],  # genid 62
                    [(0.75+0.818)/2],  # genid 78
                    [(0.75+0.809)/2],  # genid 51
                ],
            },
        ),
        (
            # same agent on all tasks
            "multitask",
            ["paper_review", "genesis_go2hop"],
            {
                "Initial agent": [
                    [(0.0+0.060/2)],
                ],
                "DGM": [
                    [(0.0+0.0)/2],  # genid 83
                    [(0.0+0.0)/2],  # genid 89
                    [(0.0+0.0)/2],  # genid 44
                    [(0.0+0.090)/2],  # genid 84
                    [(0.0+0.058)/2],  # genid 12
                ],
                "DGM-custom": [
                    [(0.52+0.363)/2],  # genid 60
                    [(0.5+0.348)/2],  # genid 79
                    [(0.64+0.296)/2],  # genid 86
                    [(0.5+0.385)/2],  # genid 87
                    [(0.52+0.305)/2],  # genid 86
                ],
                "DGM-HA w/o open-ended exploration": [
                    [(0.0+0.0)/2],  # genid 3
                    [(0.0+0.348)/2],  # genid 54
                    [(0.56+0.0)/2],  # genid 100
                    [(0.0+0.0)/2],  # genid 44
                    [(0.0+0.232)/2],  # genid 4
                ],
                "DGM-HA w/o self-improve": [
                    [(0.0+0.204)/2],  # genid 63
                    [(0.0+0.348)/2],  # genid 97
                    [(0.0+0.213)/2],  # genid 62
                    [(0.0+0.263)/2],  # genid 31
                    [(0.0+0.180)/2],  # genid 34
                ],
                "DGM-HA": [
                    [(0.65+0.433)/2],  # genid 89
                    [(0.47+0.368)/2],  # genid 91
                    [(0.59+0.177)/2],  # genid 62
                    [(0.75+0.314)/2],  # genid 78
                    [(0.75+0.422)/2],  # genid 51
                ],
            },
        ),
        (
            "transfer_dgm",
            ["imo_grading"],
            {
                "Initial agent": [
                    [0.0],
                ],
                'DGM w/o self-improve': [
                    [0.0],  # genid initial
                    [0.0],  # genid 12, timing out
                    [0.0],  # genid 25
                    [0.0],  # genid 43
                    [0.0],  # genid initial
                ],
                "Transfer agents from prev DGM": [
                    [0.0],  # timing out
                    [0.0],
                    [0.01],
                    [0.0],  # timing out
                    [0.0],
                ],
                "DGM w/o self-improve + transfer": [
                    [0.0],  # genid 12, timing out
                    [0.0],  # genid 0
                    [0.01],  # genid 31
                    [0.02],  # genid 48
                    [0.0],  # genid 4, timing out
                ],
            }
        ),
        (
            "transfer_hyp",
            ["imo_grading"],
            {
                "Initial agent": [
                    [0.0],
                ],
                "DGM-HA w/o self-improve": [
                    [0.0],  # genid initial
                    [0.01],  # genid 8
                    [0.0],  # genid initial
                    [0.0],  # genid initial
                    [0.13],  # genid 5
                ],
                "Transfer agents from prev DGM-HA": [
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                    [0.0],
                ],
                "DGM-HA w/o self-improve + transfer": [
                    [0.63],  # gen 24
                    [0.63],  # gen 14
                    [0.60],  # gen 50
                    [0.54],  # gen 24
                    [0.63],  # gen 43
                ],
            },
        ),
        (
            "transfer_continue",
            ["imo_grading"],
            {
                "DGM-HA": [
                    [0.63],  # genid 66
                    [0.51],  # genid 135
                    [0.61],  # genid 197
                    [0.60],  # genid 147
                    [0.68],  # genid 189
                ],
                "DGM-HA + transfer": [
                    [0.72],  # genid 168
                    [0.55],  # genid 92
                    [0.66],  # genid 114
                    [0.62],  # genid 151
                    [0.64],  # genid 104
                ],
                "DGM-HA + transfer + from ProofAutoGrader": [
                    [0.7],  # genid 71, chosen by mae
                ],
                "Representative Baseline (Luong et al., 2025)": [
                    [0.67],
                ],
            }
        ),
        (
            # same agent on all tasks
            "parentselect",
            ["paper_review", "genesis_go2walkback"],
            {
                "DGM-HA (random parent selection)": [
                    [(0.67+0.737)/2],  # genid 100
                    [(0.62+0.084)/2],  # genid 66
                    [(0.57+0.525)/2],  # genid 87
                    [(0.50+0.306)/2],  # genid 2
                    [(0.59+0.498)/2],  # genid 60
                ],
                "DGM-HA (modifiable parent selection)": [
                    [(0.65+0.811)/2],  # genid 98
                    [(0.78+0.664)/2],  # genid 75
                    [(0.57+0.818)/2],  # genid 56
                    [(0.56+0.265)/2],  # genid 74
                    [(0.58+0.810)/2],  # genid 44
                ],
                "DGM-HA (score-child-prop parent selection)": [
                    [(0.65+0.803)/2],  # genid 89
                    [(0.47+0.613)/2],  # genid 91
                    [(0.59+0.648)/2],  # genid 62
                    [(0.75+0.818)/2],  # genid 78
                    [(0.75+0.809)/2],  # genid 51
                ],
            },
        ),
        (
            # same agent on all tasks
            "parentselect",
            ["paper_review", "genesis_go2hop"],
            {
                "DGM-HA (random parent selection)": [
                    [(0.67+0.185)/2],  # genid 100
                    [(0.62+0.194)/2],  # genid 66
                    [(0.57+0.293)/2],  # genid 87
                    [(0.50+0.330)/2],  # genid 2
                    [(0.59+0.271)/2],  # genid 60
                ],
                "DGM-HA (modifiable parent selection)": [
                    [(0.65+0.332)/2],  # genid 98
                    [(0.78+0.243)/2],  # genid 75
                    [(0.57+0.439)/2],  # genid 56
                    [(0.56+0.213)/2],  # genid 74
                    [(0.58+0.314)/2],  # genid 44
                ],
                "DGM-HA (score-child-prop parent selection)": [
                    [(0.65+0.433)/2],  # genid 89
                    [(0.47+0.368)/2],  # genid 91
                    [(0.59+0.177)/2],  # genid 62
                    [(0.75+0.314)/2],  # genid 78
                    [(0.75+0.422)/2],  # genid 51
                ],
            },
        ),
        (
            "diffjudges",
            ["imo_proof"],
            {
                "Initial agent (test: ProofAutoGrader)": [
                    [0.0],
                ],
                "Initial agent (test: BetterGrader)": [
                    [0.0],
                ],
                "DGM-HA (training: ProofAutoGrader) (test: ProofAutoGrader)": [
                    [0.471],  # genid 20
                    [0.443],  # genid 44
                    [0.569],  # genid 46
                    [0.307],  # genid 27
                    [0.460],  # genid 13
                ],
                "DGM-HA (training: ProofAutoGrader) (test: BetterGrader)": [
                    [0.486],
                    [0.443],
                    [0.571],
                    [0.290],
                    [0.345],
                ],
                "DGM-HA (training: ProofAutoGrader) (test: human eval)": [
                    [],
                ],
                "DGM-HA long run (training: ProofAutoGrader) (test: ProofAutoGrader)": [
                    [0.662],  # genid 358
                ],
                "DGM-HA long run (training: ProofAutoGrader) (test: BetterGrader)": [
                    [0.543],
                ],
                "DGM-HA long run (training: ProofAutoGrader) (test: human eval)": [
                    [],
                ],
            }
        ),
    ]

    # Ensure output directory exists
    out_dir = "./analysis/outputs"
    os.makedirs(out_dir, exist_ok=True)

    saved_paths = []

    # Updated loop to handle nested runs with top_n truncation
    for group, domains, methods in data:
        domain_label = "+".join(domains)

        method_names = []
        central_values = []  # mean or median depending on bootstrap
        lower_errors = []    # std or lower CI
        upper_errors = []    # std or upper CI
        methods_data = {}    # Store raw data for significance testing

        for mname, runs in methods.items():
            # runs: list of lists; take first n from each run and concatenate
            concatenated = []
            for run in runs:
                if run:  # guard empty runs
                    concatenated.extend(run[:n])
            if len(concatenated) == 0:
                continue  # skip methods with no data
            arr = np.asarray(concatenated, dtype=float)
            method_names.append(mname)
            methods_data[mname] = arr  # Store for significance testing

            if bootstrap:
                # Compute median and bootstrap confidence intervals using shared function
                median, ci_lower, ci_upper = compute_bootstrap_ci(arr, n_bootstrap=1000, ci_level=0.95, random_seed=42)
                central_values.append(median)

                # Store as errors relative to median
                lower_errors.append(median - ci_lower)
                upper_errors.append(ci_upper - median)
            else:
                # Original: mean and std
                central_values.append(np.mean(arr))
                std = np.std(arr)
                lower_errors.append(std)
                upper_errors.append(std)

        if not method_names:
            # no plottable methods for this (group, domains) entry
            continue

        central_values = np.array(central_values)
        lower_errors = np.array(lower_errors)
        upper_errors = np.array(upper_errors)

        # ----------------
        # Save significance tests
        # ----------------
        output_file = os.path.join('./analysis/outputs', f"testevals_significance_{group}_{domain_label}_top{n}.txt")
        metadata = {
            'group': group,
            'domains': domain_label,
            'top_n': n
        }
        save_significance_tests(methods_data, output_file, metadata, use_bootstrap=bootstrap)

        # --- Plot ---
        if TO_FORMAT:
            plt.figure(figsize=(10, 3))
        else:
            # plt.figure(figsize=(10, 3.5))
            plt.figure(figsize=(10, 6))
        zero_offset = 0.03

        # Split methods into bar vs static
        bar_methods, bar_values, bar_lower_errs, bar_upper_errs, bar_colors = [], [], [], [], []
        fallback_colors = list(plt.cm.tab10.colors)
        fallback_idx = 0
        for i, mname in enumerate(method_names):
            if not any(s in mname for s in static_methods):
                bar_methods.append(mname)
                bar_values.append(central_values[i])
                bar_lower_errs.append(lower_errors[i])
                bar_upper_errs.append(upper_errors[i])
                # Assign color
                color = get_method_color(mname)
                if color is not None:
                    bar_colors.append(color)
                else:
                    bar_colors.append(fallback_colors[fallback_idx % len(fallback_colors)])
                    fallback_idx += 1

        # Positions for bars (contiguous)
        bar_indices = np.arange(len(bar_methods))

        # Draw bars (if any)
        if len(bar_methods) > 0:
            # Draw bars separately: Initial agent without error bars, others with
            for idx, mname in enumerate(bar_methods):
                if mname == "Initial agent" or mname == "DGM-HA + transfer + from ProofAutoGrader":
                    # Draw without error bars
                    plt.bar(
                        bar_indices[idx],
                        bar_values[idx] + zero_offset,
                        bottom=-zero_offset,
                        color=bar_colors[idx],
                    )
                else:
                    # Draw with error bars
                    if bootstrap:
                        yerr = [[bar_lower_errs[idx]], [bar_upper_errs[idx]]]
                    else:
                        yerr = bar_lower_errs[idx]
                    plt.bar(
                        bar_indices[idx],
                        bar_values[idx] + zero_offset,
                        yerr=yerr,
                        bottom=-zero_offset,
                        capsize=6,
                        color=bar_colors[idx],
                    )

            # x-ticks ONLY for bar methods (wrap long labels)
            wrapped_labels = [wrap_label(m) for m in bar_methods]
            if TO_FORMAT:
                plt.xticks([], [])
            else:
                plt.xticks(bar_indices, wrapped_labels, fontsize=12)
            plt.yticks(fontsize=12)
            # Keep x-limits tight to bars
            plt.xlim(-0.5, len(bar_methods) - 0.5)
        else:
            # No bars to show (only static baselines) — keep a simple x-axis
            plt.xticks([], [])
            plt.yticks(fontsize=12)
            plt.xlim(0, 1)

        # Plot static (baseline) methods as horizontal lines + CI/std shading + right-side labels
        ax = plt.gca()
        x_left, x_right = ax.get_xlim()

        for i, mname in enumerate(method_names):
            if any(s in mname for s in static_methods):
                y_val = central_values[i]
                lower_err = lower_errors[i]
                upper_err = upper_errors[i]
                ax.axhline(y=y_val, color="black", linestyle="--", linewidth=1, label=mname)
                # Shaded band (CI or std)
                ax.fill_between(
                    [x_left, x_right],
                    y_val - lower_err,
                    y_val + upper_err,
                    alpha=0.1,
                    color="black",
                )

        # After drawing, place labels for static lines near the left edge
        x_span = x_right - x_left
        label_x = x_left + 0.05 * x_span  # a little padding from the left
        for i, mname in enumerate(method_names):
            if any(s in mname for s in static_methods):
                y_val = central_values[i]
                ax.text(label_x, y_val + 0.01, mname, color="black", fontsize=12, ha="left", va="bottom")

        # Axis and layout
        # plt.ylabel("Performance of Best Agent")
        # Calculate ylim as average of domain-specific limits
        ylim_max = sum(DOMAIN_YLIM.get(d, 1.0) for d in domains) / len(domains)
        ylim_min = 0.3 if group == "transfer_continue" else -zero_offset
        plt.ylim(ylim_min, ylim_max)
        plt.tight_layout()

        # Save PNG
        filename = f"testeval_{group}_{domain_label}_top{n}_barplot.png"
        path = os.path.join(out_dir, filename)
        plt.savefig(path, bbox_inches="tight")
        saved_paths.append(path)

        # Save PDF and SVG with transparent background
        pdfs_dir = os.path.join(out_dir, "pdfs")
        os.makedirs(pdfs_dir, exist_ok=True)
        base_name = f"testeval_{group}_{domain_label}_top{n}_barplot"
        plt.savefig(os.path.join(pdfs_dir, f"{base_name}.pdf"), bbox_inches="tight", transparent=True)
        plt.savefig(os.path.join(pdfs_dir, f"{base_name}.svg"), bbox_inches="tight", transparent=True)

        plt.close()

    print(f"Saved plots: {saved_paths}")
