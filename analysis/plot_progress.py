import argparse
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from utils.gl_utils import (
    get_saved_score,
    get_parent_genid,
    is_starting_node,
    load_archive_data,
    get_patch_files,
)
from utils.domain_utils import get_domain_splits, can_domain_ensembled

__all__ = ["plot_progress_single", "plot_progress_together"]

def _plot_progress_core(
    scores_by_genid,
    exp_dir,
    label_suffix,
    color="blue",
    split="train",
    type="agent",
    svg=False,
):
    """
    scores_by_genid: dict[genid -> float or None]
      (None means non-compilable/invalid; excluded from averages)
    """
    log_lines = []

    iterations = []
    best_scores = []
    avg_scores = []
    it_genid_dict = {}

    seen_genids = set()
    all_scores = []
    best_genid = None

    archive_path = os.path.join(exp_dir, "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=False)

    for entry in archive_data:
        archive_genids = entry.get("archive", [])
        for genid in archive_genids:
            if genid in seen_genids:
                continue
            seen_genids.add(genid)

            iteration = len(best_scores)
            iterations.append(iteration)
            it_genid_dict[iteration] = genid

            score = scores_by_genid.get(genid, None)

            if score is not None:
                all_scores.append(score)
                if len(best_scores) == 0 or score > best_scores[-1]:
                    best_genid = genid

            best_scores.append(max(all_scores) if all_scores else 0.0)
            avg_scores.append(sum(all_scores) / len(all_scores) if all_scores else 0.0)

    log_lines.append(f"Total iterations: {len(iterations)}")
    log_lines.append(f"Best scores: {[round(xs, 5) for xs in best_scores]}")

    # Best lineage
    lineage_genids = []
    curr = best_genid
    if curr is not None:
        while not is_starting_node(curr):
            lineage_genids.append(curr)
            curr = get_parent_genid(exp_dir, curr)
        lineage_genids.append(curr)
        lineage_genids.reverse()

    lineage_iterations = [it for it, gid in it_genid_dict.items() if gid in lineage_genids]
    lineage_scores = [(scores_by_genid.get(gid) or 0.0) for gid in lineage_genids]

    log_lines.append(f"Best lineage genids: {lineage_genids}")
    log_lines.append(f"Best lineage scores: {[round(score, 5) for score in lineage_scores]}")
    if lineage_genids:
        log_lines.append(f"Best lineage patches: {' '.join(get_patch_files(exp_dir, lineage_genids[-1]))}")

    color_schemes = {
        "blue": ['#4285F4', '#42d6f5', '#122240'],
        "green": ['#0F9D58', '#9e9c10', '#042A17'],
        "orange": ['#FF9C03', '#f56a00', '#533302'],
    }
    color_scheme = color_schemes.get(color, color_schemes["orange"])

    plt.plot(iterations, avg_scores, marker='.', color=color_scheme[1], label='Average of Archive')
    plt.plot(iterations, best_scores, marker='.', color=color_scheme[0], label='Best Agent')
    if lineage_iterations:
        plt.plot(lineage_iterations, lineage_scores, marker='o', color=color_scheme[2], label='Lineage to Final Best Agent')
    plt.xlabel("Iterations", fontsize=18)
    plt.ylabel("Score", fontsize=18)
    plt.grid()
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plot_png = os.path.join(exp_dir, f"progress_plot_{label_suffix}_{split}_{type}.png")
    plt.savefig(plot_png)

    log_lines.append("Progress plot saved at:")
    log_lines.append(f"- {plot_png}")

    if svg:
        plot_svg = os.path.join(exp_dir, f"progress_plot_{label_suffix}_{split}_{type}.svg")
        plt.savefig(plot_svg)
        log_lines.append(f"- {plot_svg}")

    plt.close()

    print("\n".join(log_lines), '\n')
    log_path = os.path.join(exp_dir, f"progress_info_{label_suffix}_{split}_{type}.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))


def _collect_scores_single_domain(domain, exp_dir, split, type_):
    """Return {genid: score or None} for one domain."""
    archive_path = os.path.join(os.path.normpath(exp_dir), "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=False)
    scores = {}
    for entry in archive_data:
        for genid in entry.get("archive", []):
            if genid not in scores:
                scores[genid] = get_saved_score(domain, exp_dir, genid, split=split, type=type_)
    return scores


def _collect_scores_together(domains, exp_dir, split, type_):
    """
    Aggregate across domains:
      - If ANY domain score is None -> None
      - else average of all domain scores
    Returns {genid: aggregated_score or None}.
    """
    archive_path = os.path.join(os.path.normpath(exp_dir), "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=False)

    all_genids = []
    seen = set()
    for entry in archive_data:
        for gid in entry.get("archive", []):
            if gid not in seen:
                seen.add(gid)
                all_genids.append(gid)

    agg = {}
    for gid in all_genids:
        per_domain = []
        compilable_everywhere = True
        for d in domains:
            s = get_saved_score(d, exp_dir, gid, split=split, type=type_)
            if s is None:
                compilable_everywhere = False
                break
            per_domain.append(s)
        agg[gid] = (sum(per_domain) / len(per_domain)) if (compilable_everywhere and per_domain) else None
    return agg

def plot_progress_single(
    domain,
    exp_dir,
    split="train",
    type="agent",
    color="blue",
    svg=False,
):
    """
    Generate a per-domain progress plot and info file.
    Saves:
      progress_plot_{domain}_{split}_{type}.png
      progress_info_{domain}_{split}_{type}.txt
    """
    scores = _collect_scores_single_domain(domain, exp_dir, split, type)
    _plot_progress_core(scores, exp_dir, label_suffix=domain, color=color, split=split, type=type, svg=svg)


def plot_progress_together(
    domains,
    exp_dir,
    split="train",
    type="agent",
    color="blue",
    svg=False,
):
    """
    Generate a combined (aggregated) progress plot over multiple domains.
    Aggregation rule: None if any domain is None, else average of domain scores.
    Saves:
      progress_plot_together_{domA}_{domB}_..._{split}_{type}.png
      progress_info_together_{domA}_{domB}_..._{split}_{type}.txt
    """
    scores = _collect_scores_together(domains, exp_dir, split, type)
    label_suffix = "together_" + "_".join(domains)
    _plot_progress_core(scores, exp_dir, label_suffix=label_suffix, color=color, split=split, type=type, svg=svg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot progress curves from archive.jsonl.")
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        required=True,
        help="One or more domains. If more than one is passed, scores are aggregated together.",
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the experiment run directory.")
    parser.add_argument("--color", type=str, default="blue", help="Color scheme for the plot.")
    parser.add_argument("--check_ensemble", action="store_true", help="If set, check if domains support ensembling.")
    parser.add_argument("--svg", action="store_true", help="If set, also save SVG version of the plot.")
    args = parser.parse_args()

    if len(args.domains) == 1:
        domain = args.domains[0]
        ensemble_domain = args.check_ensemble and can_domain_ensembled(domain)
        splits = get_domain_splits(domain)
        score_types = ["agent", "ensemble", "max"] if ensemble_domain else ["agent"]
        for split in splits:  # pyright: ignore
            for stype in score_types:
                plot_progress_single(domain, args.path, split=split, type=stype, color=args.color, svg=args.svg)
    else:
        # Multiple domains -> together mode
        all_ensemble_capable = args.check_ensemble and all(can_domain_ensembled(d) for d in args.domains)
        splits_sets = [set(get_domain_splits(d)) for d in args.domains]
        common_splits = sorted(list(set.intersection(*splits_sets)))
        if not common_splits:
            raise SystemExit("No common splits across the selected domains.")
        score_types = ["agent", "ensemble", "max"] if all_ensemble_capable else ["agent"]
        for split in common_splits:
            for stype in score_types:
                plot_progress_together(args.domains, args.path, split=split, type=stype, color=args.color, svg=args.svg)
