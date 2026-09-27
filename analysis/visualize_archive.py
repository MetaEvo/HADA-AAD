import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib.colors import LinearSegmentedColormap

from utils.gl_utils import (
    load_archive_data,
    get_saved_score,
    get_parent_genid,
    get_node_metadata_key,
    is_starting_node,
)
from utils.domain_utils import get_domain_splits, can_domain_ensembled


def build_graph_single_domain(domain, archive_data, output_dir, trunc_its=-1, split="train", type="agent"):
    G = nx.DiGraph()
    score_map = {}

    for entry in archive_data:
        archive = entry["archive"]
        for genid in archive:
            if trunc_its > 0 and len(score_map) >= trunc_its:
                break

            genid_str = str(genid)
            parent_genid = get_parent_genid(output_dir, genid)
            parent_str = str(parent_genid)

            if parent_genid is not None and parent_str != genid_str:
                G.add_edge(parent_str, genid_str)

            if genid_str not in score_map:
                score_map[genid_str] = get_saved_score(domain, output_dir, genid, split=split, type=type)
            if parent_genid is not None and parent_str not in score_map:
                score_map[parent_str] = get_saved_score(domain, output_dir, parent_genid, split=split, type=type)

    # In G and score_map, rename all nodes or keys "initial" to "0"
    for n in G.nodes():
        if n == "initial":
            G = nx.relabel_nodes(G, {"initial": "0"})
            score_map["0"] = score_map.pop("initial")
            break

    return G, score_map


def build_graph_together(domains, archive_data, output_dir, trunc_its=-1, split="train", type="agent"):
    """
    Build a single unified graph. For each node, compute the overall score as:
      - None if ANY domain's score is None (not compilable in that domain)
      - otherwise the average of all domain scores
    """
    G = nx.DiGraph()
    score_map = {}  # genid_str -> aggregated score (None or float)

    # Build edges once from the archive
    seen_nodes = set()
    for entry in archive_data:
        archive = entry["archive"]
        for genid in archive:
            genid_str = str(genid)
            parent_genid = get_parent_genid(output_dir, genid)
            parent_str = str(parent_genid)

            if parent_genid is not None and parent_str != genid_str:
                G.add_edge(parent_str, genid_str)

            seen_nodes.add(genid_str)
            if parent_genid is not None:
                seen_nodes.add(parent_str)

    # Optionally truncate by iteration order (arbitrary but deterministic over seen order)
    nodes_ordered = list(seen_nodes)
    if trunc_its > 0:
        nodes_ordered = nodes_ordered[:trunc_its]

    # Compute aggregated scores
    for n in nodes_ordered:
        per_domain_scores = []
        compilable_everywhere = True
        for d in domains:
            s = get_saved_score(d, output_dir, n, split=split, type=type)
            if s is None:
                compilable_everywhere = False
                break
            per_domain_scores.append(s)

    # Ensure all nodes in graph have a score entry (even if truncated eliminated some)
        score_map[n] = (float(np.mean(per_domain_scores)) if compilable_everywhere and per_domain_scores else None)

    for n in G.nodes():
        if n not in score_map:
            per_domain_scores = []
            compilable_everywhere = True
            for d in domains:
                s = get_saved_score(d, output_dir, n, split=split, type=type)
                if s is None:
                    compilable_everywhere = False
                    break
                per_domain_scores.append(s)
            score_map[n] = (float(np.mean(per_domain_scores)) if compilable_everywhere and per_domain_scores else None)

    # In G and score_map, rename all nodes or keys "initial" to "0"
    for n in G.nodes():
        if n == "initial":
            G = nx.relabel_nodes(G, {"initial": "0"})
            score_map["0"] = score_map.pop("initial")
            break

    return G, score_map


def visualize_graph(G, score_map, output_dir, name_suffix, split="train", type="agent", plot_borders=False, save_svg=False):
    if len(G.nodes()) == 0:
        print("No nodes to visualize.")
        return

    pos = graphviz_layout(G, prog="dot")

    nodes = list(G.nodes())
    scores = [score_map.get(node, None) for node in nodes]

    # Separate scores into valid and invalid
    valid_scores = [s for s in scores if s is not None]
    if len(valid_scores) == 0:
        # Degenerate case: everyone invalid. Still draw, with a neutral map.
        min_score = 0.0
        max_score = 1.0
    else:
        min_score = min(valid_scores)
        max_score = max(valid_scores)

    # Normalize only valid scores
    normalized_scores = []
    for s in scores:
        if s is None:
            normalized_scores.append(0.0)  # will still get color at low end
        else:
            normalized_scores.append((s - min_score) / (max_score - min_score) if max_score != min_score else 0.5)

    # Create colormap
    colors = [
        (0.0, "#ffaf00"),
        (0.4, "#ffd000"),
        (0.8, "#fffb00"),
        (1.0, "#03ff00"),
    ]
    cmap = LinearSegmentedColormap.from_list("orange_yellow_green", colors, N=256)
    node_colors = cmap(normalized_scores)
    # cmap = plt.cm.viridis
    # node_colors = plt.cm.viridis(normalized_scores)

    # Determine valid_parent via metadata; starting node is always valid
    def _is_valid_parent(node_id: str) -> bool:
        try:
            if is_starting_node(node_id):
                return True
            vp = get_node_metadata_key(output_dir, node_id, "valid_parent")
            # Default to True if missing
            return True if vp is None else bool(vp)
        except Exception:
            # On any read error, default to True to avoid false negatives
            return True

    # Border color rule requested:
    #   If NOT a valid_parent -> red border
    #   Else -> green border
    border_colors = ['red' if not _is_valid_parent(n) else 'green' for n in nodes]

    # Identify max score node among valid
    valid_score_map = {k: v for k, v in score_map.items() if v is not None}
    max_score_node = max(valid_score_map, key=valid_score_map.get) if valid_score_map else None

    fig, ax = plt.subplots(figsize=(16, 10))

    # Draw all nodes with custom border colors
    for i, n in enumerate(nodes):
        shape = 'D' if (max_score_node is not None and n == max_score_node) else 'o'
        nx.draw_networkx_nodes(
            G, pos, nodelist=[n],
            node_color=[node_colors[i]],
            **({"edgecolors": [border_colors[i]]} if plot_borders else {"edgecolors": ["black"]}),
            node_size=800,
            node_shape=shape,
            linewidths=2 if plot_borders else 1,
            ax=ax
        )

    # Draw labels, first line is node ID, second line is score
    labels = {}
    for n in nodes:
        s = score_map.get(n)
        if s is None:
            score_str = "N/A"
        else:
            try:
                score_str = f"{s:.3f}"
            except Exception:
                score_str = str(s)
        labels[n] = f"#{n}\n{score_str}"

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8, font_color='black')

    # Colorbar for valid score range (or a dummy range if none valid)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(np.linspace(min_score, max_score, 256))
    cbar = fig.colorbar(sm, ax=ax, label="Score (aggregated)" if isinstance(name_suffix, str) and "together" in name_suffix else "Score")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(cbar.ax.get_ylabel(), fontsize=14)

    ax.axis('off')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"archive_graph_{name_suffix}_{split}_{type}.png")
    plt.savefig(output_path, dpi=300)
    print(f"Graph saved to {output_path}")
    if save_svg:
        svg_path = os.path.join(output_dir, f"archive_graph_{name_suffix}_{split}_{type}.svg")
        plt.savefig(svg_path, format='svg', transparent=True)
        print(f"SVG saved to {svg_path}")
    print()
    plt.close()


def visualize_archive_single(domain, exp_dir, trunc_its=-1, split="train", type="agent", plot_borders=False, save_svg=False):
    exp_dir = os.path.normpath(exp_dir)
    archive_path = os.path.join(exp_dir, "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=False)

    G, score_map = build_graph_single_domain(domain, archive_data, exp_dir, trunc_its=trunc_its, split=split, type=type)
    visualize_graph(G, score_map, exp_dir, name_suffix=domain, split=split, type=type, plot_borders=plot_borders, save_svg=save_svg)


def visualize_archive_together(domains, exp_dir, trunc_its=-1, split="train", type="agent", plot_borders=False, save_svg=False):
    exp_dir = os.path.normpath(exp_dir)
    archive_path = os.path.join(exp_dir, "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=False)

    G, score_map = build_graph_together(domains, archive_data, exp_dir, trunc_its=trunc_its, split=split, type=type)
    name_suffix = "together_" + "_".join(domains)
    visualize_graph(G, score_map, exp_dir, name_suffix=name_suffix, split=split, type=type, plot_borders=plot_borders, save_svg=save_svg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",  # one or more domains
        required=True,
        help="One or more domains to evaluate (must be from the allowed list)",
    )
    parser.add_argument("--path", type=str, default="./outputs/generate_20250617_080733_189684", help="Path to the run.")
    parser.add_argument("--trunc_its", type=int, default=-1, help="Truncate iterations to this number, -1 for no truncation")
    parser.add_argument(
        "--together",
        action="store_true",
        help="If set, plot all specified domains together on a single graph, aggregating scores per node."
    )
    parser.add_argument("--plot_borders", action="store_true", help="If set, plot node borders based on valid_parent metadata.")
    parser.add_argument("--check_ensemble", action="store_true", help="If set, check if domains support ensembling.")
    parser.add_argument("--svg", action="store_true", help="If set, save the figure as SVG format.")
    args = parser.parse_args()

    exp_dir = args.path
    domains = args.domains

    if args.together:
        # Determine common splits across the chosen domains
        domain_splits = [set(get_domain_splits(d)) for d in domains]
        common_splits = sorted(list(set.intersection(*domain_splits))) if domain_splits else []

        if not common_splits:
            print("No common splits across the selected domains. Nothing to plot.")
            raise SystemExit(0)

        # Only use ensemble types if ALL domains support ensembling
        ensemble_domain_all = args.check_ensemble and all(can_domain_ensembled(d) for d in domains)
        score_types = ["agent", "ensemble", "max"] if ensemble_domain_all else ["agent"]

        for split in common_splits:
            for stype in score_types:
                visualize_archive_together(
                    domains,
                    exp_dir,
                    trunc_its=args.trunc_its,
                    split=split,
                    type=stype,
                    plot_borders=args.plot_borders,
                    save_svg=args.svg,
                )
    else:
        for domain in domains:
            ensemble_domain = args.check_ensemble and can_domain_ensembled(domain)
            splits = get_domain_splits(domain)
            score_types = ["agent", "ensemble", "max"] if ensemble_domain else ["agent"]
            for split in splits:  # pyright: ignore
                for stype in score_types:
                    visualize_archive_single(
                        domain,
                        exp_dir,
                        trunc_its=args.trunc_its,
                        split=split,
                        type=stype,
                        plot_borders=args.plot_borders,
                        save_svg=args.svg,
                    )
