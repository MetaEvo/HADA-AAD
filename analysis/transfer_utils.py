import argparse
from collections import defaultdict, deque
import os
import pprint

from utils.constants import REPO_NAME
from utils.gl_utils import get_saved_score, load_archive_data, get_parent_genid, get_patch_files
from utils.domain_utils import has_domain_val_subset
from domains.imo.setup_proofgrader_repo import get_mae_score



def _build_children_index(genid_to_parent):
    """Return mapping: parent -> set(children)."""
    children = defaultdict(set)
    for child, parent in genid_to_parent.items():
        if parent is not None:
            children[parent].add(child)
    return children

def _compute_growth_scores(genid_to_scores, genid_to_parent, gamma=0.6, max_depth=-1, immediate_diff=False):
    """
    For each node n, growth(n) = sum_{desc in descendants(n)}
        [ max(score(desc) - score(parent(desc)), 0) * (gamma ** dist(n, desc)) ]
    where dist(n, desc) is the distance (in edges) from n to desc.
    This version computes deltas relative to each node’s *immediate parent*,
    not the root node.
    """
    parent_to_children = _build_children_index(genid_to_parent)
    growth = {g: [] for g in genid_to_scores.keys()}

    # BFS from each root node to compute growth contributions
    for root in genid_to_scores.keys():
        if root not in parent_to_children:
            continue  # leaf node
        q = deque([(c, 1) for c in parent_to_children[root]])
        seen = set()
        while q:
            node, depth = q.popleft()
            if node in seen or (max_depth > 0 and depth > max_depth):
                continue
            seen.add(node)

            parent = genid_to_parent.get(node)
            if parent in genid_to_scores and node in genid_to_scores:
                if immediate_diff:
                    # Difference with parent's score
                    delta = genid_to_scores[node] - genid_to_scores[parent]
                else:
                    # Difference with root's score
                    delta = genid_to_scores[node] - genid_to_scores[root]

                # Always normalize by distance from parent
                growth[root].append(delta * (gamma ** depth))

            # Continue exploring deeper descendants
            for ch in parent_to_children.get(node, ()):
                q.append((ch, depth + 1))

    # Only consider nodes with at least N descendants
    growth = {g: s for g, s in growth.items() if len(s) >= 3}
    # Average growth scores
    avg_growth = {g: (sum(scores) / len(scores) if len(scores) > 0 else 0.0) for g, scores in growth.items()}

    # pprint.pprint(avg_growth)
    # pprint.pprint(growth)
    return avg_growth

def choose_node_for_transfer(genid_to_scores, genid_to_parent, method="max_score", top_n=3, gamma=0.6):
    """
    method="max_score": pick node with highest absolute score.
    method="growth": pick node with highest descendant-driven growth potential.
    method="growth_imd": pick node with highest descendant-driven growth potential, with immediate difference from its parent.
    """
    if not genid_to_scores:
        return None

    if method == "max_score":
        return sorted(genid_to_scores, key=genid_to_scores.get, reverse=True)[:top_n]

    elif method == "growth":
        growth = _compute_growth_scores(genid_to_scores, genid_to_parent, gamma=gamma, max_depth=-1, immediate_diff=False)
        return sorted(growth, key=growth.get, reverse=True)[:top_n]

    elif method == "growth_imd":
        growth = _compute_growth_scores(genid_to_scores, genid_to_parent, gamma=gamma, max_depth=-1, immediate_diff=True)
        return sorted(growth, key=growth.get, reverse=True)[:top_n]

    return None

def get_run_eval_commands(output_dir, genids, domains):
    commands = []
    copy_root_dir = os.path.join(output_dir, f"gen_initial/{REPO_NAME}")
    test_domains = [("genesis_go2hop" if d == "genesis_go2walking" else d) for d in domains]
    for genid in genids:
        patch_files = get_patch_files(output_dir, genid)
        command = f"python -m domains.run_eval --output_dir {output_dir.rstrip('/')}_testevals --domains {' '.join(test_domains)} --run_id gen_{genid} --copy_root_dir {copy_root_dir} --patch_files {' '.join(patch_files)}"
        commands.append(command)
    return commands


if __name__ == "__main__":
    """
    Script to get a node for zero-shot transfer of self-referential improvement experiment
    """
    parser = argparse.ArgumentParser(description="Plot progress curves from archive.jsonl.")
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        required=True,
        help="One or more domains. If more than one is passed, scores are aggregated together.",
    )
    parser.add_argument("--path", type=str, required=True, help="Path to the experiment run directory.")
    parser.add_argument("--top_n", type=int, default=3, help="Number of nodes to choose from.")
    parser.add_argument("--get_commands", action="store_true", default=False, help="If set, get commands to run evals.")
    args = parser.parse_args()

    domains = args.domains
    output_dir = args.path

    # Load archive
    archive = load_archive_data(os.path.join(output_dir, 'archive.jsonl'), last_only=True)['archive']
    print(f"Archive length: {len(archive)}")
    
    # Get scores for each node
    genid_to_scores = {}
    for genid in archive:
        splits = ["val" if has_domain_val_subset(d) else "train" for d in domains]
        domain_scores = [get_saved_score(d, output_dir, genid, split=splits[i]) for i, d in enumerate(domains)]
        # domain_scores = [get_mae_score(d, output_dir, genid, split=splits[i]) for i, d in enumerate(domains)]
        # domain_scores = [(-score if score is not None else None) for score in domain_scores]
        if all(score is not None for score in domain_scores):
            genid_to_scores[genid] = sum(domain_scores) / len(domain_scores)
    print(f"Valid genids: {len(genid_to_scores)}")

    # Get parent relationships for each node
    genid_to_parent = {}
    for genid in archive:
        parent = get_parent_genid(output_dir, genid)
        genid_to_parent[genid] = parent

    # Choose node for transfer
    methods_params = [
        'max_score',
        'growth-1.0', 'growth-0.8', 'growth-0.6',
        'growth-0.5',
        'growth-0.4', 'growth-0.3',
        # 'growth_imd-1.0', 'growth_imd-0.8', 'growth_imd-0.6',
    ]
    for method_param in methods_params:
        splits = method_param.split('-')
        method = splits[0]
        gamma = float(splits[1]) if len(splits) > 1 else 0.6
        chosen_nodes = choose_node_for_transfer(
            genid_to_scores, genid_to_parent,
            method=method, top_n=args.top_n, gamma=gamma,
        )
        print(f"\nChosen node with {method_param}: {chosen_nodes}")

        for chosen_node in chosen_nodes[:1]:
            patch_files = get_patch_files(output_dir, chosen_node)
            print(f"Genid {chosen_node} Patch files: {' '.join(patch_files)}")

    # Get commands to run evals
    if args.get_commands:
        genids = choose_node_for_transfer(genid_to_scores, genid_to_parent, method="max_score", top_n=args.top_n)
        commands = get_run_eval_commands(output_dir, genids, domains)
        print("\n" + "\n\n".join(commands))
