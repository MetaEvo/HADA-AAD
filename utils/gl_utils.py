"""
Utils for generate_loop.py
"""

import fnmatch
import json
import math
import os
import random
import re
import shutil

import numpy as np

from utils.common import read_file
from utils.constants import REPO_NAME
from utils.docker_utils import copy_to_container, log_container_output, safe_log
from utils.domain_utils import (
    can_domain_ensembled,
    get_domain_score_key,
    get_domain_splits,
    get_domain_stagedeval_frac,
)
from utils.git_utils import commit_repo, get_git_commit_hash


def is_starting_node(genid):
    # Starting nodes are initial or 0
    return genid == "initial" or genid == 0


def get_saved_score(domain, output_dir, genid, split="train", type="agent"):
    # Type can be "agent" or "ensemble" or "max"
    agent_score = get_score(domain, output_dir, genid, split=split)
    ensemble_score = get_saved_ensemble_score(domain, output_dir, genid, split=split)

    # If full eval is not run, adjust score
    run_full_eval = get_node_metadata_key(output_dir, genid, "run_full_eval")
    if genid == "initial" or (run_full_eval is None or not run_full_eval):
        stagedeval_frac = get_domain_stagedeval_frac(domain)
        agent_score = agent_score * stagedeval_frac if agent_score is not None else None
        ensemble_score = (
            ensemble_score * stagedeval_frac if ensemble_score is not None else None
        )

    # Get score based on type
    if type == "agent":
        score = agent_score
    elif type == "ensemble":
        score = ensemble_score
    elif type == "max":
        if agent_score is not None and ensemble_score is not None:
            score = max(agent_score, ensemble_score)
        elif agent_score is not None:
            score = agent_score
        elif ensemble_score is not None:
            score = ensemble_score
        else:
            score = None
    else:
        raise ValueError(f"Unknown type '{type}'")
    return score


def get_score(domain, output_dir, genid, split="train"):
    # Get score from eval file
    eval_dirname = f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}"
    eval_file = os.path.join(output_dir, f"gen_{genid}/{eval_dirname}/report.json")
    score_key = get_domain_score_key(domain)
    try:
        with open(eval_file, "r") as f:
            eval_results = json.load(f)
        score = eval_results[score_key]
        # Filter nan values
        if math.isnan(score):
            score = None

        if "balrog" in domain:
            # Check if evals on environments ran
            if len(eval_results["environments"]) <= 0:
                return None
            # Normalize score
            return score / 100.0

        if "polyglot" in domain:
            # Check if any task was evaled without error
            noerror_tasks = eval_results["total_unresolved_ids"] + eval_results["total_emptypatch_ids"] + eval_results["total_resolved_ids"]
            if len(noerror_tasks) <= 0:
                return None
        print(f"score = {score} for genid {genid} on domain {domain} split {split}")
        return score
    except Exception:
        return None  # If score is missing or file not found


def get_saved_ensemble_score(domain, output_dir, genid, split="train"):
    # Get score from eval file for ensemble
    eval_file = os.path.join(
        output_dir, f"gen_{genid}/report_ensemble_{domain}_{split}.json"
    )
    score_key = get_domain_score_key(domain)
    try:
        with open(eval_file, "r") as f:
            eval_results = json.load(f)
        return eval_results[score_key]
    except Exception:
        return None  # If score is missing or file not found


def get_parent_genid(output_dir, genid):
    folder_prefix = "gen"
    metadata_file = os.path.join(output_dir, f"{folder_prefix}_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return None
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata.get("parent_genid", None)


def get_patch_files(output_dir, genid, visited=None):
    """递归获取所有祖先节点的 patch files。
    
    从当前节点开始，递归收集所有祖先节点的 hyper_agent_patch.diff 和 all_patch.diff。
    确保子节点继承父节点的所有 patches，包括父节点的父节点的 patches。
    
    Args:
        output_dir: 输出目录路径
        genid: 当前代际 ID 或 "initial"
        visited: 已访问的节点集合（防止循环）
    
    Returns:
        list: 所有祖先节点的 patch 文件路径列表（按从老到新的顺序）
    """
    if visited is None:
        visited = set()
    
    # 防止循环访问
    if genid in visited or genid == "initial" or genid is None:
        return []
    visited.add(genid)
    
    folder_prefix = "gen"
    metadata_file = os.path.join(output_dir, f"{folder_prefix}_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return []
    
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    # 获取当前节点的 patches
    curr_patches = metadata.get("curr_patch_files", [])
    
    # 递归获取父节点的 patches
    parent_genid = metadata.get("parent_genid")
    parent_patches = get_patch_files(output_dir, parent_genid, visited)
    
    # 合并：父节点的 patches + 当前节点的 patches
    # 使用 dict.fromkeys 去重，保持顺序
    all_patches = parent_patches + curr_patches
    unique_patches = list(dict.fromkeys(all_patches))
    
    return unique_patches


def update_node_metadata(output_dir, genid, data_update):
    # Load metadata from genid
    folder_prefix = "gen"
    metadata_file = os.path.join(output_dir, f"{folder_prefix}_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    # Update metadata
    metadata.update(data_update)
    # Save metadata
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)


def get_node_metadata_key(output_dir, genid, key):
    # Load metadata from genid
    folder_prefix = "gen"
    metadata_file = os.path.join(output_dir, f"{folder_prefix}_{genid}/metadata.json")
    if not os.path.exists(metadata_file):
        return None
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata.get(key, None)


def update_and_save_archive(output_dir, archive, new_node):
    # Update archive
    archive.append(new_node)
    # Save archive
    archive_file = os.path.join(output_dir, "archive.jsonl")
    with open(archive_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "current_genid": new_node,
                    "archive": archive,
                },
            )
            + "\n"
        )
    # Return updated archive
    return archive


def load_archive_data(filepath, last_only=True):
    # Load all archives from given metadata file
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found at {filepath}")
    # Read all JSON entries from the metadata file
    content = read_file(filepath)
    json_entries = content.split("\n{")
    # Parse all JSON entries
    archive_data = []
    for json_entry in json_entries:
        # Add back the { if it was removed by split
        if not json_entry.startswith("{"):
            json_entry = "{" + json_entry
        # Parse the JSON entry
        metadata = json.loads(json_entry)
        archive_data.append(metadata)
    # Return the last entry
    if last_only:
        return archive_data[-1]
    # Return all entries
    return archive_data


def get_archive_len(output_dir):
    # Get number of generations from archive
    archive_file = os.path.join(output_dir, "archive.jsonl")
    if not os.path.exists(archive_file):
        return 0
    archive_data = load_archive_data(archive_file, last_only=True)
    return len(archive_data.get("archive", []))


def setup_initial_gen(
    output_dir,
    domains,
    copy_root_dir=None,
    subsets=[],
    resume=False,
    copy_eval=True,
    optimize_option="only_agent",
    run_baseline=None,
    eval_test=False,
    edit_select_parent=False,
):
    # Resume from previous run
    if resume:
        root_dir = os.path.abspath(os.path.join(output_dir, f"gen_initial/{REPO_NAME}"))
        commit_hash = get_git_commit_hash(root_dir)
        return root_dir, commit_hash

    # Make a copy of the eval folder
    if copy_eval:
        for domain, subset in zip(domains, subsets):
            splits = get_domain_splits(domain, eval_test=eval_test)
            # Eval on training set
            gen_output_dir = os.path.join(output_dir, f"gen_initial/{domain}_eval")
            shutil.copytree(
                f"./outputs/initial_{domain}{subset}_0/",
                gen_output_dir,
                dirs_exist_ok=True,
            )
            # Eval on validation set
            if "val" in splits:  # pyright: ignore
                gen_output_dir = os.path.join(
                    output_dir, f"gen_initial/{domain}_eval_val"
                )
                shutil.copytree(
                    f'./outputs/initial_{domain}{subset.replace("_train", "_val")}_0/',
                    gen_output_dir,
                    dirs_exist_ok=True,
                )
            # Eval on test set
            if "test" in splits:  # pyright: ignore
                gen_output_dir = os.path.join(
                    output_dir, f"gen_initial/{domain}_eval_test"
                )
                shutil.copytree(
                    f'./outputs/initial_{domain}{subset.replace("_train", "_test")}_0/',
                    gen_output_dir,
                    dirs_exist_ok=True,
                )

    # Make a copy of the repo
    root_dir = os.path.abspath(os.path.join(output_dir, f"gen_initial/{REPO_NAME}"))
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Define exclusion criteria
    excluded_dirs = {
        ".claude",
        "outputs",
        "analysis",
        "misc",
        "baselines",
        "domains",
    }
    excluded_files = {
        "Dockerfile",
        ".dockerignore",
        "setup_initial.sh",
        "LICENSE.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
    }
    if "polyglot" not in domains:
        excluded_files.add("run_task_agent.py")
    excluded_patterns = ["venv*", "__pycache__*", "*.png", "outputs_os*"]
    if "ensemble" not in optimize_option or not any(can_domain_ensembled(d) for d in domains) and not copy_root_dir:
        excluded_patterns.append("*ensemble*")
    if not edit_select_parent and not copy_root_dir:
        excluded_patterns.append("*select_next_parent*")

    # Define exclusion criteria for domains
    excluded_dirs_domains = {
        "search_arena/saved",
        "polyglot/polyglot_benchmark_metadata.json",
        "polyglot/polyglot-benchmark",
        "polyglot/predictions",
        "polyglot/SWE-bench",
        "polyglot/logs",
    }
    excluded_dirs_domains.update({  # exclude domains that are not in the current run
        f"{folder_name}"
        for folder_name in os.listdir("./domains")
        if os.path.isdir(os.path.join("./domains", folder_name)) and not any(folder_name in d for d in domains)
    })
    excluded_files_domains = {}
    excluded_patterns_domains = ["venv*", "__pycache__*", "*.png"]

    # Function to ignore specific files/directories
    source_root = os.path.abspath(copy_root_dir or "./")
    def ignore_function(src, names):
        ignored = []
        for name in names:
            full_path = os.path.join(src, name)
            rel = os.path.relpath(full_path, start=source_root).replace(os.sep, "/")
            if any(rel == d or rel.startswith(d.rstrip("/") + "/") for d in excluded_dirs):
                ignored.append(name)
            elif name in excluded_files and os.path.isfile(full_path):
                ignored.append(name)
            else:
                for pattern in excluded_patterns:
                    if fnmatch.fnmatch(name, pattern):
                        ignored.append(name)
                        break
        return ignored
    source_root_domains = "./domains"
    def ignore_function_domains(src, names):
        ignored = []
        for name in names:
            full_path = os.path.join(src, name)
            rel = os.path.relpath(full_path, start=source_root_domains).replace(os.sep, "/")
            if any(rel == d or rel.startswith(d.rstrip("/") + "/") for d in excluded_dirs_domains):
                ignored.append(name)
            elif name in excluded_files_domains and os.path.isfile(full_path):
                ignored.append(name)
            else:
                for pattern in excluded_patterns_domains:
                    if fnmatch.fnmatch(name, pattern):
                        ignored.append(name)
                        break
        return ignored

    # Copy the repo
    if copy_root_dir is not None:
        shutil.copytree(copy_root_dir, root_dir, dirs_exist_ok=True, ignore=ignore_function)
        shutil.copytree("./domains", os.path.join(root_dir, "domains"), dirs_exist_ok=True, ignore=ignore_function_domains)
    else:
        shutil.copytree("./", root_dir, dirs_exist_ok=True, ignore=ignore_function)
        shutil.copytree("./domains", os.path.join(root_dir, "domains"), dirs_exist_ok=True, ignore=ignore_function_domains)

    # Setup README.md
    readme_path = os.path.join(root_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        readme_desc = get_readme_description(
            ensemble="ensemble" in optimize_option,
            edit_select_parent=edit_select_parent,
        )
        f.write(readme_desc)

    # Setup files for dgm baseline
    if copy_root_dir is None and run_baseline and "dgm" in run_baseline:
        shutil.copyfile("./baselines/dgm/coding_agent.py", os.path.join(root_dir, "coding_agent.py"))
        os.remove(os.path.join(root_dir, "hyper_agent.py"))
        os.remove(os.path.join(root_dir, "run_hyper_agent.py"))

    # Get commit hash
    commit_hash = commit_repo(root_dir)

    # Return info
    return root_dir, commit_hash


def get_readme_description(ensemble=False, edit_select_parent=False):
    desc = """# Self-Improving AI

This system is designed to automatically produce agents for solving downstream tasks. The system iteratively improves the generated agents through code editing. To enable continuous improvement, the system should look at its code repository and the provided path to previously generated agents and their evaluation results, and then edit and enhance its own mechanisms for generating agents. This process creates a recursive loop of self-improvement.
"""
    if ensemble:
        desc += """\n## Optimize the Ensemble of Agents

Given a fixed archive of agents, optimize the performance of the ensemble without modifying the individual agents. The archive of agents is provided in `/tmp/agent_archive/`. You can edit and improve the ensemble logic in `ensemble.py`."""

    if edit_select_parent:
        desc += """\n## Parent Selection Mechanism

The parent selection mechanism should sample agents from the archive using a non-greedy, diversity-preserving strategy that allows weaker, novel, or niche agents to produce offspring. This enables the discovery of interesting stepping stones that can unlock larger future improvements. The goal is to maximize long-term innovation and avoid premature convergence, while still filtering out uninteresting or unproductive exploration paths to use compute efficiently. You can edit and improve the select parent logic in `select_next_parent.py`.

Note that:
- A node with no children does not mean that its path has not been explored. The node's lineage depth indicates the depth of exploration."""

    return desc

def filter_patch_by_files(patch_str, target_files):
    """
    Filters out the diff blocks related to any of the target_files in a patch string.

    Args:
        patch_str (str): The complete patch text.
        target_files (list[str]): A list of filenames for which to extract changes (e.g. ['affine_cipher.py', 'other.py']).

    Returns:
        str: A string containing only the diff blocks for the specified target files.
    """
    lines = patch_str.splitlines()
    filtered_lines = []
    include_block = False

    for line in lines:
        # When we encounter a new diff block header, check if the block is for any of the target files.
        if line.startswith("diff --git"):
            include_block = not any(f"a/{target}" in line and f"b/{target}" in line for target in target_files)
        if include_block:
            filtered_lines.append(line)
    return "\n".join(filtered_lines) + "\n"

def process_meta_patch_files(meta_patch_files, output_dir, reset_task_agent=False, reset_hyper_agent=False):
    new_meta_patch_files = []
    meta_patch_dir = os.path.join(output_dir, "meta_patch_files")
    os.makedirs(meta_patch_dir, exist_ok=True)

    # Process each patch file
    for i, patch_file in enumerate(meta_patch_files):
        patch_str = read_file(patch_file)
        # Filter out the diff blocks related to task_agent.py
        if reset_task_agent:
            patch_str = filter_patch_by_files(patch_str, ["task_agent.py"])
        # Filter out the diff blocks related to hyper_agent.py
        if reset_hyper_agent:
            patch_str = filter_patch_by_files(patch_str, ["hyper_agent.py"])
        # Write the patch to new file
        new_meta_patch_file = os.path.join(meta_patch_dir, f"hyper_agent_patch_{i}.diff")
        with open(new_meta_patch_file, "w") as f:
            f.write(patch_str)
        new_meta_patch_files.append(new_meta_patch_file)

    return new_meta_patch_files

def is_patch_already_applied(container, patch_content, repo_name=REPO_NAME):
    """
    Check if a patch has already been applied by testing it in dry-run mode.
    """
    import tempfile
    
    # Write patch to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(patch_content)
        patch_file = f.name

    try:
        # Copy patch to container
        from utils.docker_utils import copy_to_container
        copy_to_container(container, patch_file, f"/{repo_name}/test_patch.txt", verbose=False)
        
        # Test patch in dry-run mode
        exec_result = container.exec_run(
            f"/bin/sh -c 'patch -p1 --dry-run < /{repo_name}/test_patch.txt'",
            workdir=f"/{repo_name}",
        )
        
        # Clean up test file
        container.exec_run(f"rm /{repo_name}/test_patch.txt", workdir=f"/{repo_name}")
        
        # If dry-run succeeds (exit_code == 0), patch is NOT applied yet → return False
        # If dry-run fails (exit_code != 0), patch is already applied or conflicts → return True
        if exec_result.exit_code == 0:
            return False
        # Patch failed to apply - check if it's because it's already applied
        output = exec_result.output.decode().lower() if exec_result.output else ""
        return "reversed patch" in output or "previously applied" in output
        
    except Exception:
        return False
    finally:
        os.remove(patch_file)

def apply_diffs_container(container, patch_files, repo_name=REPO_NAME, verbose=True):
    # Apply all diffs
    patch_files = patch_files or []
    applied_count = 0
    
    for patch_file in patch_files:
        # Read and filter the patch to exclude domains/ folder changes
        patch_content = read_file(patch_file)
        filtered_patch = filter_patch_by_files(patch_content, ["domains/"])
        
        # Check if patch is already applied
        if is_patch_already_applied(container, filtered_patch, repo_name):
            safe_log(f"Patch already applied, skipping: {os.path.basename(patch_file)}", verbose=verbose)
            continue

        # Write filtered patch to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(filtered_patch)
            filtered_patch_file = f.name

        try:
            copy_to_container(container, filtered_patch_file, f"/{repo_name}/parent_patch.txt", verbose=verbose)
            
            # Apply patch with force flag to handle minor conflicts
            exec_result = container.exec_run(
                f"/bin/sh -c 'patch -p1 -f < /{repo_name}/parent_patch.txt'",
                workdir=f"/{repo_name}",
            )
            
            # Log output without raising exception for patch failures
            if isinstance(exec_result.output, bytes):
                safe_log(f"Container output: {exec_result.output.decode()}", verbose=verbose)
            else:
                for chunk in exec_result.output:
                    if chunk:
                        safe_log(f"Container output: {chunk.decode().strip()}", verbose=verbose)
            
            # Check if patch applied successfully - don't raise exception on failure
            if exec_result.exit_code == 0:
                applied_count += 1
                safe_log(f"Successfully applied patch: {os.path.basename(patch_file)}", verbose=verbose)
            else:
                safe_log(f"WARNING: Failed to apply patch: {os.path.basename(patch_file)} (exit code {exec_result.exit_code}). Continuing anyway...", verbose=verbose)
                
            exec_result = container.exec_run(
                f"rm /{repo_name}/parent_patch.txt", workdir=f"/{repo_name}"
            )
            if isinstance(exec_result.output, bytes):
                safe_log(f"Container output: {exec_result.output.decode()}", verbose=verbose)
            else:
                for chunk in exec_result.output:
                    if chunk:
                        safe_log(f"Container output: {chunk.decode().strip()}", verbose=verbose)
        finally:
            os.remove(filtered_patch_file)

    safe_log(f"Applied {applied_count} out of {len(patch_files)} patch files", verbose=verbose)

    # Only stage and commit if patches were actually applied
    if applied_count > 0:
        # Stage all changes
        exec_result = container.exec_run("git add --all", workdir=f"/{repo_name}")
        log_container_output(exec_result, verbose=verbose)

        # Get Git user configuration from environment variables with defaults
        git_user_name = os.environ.get("GIT_USER_NAME", "HADA Bot")
        git_user_email = os.environ.get("GIT_USER_EMAIL", "bot@hada.ai")
        
        safe_log(f"Using Git config: {git_user_name} <{git_user_email}>", verbose=verbose)

        # Commit the changes
        exec_result = container.exec_run(
            f'git -c user.name="{git_user_name}" -c user.email="{git_user_email}" commit -m "a nonsense commit message"', workdir=f"/{repo_name}"
        )
        log_container_output(exec_result, verbose=verbose)

    # Skip Git status check entirely to avoid submodule issues
    # Always return the current HEAD commit hash
    exec_result = container.exec_run("git rev-parse HEAD", workdir=f"/{repo_name}/")
    log_container_output(exec_result, verbose=verbose)
    commit_hash = exec_result.output.decode("utf-8").strip()
    
    safe_log(f"Using current HEAD commit: {commit_hash}", verbose=verbose)

    # # Install requirements again in case of any changes
    # container.exec_run(
    #     cmd=[
    #         "/bin/bash",
    #         "-lc",
    #         "HTTPS_PROXY=http://fwdproxy:8080 "
    #         "HTTP_PROXY=http://fwdproxy:8080 "
    #         "FTP_PROXY=http://fwdproxy:8080 "
    #         "https_proxy=http://fwdproxy:8080 "
    #         "http_proxy=http://fwdproxy:8080 "
    #         "ftp_proxy=http://fwdproxy:8080 "
    #         "http_no_proxy='.facebook.com,.tfbnw.net,.fb.com' "
    #         f"python -m pip install -r '/{REPO_NAME}/requirements.txt'",
    #     ],
    #     workdir="/",
    # )
    # log_container_output(exec_result)

    return commit_hash


def select_parent(archive, output_dir, domains, method="best"):
    # Get candidate scores (averaged across domains)
    candidates = {}
    for genid in archive:
        # Skip non-valid parents
        valid_parent = (
            get_node_metadata_key(output_dir, genid, "valid_parent")
            if not is_starting_node(genid)
            else True
        )
        if not valid_parent:
            continue
        # Get per-domain scores
        per_domain_scores = []
        for dom in domains:
            split = "val" if "val" in get_domain_splits(dom) else "train"
            score = get_saved_score(dom, output_dir, genid, split=split, type="max")
            per_domain_scores.append(score)
        if per_domain_scores and all(score is not None for score in per_domain_scores):
            candidates[genid] = sum(per_domain_scores) / len(per_domain_scores)

    if not candidates:
        # Get the first initial node as the only candidate
        candidates[archive[0]] = 0.0
        # raise ValueError("No evaluation results found in archive.")

    # Build child counts from metadata
    child_counts = {genid: 0 for genid in candidates}
    for genid in archive:
        parent = get_parent_genid(output_dir, genid)
        if parent in child_counts:
            child_counts[parent] += 1

    # Select parent randomly
    if method == "random":
        return random.choice(list(candidates.keys()))

    # Select the latest compiled node
    elif method == "latest":
        return list(candidates.keys())[-1]

    # Select the best compiled node
    elif method == "best":
        return max(candidates, key=candidates.get)  # pyright: ignore[reportCallIssue]

    # Select the best compiled node with probability proportional to score
    elif method == "score_prop":
        commits = list(candidates.keys())
        scores = [candidates[commit] for commit in commits]
        total = sum(scores)
        probabilities = (
            [s / total for s in scores]
            if total > 0
            else [1 / len(scores)] * len(scores)
        )
        return random.choices(commits, weights=probabilities)[0]

    # Select the best compiled node with probability proportional to score and inversely proportional to number of children
    elif method == "score_child_prop":
        commits = list(candidates.keys())
        scores = [candidates[commit] for commit in commits]
        penalties = [math.exp(-(child_counts[commit]/8)**3) for commit in commits]
        combined = [s * p for s, p in zip(scores, penalties)]
        total = sum(combined)
        probabilities = (
            [c / total for c in combined]
            if total > 0
            else [1 / len(combined)] * len(combined)
        )
        return random.choices(commits, weights=probabilities)[0]

    else:
        raise ValueError(f"Unknown method '{method}'")

def get_latest_can_select_parent(archive, output_dir, trunc_genid=None):
    # Truncate archive
    if trunc_genid is not None:
        if is_starting_node(trunc_genid):
            return None
        archive = [genid for genid in archive if is_starting_node(genid) or genid < trunc_genid]

    # Get latest can_select_parent
    for genid in archive[::-1]:
        if is_starting_node(genid):
            return genid
        can_select_next_parent = get_node_metadata_key(output_dir, genid, "can_select_next_parent")
        if can_select_next_parent:
            return genid

    # Shouldn't reach here
    print("shouldn't reach here")
    return None

def run_commands_to_check_compilation(container, run_baseline=None, edit_select_parent=False):
    # Run commands to check if the agents are compilable
    if run_baseline and "dgm" in run_baseline:
        command = [
            "timeout",
            "300",  # 5m timeout
            "python",
            "-c",
            "from coding_agent import CodingAgent",
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        if exec_result.exit_code != 0:
            raise Exception("coding_agent is not compilable")

    else:
        command = [
            "timeout",
            "300",  # 5m timeout
            "python",
            "-c",
            "from hyper_agent import HyperAgent",
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        if exec_result.exit_code != 0:
            raise Exception("hyper_agent is not compilable")

    command = [
        "timeout",
        "300",  # 5m timeout
        "python",
        "-c",
        "from task_agent import TaskAgent",
    ]
    exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
    log_container_output(exec_result)
    if exec_result.exit_code != 0:
        raise Exception("task_agent is not compilable")

    if edit_select_parent:
        command = [
            "timeout",
            "300",  # 5m timeout
            "python",
            "-c",
            "from select_next_parent import select_next_parent",
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        if exec_result.exit_code != 0:
            raise Exception("select_next_parent is not compilable")

def collect_previous_gen_info(output_dir, parent_genid, domains):
    """
    收集前代的评估信息，用于传递给当前代的 task agent。
    
    Args:
        output_dir: 输出目录
        parent_genid: 父代 ID
        domains: 域列表
    
    Returns:
        dict: 每个域对应的前代信息，格式为 {domain: info_dict}
    """
    if is_starting_node(parent_genid):
        return None
    
    prev_gen_info = {}
    
    for domain in domains:
        # 构建前代评估结果路径
        prev_eval_folder = os.path.join(output_dir, f"gen_{parent_genid}", f"{domain}_eval")
        # predictions.csv 在 eval 根目录，不在 agent_evals 子目录
        predictions_path = os.path.join(prev_eval_folder, "predictions.csv")
        
        if not os.path.exists(predictions_path):
            safe_log(f"Previous predictions not found for {domain} gen_{parent_genid}")
            continue
        
        try:
            # 读取前代预测结果
            import pandas as pd
            predictions_df = pd.read_csv(predictions_path)
            
            # 提取关键信息
            domain_info = {
                "generation": parent_genid,
                "total_problems": len(predictions_df),
                "predictions": predictions_df.to_dict(orient="records"),
            }
            
            # 读取 generate.log（包含 task agent 运行日志和错误信息）
            generate_log_path = os.path.join(os.path.dirname(prev_eval_folder), "generate.log")
            if os.path.exists(generate_log_path):
                with open(generate_log_path, 'r') as f:
                    generate_log_content = f.read()
                    domain_info["generate_log"] = generate_log_content
                safe_log(f"Collected generate.log for {domain} gen_{parent_genid}")
            
            # 读取前代报告（如果存在）
            report_path = os.path.join(os.path.dirname(prev_eval_folder), "report.json")
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                    domain_info["report"] = report_data
            
            # 读取 patch 文件（如果存在）
            patch_path = os.path.join(prev_eval_folder, "all_patch.diff")
            if os.path.exists(patch_path) and os.path.getsize(patch_path) > 0:
                with open(patch_path, 'r') as f:
                    domain_info["patch"] = f.read()
            
            prev_gen_info[domain] = domain_info
            safe_log(f"Collected previous gen info for {domain} gen_{parent_genid}: {len(predictions_df)} problems")
            
        except Exception as e:
            safe_log(f"Error collecting previous gen info for {domain}: {e}")
            continue
    
    return prev_gen_info if prev_gen_info else None