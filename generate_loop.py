import sys
import warnings

warnings.filterwarnings("ignore", message="Event loop is closed")

sys.path.insert(0, '/hada')
from config import set_global_seed, SEED, MODEL_NAME

set_global_seed()

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import docker
from analysis.plot_progress import plot_progress_single, plot_progress_together
from analysis.visualize_archive import (
    visualize_archive_single,
    visualize_archive_together,
)

from utils.common import file_exist_and_not_empty, load_json_file
from utils.constants import REPO_NAME
from utils.docker_utils import (
    build_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    safe_log,
    setup_logger,
)
from utils.domain_utils import (
    can_domain_ensembled,
    get_domain_eval_subset,
    get_domain_splits,
)
from utils.gl_utils import (
    apply_diffs_container,
    get_patch_files,
    get_score,
    load_archive_data,
    run_commands_to_check_compilation,
    select_parent,
    setup_initial_gen,
    update_and_save_archive,
    update_node_metadata,
    get_latest_can_select_parent,
    is_starting_node,
    collect_previous_gen_info,
)


def select_next_parent_container(
    docker_client,
    domains,
    generate_output_dir,
    archive,
    root_dir="./",
    root_commit="HEAD",
    max_attempts=10,
):
    logger = setup_logger(os.path.join(generate_output_dir, "select_next_parent.log"))

    latest_node = get_latest_can_select_parent(archive, generate_output_dir)
    safe_log(f"select_next_parent_container: latest_node={latest_node}")
    prev_patch_files = get_patch_files(generate_output_dir, latest_node)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container_name = f"{REPO_NAME}-nextparent-container-{run_id}"
    container = build_container(docker_client, root_dir, "hada", container_name, verbose=False, domains=domains)
    container.start()
    container_output_folder = "/tmp/"

    try:
        commit_hash = apply_diffs_container(container, prev_patch_files, verbose=False)

        container_generate_output_dir = os.path.join(
            container_output_folder, generate_output_dir.split(os.sep)[-1]
        )
        copy_to_container(
            container,
            source_path=generate_output_dir,
            dest_path=container_generate_output_dir,
            verbose=False,
        )

        command = [
            "timeout",
            "7200",
            "python",
            "-m",
            "utils.run_select_next_parent",
            "--domains",
            *domains,
            "--generate_output_dir",
            container_generate_output_dir,
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result, verbose=True)

        container_output_strings = exec_result.output.decode().strip().split("\n")
        next_parent_genid = container_output_strings[-1]
        next_parent_genid = int(next_parent_genid) if not is_starting_node(next_parent_genid) else next_parent_genid

    except Exception as e:
        safe_log(f"Error in select_next_parent_container: {e}")
        update_node_metadata(generate_output_dir, latest_node, {"can_select_next_parent": False})
        next_parent_genid = None

    finally:
        exec_result = container.exec_run(
            cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result, verbose=False)
        exec_result = container.exec_run(
            cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result, verbose=False)

        cleanup_container(container, verbose=False)

    if next_parent_genid is None:
        if max_attempts > 0:
            next_parent_genid = select_next_parent_container(
                docker_client,
                domains,
                generate_output_dir,
                archive,
                root_dir,
                root_commit,
                max_attempts=max_attempts - 1,
            )
        else:
            raise Exception("Max attempts reached in select_next_parent_container")

    return next_parent_genid


def get_ensemble_scores_container(
    docker_client,
    domain,
    generate_output_dir,
    gen_output_dir,
    root_dir="./",
    root_commit="HEAD",
    prev_patch_files=[],
    num_samples=-1,
    max_workers=5,
    subsets=[],
):
    logger = setup_logger(os.path.join(gen_output_dir, "ensemble.log"))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container_name = f"{REPO_NAME}-ens-container-{run_id}"
    container = build_container(docker_client, root_dir, "hada", container_name, domains=[domain])
    container.start()
    container_output_folder = "/tmp/"

    try:
        commit_hash = apply_diffs_container(container, prev_patch_files)

        container_generate_output_dir = os.path.join(
            container_output_folder, generate_output_dir.split(os.sep)[-1]
        )
        copy_to_container(
            container,
            source_path=generate_output_dir,
            dest_path=container_generate_output_dir,
        )

        scores = []
        for subset in subsets:
            command = [
                "timeout",
                "10800",
                "python",
                "-m",
                "utils.run_ensemble",
                "--domain",
                domain,
                "--generate_output_dir",
                container_generate_output_dir,
                "--num_samples",
                str(num_samples),
                "--max_workers",
                str(max_workers),
                "--subset",
                subset,
            ]
            exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
            log_container_output(exec_result)

            container_output_strings = exec_result.output.decode().strip().split("\n")
            score = float(container_output_strings[-3])
            scores.append(score)
            container_predictions_path = container_output_strings[-2]
            container_report_path = container_output_strings[-1]

            predictions_file = os.path.basename(container_predictions_path)
            report_file = os.path.basename(container_report_path)
            local_predictions_path = os.path.join(gen_output_dir, predictions_file)
            local_report_path = os.path.join(gen_output_dir, report_file)
            copy_from_container(
                container,
                source_path=container_predictions_path,
                dest_path=local_predictions_path,
            )
            copy_from_container(
                container,
                source_path=container_report_path,
                dest_path=local_report_path,
            )

    except Exception as e:
        safe_log(f"Error in get_ensemble_scores_container: {e}")
        scores = [None] * len(subsets)

    finally:
        exec_result = container.exec_run(
            cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)
        exec_result = container.exec_run(
            cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)

        cleanup_container(container)

    return scores


def eval_produced_agent(
    container,
    container_output_folder,
    gen_output_dir,
    domain,
    eval_samples=-1,
    eval_workers=10,
    eval_subset="_filtered_100_train",
    eval_test=False,
    prev_gen_info=None,
):
    splits = get_domain_splits(domain, eval_test=eval_test)

    for split in splits:
        safe_log(f"Evaluating the produced agent on {domain} {eval_samples} {split}...")

        eval_run_id = f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}"

        container_evaloutput_folder = os.path.join(container_output_folder, eval_run_id)

        agent_path = "./task_agent.py"
        
        if domain in ['metabox_mo']:
            subset_val = eval_subset.replace('_train', f'_{split}')
            
            container_prev_info_path = None
            if prev_gen_info and domain in prev_gen_info:
                local_prev_info_file = os.path.join(gen_output_dir, f"prev_gen_info_{domain}.json")
                with open(local_prev_info_file, 'w') as f:
                    json.dump(prev_gen_info.get(domain), f)
                safe_log(f"Saved prev_gen_info to {local_prev_info_file}")
                
                container_prev_info_path = f"/{REPO_NAME}/prev_gen_info_{domain}.json"
                from utils.docker_utils import copy_to_container
                copy_to_container(container, source_path=local_prev_info_file, dest_path=container_prev_info_path)
                safe_log(f"Copied prev_gen_info to container: {container_prev_info_path}")
            
            harness_cmd = (
                'import sys, json, os; '
                'sys.path = ["/' + REPO_NAME + '"] + [p for p in sys.path if "metaevobox" not in p]; '
                'prev_info_file = "' + (container_prev_info_path if container_prev_info_path else '') + '"; '
                'prev_info = json.load(open(prev_info_file, "r")) if (prev_info_file and os.path.exists(prev_info_file)) else None; '
                'sys.argv = ["harness", '
                '"--agent_path", "' + agent_path + '", '
                '"--output_dir", "' + container_output_folder + '", '
                '"--run_id", "' + eval_run_id + '", '
                '"--domain", "' + domain + '", '
                '"--num_samples", "' + str(eval_samples) + '", '
                '"--num_workers", "' + str(eval_workers) + '", '
                '"--subset", "' + subset_val + '"]; '
                'from domains.harness import main; main()'
            )
            command = [
                "timeout",
                "999999",
                "python",
                "-c",
                harness_cmd,
            ]
        else:
            command = [
                "timeout",
                "18000",
                "python",
                "-m",
                "domains.harness",
                "--agent_path", agent_path,
                "--output_dir", container_output_folder,
                "--run_id", eval_run_id,
                "--domain", domain,
                "--num_samples", str(eval_samples),
                "--num_workers", str(eval_workers),
                "--subset", eval_subset.replace("_train", f"_{split}"),
            ]

        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)

        if domain in ['metabox_mo']:
            dname_val = os.path.join(container_output_folder, eval_run_id)
            report_cmd = (
                'import sys; '
                'sys.path = ["/' + REPO_NAME + '"] + [p for p in sys.path if "metaevobox" not in p]; '
                'sys.argv = ["report", '
                '"--domain", "' + domain + '", '
                '"--dname", "' + dname_val + '"]; '
                'from domains.report import main; main()'
            )
            command = [
                "timeout",
                "10800",
                "python",
                "-c",
                report_cmd,
            ]
        else:
            command = [
                "timeout",
                "10800",
                "python",
                "-m",
                "domains.report",
                "--domain", domain,
                "--dname", os.path.join(container_output_folder, eval_run_id),
            ]

        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)

        evaloutput_folder = os.path.join(gen_output_dir, eval_run_id)
        copy_from_container(
            container,
            source_path=container_evaloutput_folder,
            dest_path=evaloutput_folder,
        )


def copy_prev_eval_to_container(
    container,
    prev_eval_path,
    container_output_folder,
    current_genid=None,
    container_folder_name=None,
):
    if not os.path.exists(prev_eval_path):
        raise FileNotFoundError(f"Previous eval path not found: {prev_eval_path}")

    prev_eval_path = os.path.normpath(prev_eval_path)
    tail = os.path.join(*prev_eval_path.split(os.sep)[-1:])
    container_prev_eval_path = os.path.join(container_output_folder, tail)

    container.exec_run(["mkdir", "-p", container_output_folder], workdir="/")

    copy_to_container(
        container, source_path=prev_eval_path, dest_path=container_prev_eval_path
    )

    prune_cmds = [
        f"find '{container_prev_eval_path}' -type d -name 'gen_{current_genid}' -prune -exec rm -rf {{}} +",
        f"find '{container_prev_eval_path}' -type d -name '*_eval_val*' -prune -exec rm -rf {{}} +",
        f"find '{container_prev_eval_path}' -type d -name '*_eval_test*' -prune -exec rm -rf {{}} +",
        f"find '{container_prev_eval_path}' -type d -name '*{REPO_NAME}*' -prune -exec rm -rf {{}} +",
        f"find '{container_prev_eval_path}' -type f -name '*.pyc' -delete",
        f"find '{container_prev_eval_path}' -type f \\( -name '*_val' -o -name '*_val.*' -o -name '*_val_*' \\) -delete",
        f"find '{container_prev_eval_path}' -type f \\( -name '*_test' -o -name '*_test.*' -o -name '*_test_*' \\) -delete",
    ]

    for cmd in prune_cmds:
        exec_result = container.exec_run(["bash", "-lc", cmd], workdir="/")

    exec_result = container.exec_run(
        ["ls", "-l", container_prev_eval_path], workdir="/"
    )
    log_container_output(exec_result)

    if container_folder_name is not None:
        new_container_prev_eval_path = os.path.join(
            container_output_folder, container_folder_name
        )
        container.exec_run(
            ["mv", container_prev_eval_path, new_container_prev_eval_path], workdir="/"
        )
        log_container_output(exec_result)
        container_prev_eval_path = new_container_prev_eval_path

    return container_prev_eval_path


def generate(
    docker_client,
    domains,
    output_dir,
    run_id,
    current_genid,
    parent_genid,
    root_dir,
    root_commit="main",
    eval_samples=-1,
    eval_workers=10,
    eval_subsets="_filtered_100",
    run_hyper_agent=True,
    run_baseline=None,
    optimize_option="only_agent",
    agent_archive_path=None,
    eval_test=False,
    edit_select_parent=False,
    max_generation=None,
):
    prev_gen_dir = os.path.join(output_dir, f"gen_{parent_genid}")
    gen_output_dir = os.path.join(output_dir, f"gen_{current_genid}")
    os.makedirs(gen_output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(gen_output_dir, "generate.log"))
    
    metadata = {
        "gen_output_dir": gen_output_dir,
        "current_genid": current_genid,
        "parent_genid": parent_genid,
        "run_baseline": run_baseline,
        "prev_patch_files": [],
        "curr_patch_files": [],
        "parent_agent_success": not run_hyper_agent,
        "optimize_option": optimize_option,
        "agent_archive_path": agent_archive_path,
        "can_select_next_parent": True,
    }
    run_eval = not run_hyper_agent
    metadata["run_eval"] = run_eval
    print(metadata)

    container_name = f"{REPO_NAME}-gl-container-{run_id}"
    container = build_container(
        docker_client,
        root_dir,
        "hada",
        container_name,
        domains=domains,
    )
    container.start()
    container_output_folder = f"/{REPO_NAME}/outputs/"

    try:
        if run_baseline and "no_selfimprove" in run_baseline:
            donottouch_reponame = f"/DONOTTOUCH_{REPO_NAME}"
            exec_result = container.exec_run(
                cmd=["cp", "-r", f"/{REPO_NAME}", donottouch_reponame],
                workdir=f"/",
            )
            log_container_output(exec_result)

        patch_files = get_patch_files(output_dir, parent_genid)
        metadata["prev_patch_files"] += patch_files
        commit_hash = apply_diffs_container(container, patch_files)

        if run_hyper_agent:
            if run_baseline and "dgm" in run_baseline:
                from baselines.dgm.utils import get_problem_statement
                problem_statement = get_problem_statement(
                    root_dir, output_dir, parent_genid, domains,
                    customized="custom" in run_baseline,
                )

            else:
                if optimize_option == "only_ensemble":
                    container_agent_archive_path = copy_prev_eval_to_container(
                        container,
                        agent_archive_path,
                        container_output_folder,
                        current_genid=current_genid,
                        container_folder_name="agent_archive",
                    )

                if run_baseline == "no_archive":
                    container_prev_eval_path = os.path.join(
                        container_output_folder, *prev_gen_dir.split(os.sep)[-2:]
                    )
                    copy_to_container(
                        container,
                        source_path=prev_gen_dir,
                        dest_path=container_prev_eval_path,
                    )
                else:
                    container_prev_eval_path = copy_prev_eval_to_container(
                        container, output_dir, container_output_folder, current_genid=current_genid,
                    )

            safe_log("Running meta agent...")
            container_agentoutput_folder = os.path.join(
                container_output_folder, "agent_output"
            )
            container_chat_history_file = os.path.join(
                container_agentoutput_folder, "hyper_agent_chat_history.md"
            )
            
            if run_baseline and "dgm" in run_baseline:
                command = [
                    "timeout",
                    "21600",
                    "python",
                    "coding_agent.py",
                    "--problem_statement",
                    problem_statement,
                    "--chat_history_file",
                    container_chat_history_file,
                    "--git_dir",
                    f"/{REPO_NAME}",
                    "--base_commit",
                    commit_hash,
                    "--outdir",
                    container_agentoutput_folder,
                ]
            else:
                command = [
                    "timeout",
                    "21600",
                    "python",
                    "run_hyper_agent.py",
                    "--chat_history_file",
                    container_chat_history_file,
                    "--repo_path",
                    f"/{REPO_NAME}/",
                    "--evals_folder",
                    container_prev_eval_path,
                    "--git_dir",
                    f"/{REPO_NAME}",
                    "--base_commit",
                    commit_hash,
                    "--outdir",
                    container_agentoutput_folder,
                    "--iterations_left",
                    str(max_generation - current_genid),
                    "--domains",
                    ",".join(domains),
                ]

            run_workdir = (
                f"/DONOTTOUCH_{REPO_NAME}"
                if run_baseline and "no_selfimprove" in run_baseline
                else f"/{REPO_NAME}"
            )
            exec_result = container.exec_run(cmd=command, workdir=run_workdir)
            log_container_output(exec_result)
            metadata["parent_agent_success"] = exec_result.exit_code == 0

            local_agentoutput_folder = os.path.join(gen_output_dir, "agent_output/")
            copy_from_container(
                container,
                source_path=container_agentoutput_folder,
                dest_path=local_agentoutput_folder,
            )

            hyper_agent_patch_path = os.path.join(local_agentoutput_folder, "hyper_agent_patch.diff")
            if os.path.exists(hyper_agent_patch_path) and os.path.getsize(hyper_agent_patch_path) > 0:
                safe_log(f"Applying hyper_agent_patch.diff to container before Task Agent runs...")
                _ = apply_diffs_container(container, [hyper_agent_patch_path])
                safe_log("hyper_agent_patch.diff applied successfully")
                metadata["curr_patch_files"].append(hyper_agent_patch_path)
                safe_log(f"Added hyper_agent_patch.diff to curr_patch_files")
            else:
                safe_log("No hyper_agent_patch.diff generated, skipping apply")

            run_eval = True
            metadata["run_eval"] = run_eval

            run_commands_to_check_compilation(container, run_baseline=run_baseline, edit_select_parent=edit_select_parent)

        if run_eval and "agent" in optimize_option:
            log_path = os.path.join(gen_output_dir, "generate.log")
            
            prev_gen_info = collect_previous_gen_info(output_dir, parent_genid, domains)

            def eval_agent_worker(domain, eval_subset, eval_n):
                setup_logger(log_path)
                eval_produced_agent(
                    container,
                    container_output_folder,
                    gen_output_dir,
                    domain=domain,
                    eval_samples=eval_n,
                    eval_workers=eval_workers,
                    eval_subset=eval_subset,
                    eval_test=eval_test,
                    prev_gen_info=prev_gen_info.get(domain) if prev_gen_info else None,
                )

            run_next_eval = True
            if run_next_eval:
                if isinstance(eval_samples, int):
                    _per_domain_eval_samples = [eval_samples] * len(domains)
                else:
                    _per_domain_eval_samples = eval_samples
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(eval_agent_worker, d, s, n)
                        for d, s, n in zip(
                            domains, eval_subsets, _per_domain_eval_samples
                        )
                    ]
                    try:
                        for f in futures:
                            f.result()
                    except Exception as e:
                        for future in futures:
                            if not future.done():
                                future.cancel()
                        raise
                metadata["run_full_eval"] = True
                
                for domain in domains:
                    all_patch_path = os.path.join(
                        gen_output_dir, f"{domain}_eval", "agent_evals", "all_patch.diff"
                    )
                    if os.path.exists(all_patch_path) and os.path.getsize(all_patch_path) > 0:
                        metadata["curr_patch_files"].append(all_patch_path)
                        safe_log(f"Added all_patch.diff for {domain} to curr_patch_files")

    except Exception as e:
        safe_log(f"Error in generate: {e}")
        safe_log(f"Error type: {type(e).__name__}")
        safe_log(f"domains type: {type(domains)}, value: {domains}")
        safe_log(f"parent_genid type: {type(parent_genid)}, value: {parent_genid}")
        import traceback
        safe_log(f"Traceback: {traceback.format_exc()}")
        metadata["run_eval"] = False

    finally:
        exec_result = container.exec_run(
            cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)
        exec_result = container.exec_run(
            cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
        )
        log_container_output(exec_result)

        cleanup_container(container)

        eval_successful = all(
            [
                get_score(domain, output_dir, current_genid) is not None
                for domain in domains
            ]
        )
        metadata["valid_parent"] = metadata["run_eval"] and eval_successful
        with open(os.path.join(gen_output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    return metadata


def generate_loop(
    domains,
    run_id=None,
    max_generation=30,
    eval_samples=-1,
    eval_workers=5,
    eval_subsets=[],
    parent_selection="score_prop",
    resume_from=None,
    resume_from_run=None,
    output_dir_parent=None,
    reset_task_agent=False,
    reset_hyper_agent=False,
    copy_root_dir=None,
    run_baseline=None,
    optimize_option="only_agent",
    agent_archive_path=None,
    eval_test=False,
    edit_select_parent=False,
):
    docker_client = docker.DockerClient()
    parent_selection = "latest" if run_baseline == "no_archive" else parent_selection
    
    if resume_from_run:
        temp_log_dir = os.path.join(os.getcwd(), "outputs", "temp")
        os.makedirs(temp_log_dir, exist_ok=True)
        setup_logger(os.path.join(temp_log_dir, "resume.log"))
        
        resume_from_run = os.path.normpath(os.path.abspath(resume_from_run))
        parts = resume_from_run.split(os.sep)
        gen_folder = parts[-1]
        run_folder = parts[-2]
        resume_genid = gen_folder.replace("gen_", "")
        
        output_dir = os.sep.join(parts[:-2]) + os.sep + run_folder
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        run_id = run_folder.split("generate_")[-1]
        safe_log(f"Resuming from run: {resume_from_run}")
        safe_log(f"Output dir: {output_dir}, Resume genid: {resume_genid}")
        
        gen_dir = os.path.join(output_dir, f"gen_{resume_genid}")
        metadata_file = os.path.join(gen_dir, "metadata.json")
        hyper_agent_patch_path = os.path.join(gen_dir, "agent_output", "hyper_agent_patch.diff")
        all_patch_path = None
        for item in os.listdir(gen_dir):
            if item.endswith("_eval"):
                candidate = os.path.join(gen_dir, item, "agent_evals", "all_patch.diff")
                if os.path.exists(candidate):
                    all_patch_path = candidate
                    break
        if all_patch_path is None:
            all_patch_path = os.path.join(gen_dir, f"{domains[0]}_eval", "agent_evals", "all_patch.diff")
        
        has_hyper_agent_patch = os.path.exists(hyper_agent_patch_path) and os.path.getsize(hyper_agent_patch_path) > 0
        has_all_patch = os.path.exists(all_patch_path) and os.path.getsize(all_patch_path) > 0
        has_metadata = os.path.exists(metadata_file)
        
        if has_metadata:
            with open(metadata_file, "r") as f:
                resume_metadata = json.load(f)
            parent_genid = resume_metadata.get("parent_genid")
        else:
            parent_genid = None
        
        safe_log(f"Has hyper_agent_patch.diff: {has_hyper_agent_patch}")
        safe_log(f"Has all_patch.diff: {has_all_patch}")
        
        new_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        new_output_dir_parent = os.path.join(os.getcwd(), "outputs/")
        new_output_dir = os.path.normpath(
            os.path.join(new_output_dir_parent, f"generate_{new_run_id}/")
        )
        os.makedirs(new_output_dir, exist_ok=True)
        
        shutil.copy2(
            os.path.join(output_dir, "archive.jsonl"),
            os.path.join(new_output_dir, "archive.jsonl")
        )
        
        for item in os.listdir(output_dir):
            if item.startswith("gen_"):
                src = os.path.join(output_dir, item)
                dst = os.path.join(new_output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
        
        output_dir = new_output_dir
        run_id = new_run_id
        
        old_output_dir = os.sep.join(parts[:-2]) + os.sep + run_folder
        if not os.path.isabs(old_output_dir):
            old_output_dir = os.path.join(os.getcwd(), old_output_dir)
        
        safe_log(f"DEBUG: old_output_dir = {old_output_dir}")
        safe_log(f"DEBUG: new_output_dir = {new_output_dir}")
        safe_log(f"DEBUG: items in new_output_dir = {os.listdir(new_output_dir)}")
        
        for item in os.listdir(new_output_dir):
            if item.startswith("gen_"):
                metadata_file = os.path.join(new_output_dir, item, "metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        gen_metadata = json.load(f)
                    
                    safe_log(f"DEBUG: Checking {item}/metadata.json")
                    safe_log(f"DEBUG: prev_patch_files = {gen_metadata.get('prev_patch_files', [])}")
                    safe_log(f"DEBUG: curr_patch_files = {gen_metadata.get('curr_patch_files', [])}")
                    
                    updated = False
                    if "prev_patch_files" in gen_metadata:
                        new_prev_files = []
                        for old_path in gen_metadata["prev_patch_files"]:
                            if old_output_dir in old_path:
                                new_path = old_path.replace(old_output_dir, new_output_dir)
                                new_prev_files.append(new_path)
                                updated = True
                                safe_log(f"Updated {item}/metadata.json prev_patch_files: {old_path} -> {new_path}")
                            else:
                                new_prev_files.append(old_path)
                        gen_metadata["prev_patch_files"] = new_prev_files
                    
                    if "curr_patch_files" in gen_metadata:
                        new_curr_files = []
                        for old_path in gen_metadata["curr_patch_files"]:
                            if old_output_dir in old_path:
                                new_path = old_path.replace(old_output_dir, new_output_dir)
                                new_curr_files.append(new_path)
                                updated = True
                                safe_log(f"Updated {item}/metadata.json curr_patch_files: {old_path} -> {new_path}")
                            else:
                                new_curr_files.append(old_path)
                        gen_metadata["curr_patch_files"] = new_curr_files
                    
                    if updated:
                        with open(metadata_file, "w") as f:
                            json.dump(gen_metadata, f, indent=4)
                        safe_log(f"Updated {item}/metadata.json with new paths")
                    else:
                        safe_log(f"DEBUG: No paths updated for {item}/metadata.json")
        
        if has_metadata and "prev_patch_files" in resume_metadata:
            updated_prev_patch_files = []
            for old_path in resume_metadata["prev_patch_files"]:
                if old_output_dir in old_path:
                    new_path = old_path.replace(old_output_dir, new_output_dir)
                    updated_prev_patch_files.append(new_path)
                    safe_log(f"Updated patch path: {old_path} -> {new_path}")
                else:
                    updated_prev_patch_files.append(old_path)
            resume_metadata["prev_patch_files"] = updated_prev_patch_files
            safe_log(f"Updated prev_patch_files paths from old run to new run")
        
        next_genid = int(resume_genid) + 1 if resume_genid != "initial" else 0
        
        if has_hyper_agent_patch and not has_all_patch:
            safe_log(f"Resuming from gen_{resume_genid}: Meta Agent done, running Task Agent...")
            start_from_hyper_agent = False
            resume_genid_int = int(resume_genid) if resume_genid != "initial" else 0
        else:
            safe_log(f"Resuming from gen_{resume_genid}: Running Meta Agent...")
            start_from_hyper_agent = True
            resume_genid_int = int(resume_genid) if resume_genid != "initial" else 0
        
        root_dir, root_commit = setup_initial_gen(
            output_dir,
            domains,
            copy_root_dir=copy_root_dir,
            subsets=eval_subsets,
            resume=True,
            optimize_option=optimize_option,
            run_baseline=run_baseline,
            eval_test=eval_test,
            edit_select_parent=edit_select_parent,
        )
        
        archive = load_archive_data(
            os.path.join(output_dir, "archive.jsonl"), last_only=True
        )["archive"]
        
        if resume_genid == "initial":
            start_genid = 0
        else:
            start_genid = int(resume_genid) + 1
        
        if has_hyper_agent_patch and not has_all_patch:
            parent_genid_for_resume = resume_metadata.get("parent_genid") if has_metadata else None
            run_hyper_agent_for_resume = False
            safe_log(f"Skipping Meta Agent for gen_{resume_genid}, running Task Agent directly")
            safe_log(f"Parent genid from metadata: {parent_genid_for_resume}")
        else:
            run_hyper_agent_for_resume = True
            safe_log(f"Gen {resume_genid} fully completed, will resume from gen_{start_genid} with normal flow")
        
        if not run_hyper_agent_for_resume:
            resume_gen_dir = os.path.join(new_output_dir, f"gen_{resume_genid}")
            resume_metadata_file = os.path.join(resume_gen_dir, "metadata.json")
            if os.path.exists(resume_metadata_file):
                with open(resume_metadata_file, "r") as f:
                    resume_gen_meta = json.load(f)
                prev_patches = resume_gen_meta.get("prev_patch_files", [])
                safe_log(f"Resume prev_patch_files count: {len(prev_patches)}")
            else:
                prev_patches = []
            
            metadata = generate(
                docker_client,
                domains,
                output_dir,
                run_id,
                current_genid=int(resume_genid),
                parent_genid=parent_genid_for_resume,
                root_dir=root_dir,
                root_commit=root_commit,
                eval_samples=eval_samples,
                eval_workers=eval_workers,
                eval_subsets=eval_subsets,
                run_hyper_agent=False,
                run_baseline=run_baseline,
                optimize_option=optimize_option,
                agent_archive_path=agent_archive_path,
                eval_test=eval_test,
                edit_select_parent=edit_select_parent,
                max_generation=max_generation,
            )
            archive = update_and_save_archive(output_dir, archive, new_node=int(resume_genid))
            start_genid = int(resume_genid) + 1
        else:
            start_genid = int(resume_genid) + 1 if resume_genid != "initial" else 0
    else:
        run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") if run_id is None else run_id
        )
        output_dir_parent = (
            os.path.join(os.getcwd(), "outputs/")
            if output_dir_parent is None
            else output_dir_parent
        )
        output_dir = os.path.normpath(
            os.path.join(output_dir_parent, f"generate_{run_id}/")
        )
        os.makedirs(output_dir, exist_ok=True)
        root_dir, root_commit = setup_initial_gen(
            output_dir,
            domains,
            copy_root_dir=copy_root_dir,
            subsets=eval_subsets,
            resume=False,
            optimize_option=optimize_option,
            run_baseline=run_baseline,
            eval_test=eval_test,
            edit_select_parent=edit_select_parent,
        )

        archive = update_and_save_archive(output_dir, [], new_node="initial")
        metadata = {
            "gen_output_dir": os.path.join(output_dir, f"gen_initial"),
            "prev_patch_files": [],
            "curr_patch_files": [],
            "run_eval": True,
        }

        eval_ensemble = (
            "ensemble" in optimize_option
            and all(can_domain_ensembled(domain) for domain in domains)
            and run_baseline != "no_archive"
        )
        if metadata["run_eval"] and eval_ensemble:
            for domain, eval_subset, eval_n in zip(domains, eval_subsets, eval_samples):
                _ = get_ensemble_scores_container(
                    docker_client,
                    domain,
                    (
                        output_dir
                        if optimize_option != "only_ensemble"
                        else agent_archive_path
                    ),
                    gen_output_dir=metadata["gen_output_dir"],
                    root_dir=root_dir,
                    root_commit=root_commit,
                    prev_patch_files=metadata["prev_patch_files"]
                    + metadata["curr_patch_files"],
                    num_samples=eval_n,
                    subsets=[
                        eval_subset,
                        eval_subset.replace("_train", "_val"),
                        *(
                            [eval_subset.replace("_train", "_test")]
                            if eval_test
                            else []
                        ),
                    ],
                )

    with open(os.path.join(output_dir, "generate_loop.log"), "a") as f:
        args_dict = {k: v for k, v in locals().items() if k not in ['docker_client', 'archive', 'metadata', 'metadata_file', 'resume_metadata']}
        args_str = ", ".join([f"{k}={v}" for k, v in args_dict.items()])
        f.write(f"Args: {args_str}\n")

    if not resume_from_run:
        start_genid = len(archive)
    
    if not resume_from_run:
        if not edit_select_parent or run_baseline == "no_archive":
            parent_genid = select_parent(archive, output_dir, domains, method=parent_selection)
        else:
            parent_genid = select_next_parent_container(
                docker_client,
                domains,
                output_dir,
                archive,
                root_dir, root_commit,
            )
    
    for current_genid in range(start_genid, max_generation + 1):
        if resume_from_run and current_genid == start_genid and start_genid > 0:
            pass
        elif current_genid > start_genid or (not resume_from_run):
            if not edit_select_parent or run_baseline == "no_archive":
                parent_genid = select_parent(archive, output_dir, domains, method=parent_selection)
            else:
                parent_genid = select_next_parent_container(
                    docker_client,
                    domains,
                    output_dir,
                    archive,
                    root_dir, root_commit,
                )
            safe_log(f"Gen {current_genid}: selected parent via select_parent = {parent_genid}")
        
        metadata = generate(
            docker_client,
            domains,
            output_dir,
            run_id,
            current_genid,
            parent_genid=parent_genid,
            root_dir=root_dir,
            root_commit=root_commit,
            eval_samples=eval_samples,
            eval_workers=eval_workers,
            eval_subsets=eval_subsets,
            run_hyper_agent=True,
            run_baseline=run_baseline,
            optimize_option=optimize_option,
            agent_archive_path=agent_archive_path,
            eval_test=eval_test,
            edit_select_parent=edit_select_parent,
            max_generation=max_generation,
        )

        archive = update_and_save_archive(output_dir, archive, new_node=current_genid)

        if not metadata["parent_agent_success"]:
            update_node_metadata(output_dir, parent_genid, {"valid_parent": False})

        eval_ensemble = (
            "ensemble" in optimize_option
            and all(can_domain_ensembled(domain) for domain in domains)
            and run_baseline != "no_archive"
        )
        if metadata["run_eval"] and eval_ensemble:
            for domain, eval_subset, eval_n in zip(domains, eval_subsets, eval_samples):
                _ = get_ensemble_scores_container(
                    docker_client,
                    domain,
                    (
                        output_dir
                        if optimize_option != "only_ensemble"
                        else agent_archive_path
                    ),
                    gen_output_dir=metadata["gen_output_dir"],
                    root_dir=root_dir,
                    root_commit=root_commit,
                    prev_patch_files=metadata["prev_patch_files"]
                    + metadata["curr_patch_files"],
                    num_samples=eval_n,
                    subsets=[
                        eval_subset,
                        eval_subset.replace("_train", "_val"),
                        *(
                            [eval_subset.replace("_train", "_test")]
                            if eval_test
                            else []
                        ),
                    ],
                )

        for domain in domains:
            splits = get_domain_splits(domain)
            if optimize_option == "only_ensemble":
                score_types = ["ensemble"]
            elif eval_ensemble:
                score_types = ["agent", "ensemble", "max"]
            else:
                score_types = ["agent"]
            for split in splits:
                for stype in score_types:
                    plot_progress_single(domain, output_dir, split=split, type=stype)
                    visualize_archive_single(
                        domain, output_dir, split=split, type=stype
                    )

        if len(domains) > 1:
            domain_splits_sets = [set(get_domain_splits(d)) for d in domains]
            common_splits = (
                sorted(list(set.intersection(*domain_splits_sets)))
                if domain_splits_sets
                else []
            )
            if optimize_option == "only_ensemble":
                together_score_types = ["ensemble"]
            elif eval_ensemble:
                together_score_types = ["agent", "ensemble", "max"]
            else:
                together_score_types = ["agent"]
            for split in common_splits:
                for stype in together_score_types:
                    plot_progress_together(domains, output_dir, split=split, type=stype)
                    visualize_archive_together(
                        domains, output_dir, split=split, type=stype
                    )

        parent_genid = None
        if not edit_select_parent or run_baseline == "no_archive":
            parent_genid = select_parent(archive, output_dir, domains, method=parent_selection)
        else:
            parent_genid = select_next_parent_container(
                docker_client,
                domains,
                output_dir,
                archive,
                root_dir, root_commit,
            )
        print(f"generate_loop: generation {current_genid} completed, parent {parent_genid}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default=None, help="Run ID")
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        choices=[
            "bbob_unconstrained",
            "bbob_constrained",
            "metabox_mo",
        ],
        required=True,
        help="One or more domains to evaluate",
    )
    parser.add_argument(
        "--max_generation",
        type=int,
        default=10,
        help="Maximum number of evolution generations",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        nargs="+",
        default=None,
        help="Evaluation samples per domain (-1 for all). Provide one value per domain.",
    )
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=10,
        help="Number of evaluation workers in parallel",
    )
    parser.add_argument(
        "--parent_selection",
        type=str,
        default="score_child_prop",
        choices=["random", "latest", "best", "score_prop", "score_child_prop"],
        help="Parent selection method",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to an existing output folder to resume from",
    )
    parser.add_argument(
        "--resume_from_run",
        type=str,
        default=None,
        help="Resume from a specific gen in a run. Format: outputs/generate_xxx/gen_N",
    )
    parser.add_argument(
        "--output_dir_parent",
        type=str,
        default=None,
        help="Path to the parent output folder",
    )
    parser.add_argument(
        "--reset_task_agent",
        default=False,
        action="store_true",
        help="Whether to reset the changes in the task agent",
    )
    parser.add_argument(
        "--reset_hyper_agent",
        default=False,
        action="store_true",
        help="Whether to reset the changes in the meta agent",
    )
    parser.add_argument(
        "--copy_root_dir",
        type=str,
        default=None,
        help="Copy root dir for setup_initial_gen",
    )
    parser.add_argument(
        "--run_baseline",
        type=str,
        choices=[
            "no_selfimprove", "no_archive",
            "dgm", "dgm_custom",
            "dgm+no_selfimprove", "dgm_custom+no_selfimprove",
        ],
        default=None,
        help="Run baseline",
    )
    parser.add_argument(
        "--optimize_option",
        type=str,
        default="only_agent",
        choices=["both_agent_ensemble", "only_agent", "only_ensemble"],
        help="Which part of the algorithm to optimize",
    )
    parser.add_argument(
        "--agent_archive_path",
        type=str,
        default=None,
        help="Path to agent archive (required if --optimize_option=only_ensemble)",
    )
    parser.add_argument(
        "--eval_test",
        default=False,
        action="store_true",
        help="Always run test set evaluation",
    )
    parser.add_argument(
        "--edit_select_parent",
        default=False,
        action="store_true",
        help="Whether to allow the agent to edit the selection mechanism",
    )
    args = parser.parse_args()

    if args.optimize_option == "only_ensemble" and args.agent_archive_path is None:
        parser.error(
            "--agent_archive_path is required when --optimize_option=only_ensemble"
        )
    if args.eval_samples is None:
        eval_samples = [-1] * len(args.domains)
    elif len(args.eval_samples) == len(args.domains):
        eval_samples = args.eval_samples
    else:
        parser.error("--eval_samples must be a one per domain if provided")

    eval_subsets = [get_domain_eval_subset(d) for d in args.domains]
    output_dir = generate_loop(
        domains=args.domains,
        run_id=args.run_id,
        max_generation=args.max_generation,
        eval_samples=eval_samples,
        eval_workers=args.eval_workers,
        eval_subsets=eval_subsets,
        parent_selection=args.parent_selection,
        resume_from=args.resume_from,
        resume_from_run=args.resume_from_run,
        output_dir_parent=args.output_dir_parent,
        reset_task_agent=args.reset_task_agent,
        reset_hyper_agent=args.reset_hyper_agent,
        copy_root_dir=args.copy_root_dir,
        run_baseline=args.run_baseline,
        optimize_option=args.optimize_option,
        agent_archive_path=args.agent_archive_path,
        eval_test=args.eval_test,
        edit_select_parent=args.edit_select_parent,
    )