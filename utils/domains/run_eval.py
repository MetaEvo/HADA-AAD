import argparse
import os
from datetime import datetime
import shutil
import docker

from utils.constants import REPO_NAME
from utils.docker_utils import (
    build_container,
    cleanup_container,
    copy_from_container,
    log_container_output,
    safe_log,
    setup_logger,
)
from utils.domain_utils import get_domain_test_subset
from utils.gl_utils import apply_diffs_container, setup_initial_gen


def run_eval(
    output_dir,
    domain,
    run_id=None,
    num_samples=-1,
    num_workers=5,
    subset="",
    patch_files=None,
    copy_root_dir=None,
):
    # Setup
    parent_output_dir = (
        os.path.join(os.getcwd(), "outputs/") if output_dir is None else output_dir
    )
    output_dir = os.path.join(parent_output_dir, f"{run_id}/")
    if os.path.exists(output_dir):
        # If output_dir already exists, it means that the evaluation was already done, so skip
        print(f"Output directory {output_dir} already exists, skipping.")
        return
    root_dir, root_commit = setup_initial_gen(
        output_dir, [domain], copy_root_dir=copy_root_dir,
        subsets=[subset], resume=False, copy_eval=False,
    )
    logger = setup_logger(os.path.join(output_dir, "run_eval.log"))

    # Save args
    args_dict = locals()
    args_str = ", ".join([f"{k}={v}" for k, v in args_dict.items()])
    safe_log(f"Args: {args_str}\n")

    # Create and start the Docker container
    image_name = f"{REPO_NAME}"
    container_name = f"{REPO_NAME}-{domain}-eval-container-{run_id}"
    client = docker.DockerClient()
    container = build_container(
        client, root_dir, image_name, container_name, domains=[domain],
    )

    container.start()
    container_output_folder = "/tmp/"

    try:
        # Apply diffs if specified
        patch_files = patch_files if patch_files is not None else []
        commit_hash = apply_diffs_container(container, patch_files)

        # Run harness and report
        safe_log("Evaluating the agent...")
        eval_run_id = f"{domain}_eval"
        container_evaloutput_folder = os.path.join(container_output_folder, eval_run_id)
        command = [
            "timeout",
            "21600",  # 6h timeout
            "python",
            "-m",
            f"domains.harness",
            "--output_dir",
            container_output_folder,
            "--run_id",
            eval_run_id,
            "--domain",
            domain,
            "--num_samples",
            str(num_samples),
            "--num_workers",
            str(num_workers),
            "--subset",
            subset,
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        command = [
            "timeout",
            "3600",  # 1h timeout
            "python",
            "-m",
            f"domains.report",
            "--domain",
            domain,
            "--dname",
            os.path.join(container_output_folder, eval_run_id),
        ]
        exec_result = container.exec_run(cmd=command, workdir=f"/{REPO_NAME}")
        log_container_output(exec_result)
        # Copy container outputs to local, evaluation results
        evaloutput_folder = os.path.join(output_dir, f"{domain}/")
        copy_from_container(
            container,
            source_path=container_evaloutput_folder,
            dest_path=evaloutput_folder,
        )

    except Exception as e:
        print(f"Error in run_eval: {e}")
        pass

    # Even on errors or KeyboardInterrupt
    finally:
        # Reset to the root commit (only if container is still running)
        try:
            container.reload()
            exec_result = container.exec_run(
                cmd=["git", "reset", "--hard", root_commit], workdir=f"/{REPO_NAME}"
            )
            log_container_output(exec_result)
            exec_result = container.exec_run(
                cmd=["git", "clean", "-fd"], workdir=f"/{REPO_NAME}"
            )
            log_container_output(exec_result)
        except Exception as cleanup_error:
            print(f"Warning: Could not reset git in container: {cleanup_error}")

        # Cleanup container
        cleanup_container(container)

        # Remove root dir from local
        shutil.rmtree(os.path.dirname(root_dir))


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a system on the TEST subsets."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument("--run_id", type=str, default=None, help="Run ID")
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",  # one or more domains
        required=True,
        help="One or more domains to evaluate (must be from the allowed list)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate, -1 for all",
    )
    parser.add_argument(
        "--num_workers", type=int, default=10, help="Number of parallel workers"
    )
    parser.add_argument(
        "--patch_files", type=str, nargs="+", default=None, help="Patch files to apply"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Number of times to repeat this eval"
    )
    parser.add_argument(
        "--copy_root_dir", type=str, default=None, help="Path to a root dir to copy"
    )
    args = parser.parse_args()

    base_run_id = args.run_id
    base_run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if base_run_id is None
        else base_run_id
    )
    test_subsets = [get_domain_test_subset(d) for d in args.domains]

    for domain, subset in zip(args.domains, test_subsets):
        for i in range(args.repeat):
            run_id = f"{base_run_id}_{domain}_{i}"
            run_eval(
                output_dir=args.output_dir,
                domain=domain,
                run_id=run_id,
                num_samples=args.num_samples,
                num_workers=args.num_workers,
                subset=subset,
                patch_files=args.patch_files,
                copy_root_dir=args.copy_root_dir,
            )
