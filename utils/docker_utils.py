import io
import logging
import os
import tarfile
import threading
import warnings
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

import docker
from docker.models.containers import Container
from docker.types import Mount

from utils.constants import REPO_NAME

load_dotenv()

warnings.filterwarnings(
    "ignore",
    message=r"The default behavior of tarfile extraction has been changed.*",
    category=RuntimeWarning,
)

# Thread-local storage for loggers
_thread_local = threading.local()


def get_thread_logger():
    """Get the logger instance specific to the current thread."""
    return getattr(_thread_local, "logger", None)


def setup_logger(log_file):
    """
    Set up a thread-safe logger with file handler.

    Args:
        log_file (str): Path to the log file

    Returns:
        logging.Logger: Thread-specific logger instance
    """
    # Create logger with thread-specific name
    thread_id = threading.get_ident()
    logger_name = f"docker_logger_{thread_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # Create file handler with lock
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.stream.lock = threading.Lock()
    handler.flush = lambda: handler.stream.flush()  # Ensure flush works

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(threadName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    # Store logger in thread local storage
    _thread_local.logger = logger

    return logger


def safe_log(message: str, level: int = logging.INFO, verbose=True):
    """Thread-safe logging function."""
    if verbose:
        logger = get_thread_logger()
        if logger:
            logger.log(level, message)
        else:
            print(f"Warning: No logger found for thread {threading.get_ident()}")


def log_container_output(exec_result, verbose=True):
    """
    Log output from a Docker container execution, handling both streaming and non-streaming cases.
    """
    # Handle output logging
    if isinstance(exec_result.output, bytes):
        # Handle non-streaming output
        safe_log(f"Container output: {exec_result.output.decode()}", verbose=verbose)
    else:
        # Handle streaming output
        for chunk in exec_result.output:
            if chunk:
                safe_log(f"Container output: {chunk.decode().strip()}", verbose=verbose)

    # Check exit code
    if exec_result.exit_code and exec_result.exit_code != 0:
        error_msg = f"Script failed with exit code {exec_result.exit_code}"
        safe_log(error_msg, logging.ERROR, verbose=verbose)
        raise Exception(error_msg)


def build_container(
    client,
    repo_path="./",
    image_name="app",
    container_name="app-container",
    force_rebuild=False,
    domains=None,
    verbose=True,
):
    """
    Build the Docker image with proxy and host networking, then run it interactively.
    
    根据 domains 参数自动选择镜像名称和 Dockerfile：
    - 如果 domains 包含 'bbob_unconstrained' 或 'bbob_constrained'，使用 'hada' 镜像和 'Dockerfile'
    - 否则使用默认的 image_name 和 'Dockerfile'
    """
    try:
        # Set up proxy environment
        proxy_env = {
            "https_proxy": "http://fwdproxy:8080",
            "http_proxy": "http://fwdproxy:8080",
            "ftp_proxy": "http://fwdproxy:8080",
            "http_no_proxy": ".facebook.com|.tfbnw.net|*.fb.com",
        }

        # 根据 domains 参数选择镜像名称和 Dockerfile
        use_bbob_dockerfile = domains is not None and ("bbob_unconstrained" in domains or "bbob_constrained" in domains)
        dockerfile_name = "Dockerfile" if use_bbob_dockerfile else "Dockerfile"
        actual_image_name = "hada" if use_bbob_dockerfile else image_name
        
        # Check if we need to rebuild
        image_exists = any(
            actual_image_name in tag for img in client.images.list() for tag in img.tags
        )
        if force_rebuild or not image_exists:
            safe_log(
                f"Building the Docker image '{actual_image_name}' with {dockerfile_name}...",
                verbose=verbose,
            )
            image, logs = client.images.build(
                path=repo_path,  # <-- This points to the directory containing Dockerfile
                tag=actual_image_name,
                rm=True,
                network_mode="host",
                nocache=force_rebuild,
                buildargs=proxy_env,
                dockerfile=dockerfile_name,  # 指定使用的 Dockerfile
            )
            for log_entry in logs:
                if "stream" in log_entry:
                    safe_log(log_entry["stream"].strip())
            safe_log(f"Image '{actual_image_name}' built successfully.", verbose=verbose)
        else:
            safe_log(
                f"Docker image '{actual_image_name}' already exists. Skipping build.",
                verbose=verbose,
            )
            image = next(
                (img for img in client.images.list() if actual_image_name in img.tags), None
            )

    except Exception as e:
        safe_log(f"Error while building the Docker image: {e}")
        return None

    try:
        # Remove existing container if it exists
        try:
            existing = client.containers.get(container_name)
            existing.remove(force=True)
            safe_log(f"Removed existing container '{container_name}'.", verbose=verbose)
        except docker.errors.NotFound:
            pass

        # Build device_requests for GPU passthrough (conditional)
        device_requests = []
        is_podman = False  # Track if we're using Podman
        # Enable GPU only if domain contains "genesis"
        needs_gpu = domains is not None and any(
            "genesis" in domain.lower() for domain in domains
        )

        if needs_gpu:
            # Check if GPU is available on the host
            safe_log(
                "GPU requested. Checking if GPU is available on the host...",
                verbose=verbose,
            )
            try:
                info = client.info()
                gpu_info = info.get("Runtimes", {})
                has_nvidia_runtime = "nvidia" in gpu_info

                # Podman detection: Check ServerVersion, base_url, or presence of podman-specific runtimes
                # Note: Due to docker alias (alias docker=podman), the Python client may connect via
                # different interfaces that don't expose the podman socket path directly
                server_version = info.get("ServerVersion", "").lower()
                base_url = str(client.api.base_url).lower()
                runtimes = info.get("Runtimes", {})

                # Podman typically uses crun/runc instead of nvidia runtime
                has_podman_runtimes = any(
                    rt in runtimes for rt in ["crun", "crun-vm", "crun-wasm"]
                )
                is_podman = (
                    "podman" in server_version
                    or "podman" in base_url
                    or has_podman_runtimes
                )

                safe_log(
                    f"Debug - ServerVersion: {info.get('ServerVersion', 'N/A')}, has_nvidia_runtime: {has_nvidia_runtime}, is_podman: {is_podman}",
                    verbose=verbose,
                )

                # Enable GPU via device_requests for Docker or devices for Podman
                if has_nvidia_runtime:
                    # Docker with nvidia runtime uses device_requests
                    device_requests = [
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ]
                    safe_log(
                        "GPU access enabled for container (Docker nvidia runtime).",
                        verbose=verbose,
                    )
                elif is_podman:
                    # Podman needs explicit device mounts (device_requests doesn't work with Podman)
                    # This is a workaround - we'll use device_requests but log a warning
                    device_requests = [
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ]
                    safe_log(
                        "Detected Podman (via crun/runc runtimes) - attempting GPU via device_requests.",
                        verbose=verbose,
                    )
                    safe_log(
                        "Note: If GPU is not accessible, Podman may need explicit --device flags.",
                        verbose=verbose,
                    )
                else:
                    safe_log(
                        "Warning: GPU requested but neither nvidia-docker runtime nor Podman detected. Running without GPU.",
                        verbose=verbose,
                    )
            except Exception as e:
                safe_log(
                    f"Warning: Could not verify GPU availability: {e}. Running without GPU.",
                    verbose=verbose,
                )
        else:
            safe_log("GPU not requested. Running without GPU.", verbose=verbose)

        # Run the container with host networking and volume mount
        # For Podman, we need to pass GPU devices explicitly via security_opt or devices
        run_kwargs = {
            "image": image_name,
            "name": container_name,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "network_mode": "host",
            "volumes": {
                os.path.abspath(repo_path): {"bind": f"/{REPO_NAME}", "mode": "rw"}
            },
            "environment": {
                "AIHUBMIX_API_KEY": os.getenv("AIHUBMIX_API_KEY"),
                "ALIYUN_API_KEY": os.getenv("ALIYUN_API_KEY"),
                "KIMI_API_KEY": os.getenv("KIMI_API_KEY"),
                "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
                "USE_LITELLM_PROXY": "False",  # Disable LiteLLM proxy in container
                "LITELLM_PROXY_API_BASE": "",
                "LITELLM_PROXY_API_KEY": "",
            },
            "command": "tail -f /dev/null",
        }

        # Add GPU support
        if device_requests:
            if is_podman:
                # Podman's Python SDK doesn't support device_requests properly
                # We need to use Podman CLI directly to get --gpus all working
                safe_log(
                    "Using Podman CLI directly for GPU support (Python SDK doesn't support --gpus properly)",
                    verbose=verbose,
                )

                # Build the podman run command with CDI GPU support
                # Podman 5.x uses CDI (Container Device Interface) instead of --gpus
                import subprocess

                volume_mount = f"{os.path.abspath(repo_path)}:/{REPO_NAME}:rw"
                cmd = [
                    "podman",
                    "run",
                    "-d",  # detach
                    "-it",  # interactive + tty
                    "--network=host",
                    "-v",
                    volume_mount,
                    "--device",
                    "nvidia.com/gpu=all",  # CDI format for Podman 5.x
                    # Add API keys and LiteLLM proxy configuration
                    "-e",
                    f"AIHUBMIX_API_KEY={os.getenv('AIHUBMIX_API_KEY')}",
                    "-e",
                    f"ALIYUN_API_KEY={os.getenv('ALIYUN_API_KEY')}",
                    "-e",
                    f"KIMI_API_KEY={os.getenv('KIMI_API_KEY')}",
                    "-e",
                    f"DEEPSEEK_API_KEY={os.getenv('DEEPSEEK_API_KEY')}",
                    "-e",
                    "USE_LITELLM_PROXY=False",  # Disable LiteLLM proxy in container
                    "-e",
                    "LITELLM_PROXY_API_BASE=",
                    "-e",
                    "LITELLM_PROXY_API_KEY=",
                    # Add environment variables for NVIDIA libraries
                    "-e",
                    "NVIDIA_VISIBLE_DEVICES=all",
                    "-e",
                    "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                    "-e",
                    "LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib",
                    "--name",
                    container_name,
                    image_name,
                    "tail",
                    "-f",
                    "/dev/null",
                ]

                safe_log(f"Running Podman command: {' '.join(cmd)}", verbose=verbose)
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    raise Exception(f"Podman run failed: {result.stderr}")

                # Log both stdout and stderr for debugging
                if result.stdout:
                    safe_log(f"Podman stdout: {result.stdout.strip()}", verbose=verbose)
                if result.stderr:
                    safe_log(f"Podman stderr: {result.stderr.strip()}", verbose=verbose)

                # Get the container object from the client
                container = client.containers.get(container_name)
            else:
                # Docker with nvidia runtime
                run_kwargs["device_requests"] = device_requests
                container = client.containers.run(**run_kwargs)
        else:
            container = client.containers.run(**run_kwargs)
        safe_log(f"Container '{container_name}' started successfully.", verbose=verbose)

        # Verify GPU access if GPU was requested
        if needs_gpu and device_requests:
            safe_log("Verifying GPU access inside container...", verbose=verbose)
            gpu_accessible = verify_gpu_in_container(container, verbose=verbose)
            if gpu_accessible:
                safe_log(
                    "GPU verification PASSED - GPU is accessible in container!",
                    verbose=verbose,
                )
            else:
                safe_log(
                    "GPU verification FAILED - GPU is NOT accessible in container!",
                    verbose=verbose,
                )

        return container
    except Exception as e:
        safe_log(f"Error while starting the container: {e}", verbose=verbose)
        return None


def create_archive(path: Union[str, Path], data: Optional[bytes] = None) -> bytes:
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        if data is not None:
            tarinfo = tarfile.TarInfo(name=str(path))
            tarinfo.size = len(data)
            tarinfo.uid = 0
            tarinfo.gid = 0
            tar.addfile(tarinfo, io.BytesIO(data))
        else:
            path = Path(path)
            arcname = path.name
            for item in path.rglob("*"):
                tarinfo = tar.gettarinfo(
                    str(item), arcname=str(item.relative_to(path.parent))
                )
                tarinfo.uid = 0
                tarinfo.gid = 0
                if item.is_file():
                    with open(item, "rb") as f:
                        tar.addfile(tarinfo, f)
                else:
                    tar.addfile(tarinfo)
    tar_stream.seek(0)
    return tar_stream.read()


def copy_to_container(
    container, source_path: Union[str, Path], dest_path: Union[str, Path], verbose=True
) -> None:
    """
    Copy a file or directory from the local system to a Docker container.

    Args:
        container: Docker container object
        source_path (Union[str, Path]): Path to the source file/directory on local system
        dest_path (Union[str, Path]): Destination path in the container

    Raises:
        FileNotFoundError: If source path doesn't exist
        Exception: For other errors during copy operation
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        # Destination directory inside the container
        container_dest_dir = str(dest_path.parent)

        # Build tar stream in memory
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            if source_path.is_file():
                # Single file -> archive entry name is dest_path.name
                arcname = dest_path.name
                tarinfo = tar.gettarinfo(str(source_path), arcname=arcname)
                tarinfo.uid = 0
                tarinfo.gid = 0
                with open(source_path, "rb") as f:
                    tar.addfile(tarinfo, f)
            else:
                # Directory -> we want the top-level directory in the archive
                # to be dest_path.name, with its contents under it.
                def _reset_uid_gid(ti: tarfile.TarInfo) -> tarfile.TarInfo:
                    ti.uid = 0
                    ti.gid = 0
                    return ti

                tar.add(
                    str(source_path),
                    arcname=dest_path.name,
                    filter=_reset_uid_gid,
                )

        # Rewind the stream before sending
        tar_stream.seek(0)

        # Ensure destination directory exists in container
        container.exec_run(f"mkdir -p {container_dest_dir}")

        # Pass the file-like object (stream) to put_archive
        success = container.put_archive(container_dest_dir, tar_stream)

        if not success:
            raise Exception(f"Failed to copy {source_path} to container")

        safe_log(
            f"Successfully copied {source_path} to container at {dest_path}",
            verbose=verbose,
        )

    except Exception as e:
        safe_log(f"Error copying to container: {e}", logging.ERROR, verbose=verbose)
        raise


def copy_from_container(
    container, source_path: Union[str, Path], dest_path: Union[str, Path], verbose=True
) -> None:
    """
    Copy a file or directory from a Docker container to the local system.

    Args:
        container: Docker container object
        source_path (Union[str, Path]): Path to the source file/directory in container
        dest_path (Union[str, Path]): Destination path on local system

    Raises:
        FileNotFoundError: If source path doesn't exist in container
        Exception: For other errors during copy operation
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    try:
        # Check if source exists in container
        result = container.exec_run(f"test -e {source_path}")
        if result.exit_code and result.exit_code != 0:
            raise FileNotFoundError(
                f"Source path not found in container: {source_path}"
            )

        # Get file type from container
        result = container.exec_run(f"stat -f '%HT' {source_path}")
        is_file = result.output.decode().strip() == "Regular File"

        # Create destination directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Get archive from container
        bits, stat = container.get_archive(str(source_path))

        # Concatenate all chunks into a single bytes object
        archive_data = b"".join(bits)

        # Extract to temporary stream
        stream = io.BytesIO(archive_data)

        with tarfile.open(fileobj=stream, mode="r") as tar:
            # If extracting a single file
            if is_file:
                member = tar.getmembers()[0]
                source_file = tar.extractfile(member)
                if source_file is not None:
                    with source_file:
                        data = source_file.read()
                        # Write directly to destination file
                        with open(dest_path, "wb") as dest_file:
                            dest_file.write(data)
            else:
                # For directories, extract to parent directory
                tar.extractall(path=str(dest_path.parent))
                # Rename if necessary
                extracted_path = dest_path.parent / Path(stat["name"]).name
                if extracted_path != dest_path and extracted_path.exists():
                    extracted_path.rename(dest_path)

        safe_log(
            f"Successfully copied from container {source_path} to local path {dest_path}",
            verbose=verbose,
        )

    except Exception as e:
        if verbose:
            safe_log(
                f"Error copying from container: {e}", logging.ERROR, verbose=verbose
            )
        raise


def verify_gpu_in_container(container, verbose=True) -> bool:
    """
    Verify GPU is accessible inside a running container.

    Similar to the interactive test:
    - Check if nvidia-smi is available
    - Check if /dev/nvidia* devices are mounted
    - Check if PyTorch can see the GPU

    Args:
        container: Docker container object
        verbose: Whether to log output

    Returns:
        bool: True if GPU is accessible, False otherwise
    """
    try:
        # Test 1: Check if GPU devices are mounted (most important check)
        # Use sh -c to enable shell features like redirection
        result = container.exec_run(
            ["sh", "-c", "ls /dev/nvidia* 2>/dev/null || echo 'not found'"]
        )
        output = result.output.decode().strip()
        if "not found" in output:
            safe_log("GPU check: /dev/nvidia* devices not found", verbose=verbose)
            # Check what's actually in /dev to debug
            dev_result = container.exec_run(["sh", "-c", "ls /dev | head -20"])
            safe_log(
                f"GPU check: /dev contents (first 20): {dev_result.output.decode().strip()}",
                verbose=verbose,
            )
            return False
        safe_log(f"GPU check: GPU devices found - {output}", verbose=verbose)

        # Test 2: Check if nvidia-smi is available (optional - might not be in PATH)
        result = container.exec_run("which nvidia-smi")
        if result.exit_code != 0:
            safe_log(
                "GPU check: nvidia-smi not in PATH, checking /usr/bin/nvidia-smi",
                verbose=verbose,
            )
            result = container.exec_run(
                "test -f /usr/bin/nvidia-smi && echo 'found' || echo 'not found'"
            )
            if "not found" in result.output.decode():
                safe_log(
                    "GPU check: nvidia-smi not found, but GPU devices are present",
                    verbose=verbose,
                )
        else:
            safe_log("GPU check: nvidia-smi found", verbose=verbose)

        # Test 3: Try running nvidia-smi
        result = container.exec_run("nvidia-smi")
        if result.exit_code != 0:
            safe_log(
                f"GPU check: nvidia-smi failed - {result.output.decode()}",
                verbose=verbose,
            )
            return False
        safe_log("GPU check: nvidia-smi executed successfully", verbose=verbose)

        # Test 4: Check LD_LIBRARY_PATH and NVIDIA libraries
        result = container.exec_run(["sh", "-c", "echo $LD_LIBRARY_PATH"])
        ld_library_path = result.output.decode().strip()
        safe_log(f"GPU check: LD_LIBRARY_PATH = {ld_library_path}", verbose=verbose)

        # Check if NVIDIA libraries are accessible
        result = container.exec_run(
            [
                "sh",
                "-c",
                "ls /usr/local/nvidia/lib* 2>/dev/null | head -5 || echo 'NVIDIA lib dirs not found'",
            ]
        )
        safe_log(
            f"GPU check: NVIDIA lib check: {result.output.decode().strip()}",
            verbose=verbose,
        )

        # Test 5: Check PyTorch GPU availability (if PyTorch is installed)
        result = container.exec_run(
            'python -c "import torch; print(torch.cuda.is_available())"'
        )
        if result.exit_code == 0:
            cuda_available = result.output.decode().strip()
            safe_log(
                f"GPU check: PyTorch CUDA available = {cuda_available}", verbose=verbose
            )
            if cuda_available.lower() != "true":
                return False
        else:
            # PyTorch might not be installed, but GPU devices are present
            safe_log(
                "GPU check: PyTorch not available, but GPU devices are present",
                verbose=verbose,
            )

        # Test 6: Check Genesis GPU initialization (if Genesis is installed)
        result = container.exec_run(
            "python -c \"import genesis as gs; gs.init(backend=gs.cuda, logging_level='warning'); print(gs.device)\""
        )
        if result.exit_code == 0:
            genesis_device = result.output.decode().strip()
            safe_log(
                f"GPU check: Genesis initialized with device = {genesis_device}",
                verbose=verbose,
            )
            return "cuda" in genesis_device.lower()
        else:
            # Genesis might not be installed, but GPU devices are present
            safe_log(
                "GPU check: Genesis not available or failed to initialize, but GPU devices are present",
                verbose=verbose,
            )
            return True

    except Exception as e:
        safe_log(f"GPU check failed with error: {e}", verbose=verbose)
        return False


def cleanup_container(container, verbose=True):
    """
    Stop and remove a container cleanly.

    Args:
        container (docker.models.containers.Container): The container to stop and remove.
    """
    try:
        safe_log(f"Stopping container {container.name}...", verbose=verbose)
        container.stop(timeout=10)
    except Exception as e:
        safe_log(
            f"Error while stopping container {container.name}: {e}",
            level=logging.WARNING,
            verbose=verbose,
        )

    try:
        safe_log(f"Removing container {container.name}...", verbose=verbose)
        container.remove(force=True)
    except Exception as e:
        safe_log(
            f"Error while removing container {container.name}: {e}",
            level=logging.ERROR,
            verbose=verbose,
        )