import os
import git
import subprocess


def get_git_commit_hash(repo_path='.'):
    try:
        # Load the repository
        repo = git.Repo(repo_path)
        # Get the current commit hash
        commit_hash = repo.head.commit.hexsha
        return commit_hash
    except Exception as e:
        print("Error while getting git commit hash:", e)
        return None

def apply_patch(git_dname, patch_str):
    """
    Apply a patch to the repository at `git_dname`.
    """
    cmd = ["git", "-C", git_dname, "apply", "--reject", "-"]
    result = subprocess.run(
        cmd,
        input=patch_str,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False
    )
    # Check if the patch was applied successfully
    if result.returncode != 0:
        print(f"apply_patch error: Patch did not fully apply. Return code: {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}")
    else:
        print("apply_patch successful")

def diff_versus_commit(git_dname, commit):
    """
    Take a diff of `git_dname` current contents versus the `commit`, including untracked files,
    without modifying the repository state.
    """
    # Get diff of tracked files
    diff_cmd = ["git", "-C", git_dname, "diff", commit]
    result = subprocess.run(diff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    diff_output = result.stdout.decode()

    # Get list of untracked files
    untracked_files_cmd = ["git", "-C", git_dname, "ls-files", "--others", "--exclude-standard"]
    result = subprocess.run(untracked_files_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    untracked_files = result.stdout.decode().splitlines()

    # Generate diffs for untracked files
    for file in untracked_files:
        # Diff untracked file against /dev/null (empty file)
        file_path = os.path.join(git_dname, file)
        devnull = '/dev/null'
        if os.name == 'nt':  # Handle Windows
            devnull = 'NUL'
        diff_file_cmd = ["git", "-C", git_dname, "diff", "--no-index", devnull, file]
        result = subprocess.run(
            diff_file_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=git_dname,
            check=False
        )
        diff_file_output = result.stdout.decode('utf-8', errors='replace')
        diff_output += diff_file_output

    return diff_output

def reset_paths_to_commit(git_dname, commit, paths):
    """
    Reset only the given `paths` in the repository at `git_dname` to the given `commit`.
    Also cleans untracked files in those paths.

    Args:
        git_dname (str): Path to the git repository root.
        commit (str): Commit hash to reset to.
        paths (list[str]): List of paths (relative to repo root) to reset.
    """
    if not paths:
        print("reset_paths_to_commit: No paths provided, skipping.")
        return

    # Step 0: Verify the commit exists
    try:
        subprocess.run(
            ["git", "-C", git_dname, "cat-file", "-e", commit],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError:
        print(f"reset_paths_to_commit: Commit {commit} does not exist, using HEAD instead")
        commit = "HEAD"

    # Step 1: Checkout tracked files from the given commit
    try:
        subprocess.run(
            ["git", "-C", git_dname, "checkout", commit, "--"] + paths,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"reset_paths_to_commit: Checked out {paths} from {commit}")
    except subprocess.CalledProcessError as e:
        print(f"reset_paths_to_commit error (checkout): {e.stderr}")

    # Step 2: Clean untracked files in those paths
    try:
        subprocess.run(
            ["git", "-C", git_dname, "clean", "-fd", "--"] + paths,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"reset_paths_to_commit: Cleaned untracked files in {paths}")
    except subprocess.CalledProcessError as e:
        print(f"reset_paths_to_commit error (clean): {e.stderr}")

def reset_to_commit(git_dname, commit):
    """
    Reset the repository at `git_dname` to the given `commit`.
    """
    # Step 1: Hard-reset tracked files
    reset_cmd = ["git", "-C", git_dname, "reset", "--hard", commit]
    result_reset = subprocess.run(
        reset_cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if result_reset.returncode != 0:
        print(f"reset_to_commit error: Failed to reset {git_dname} to commit '{commit}'. STDOUT: {result_reset.stdout} STDERR: {result_reset.stderr}")
    else:
        print(f"reset_to_commit successful: {commit}")

    # Step 2: Clean untracked files (the "new files") and directories
    clean_cmd = ["git", "-C", git_dname, "clean", "-fd"]
    result_clean = subprocess.run(
        clean_cmd,
        capture_output=True,
        text=True,
        check=False
    )
    if result_clean.returncode != 0:
        print(f"reset_to_commit clean error: Failed to clean {git_dname}. STDOUT: {result_clean.stdout} STDERR: {result_clean.stderr}")
    else:
        print(f"reset_to_commit clean successful: {commit}")

def commit_repo(git_dname, commit_message="a nonsense commit message", user_name="user", user_email="you@example.com"):
    """
    Stages all changes and commits them with a given message.
    If there's nothing to commit, returns the current commit hash.
    """
    # Stage all changes
    add_cmd = ["git", "-C", git_dname, "add", "--all"]
    result_add = subprocess.run(add_cmd, capture_output=True, text=True)
    if result_add.returncode != 0:
        print(f"commit_repo error (add): {result_add.stderr}")
        return None

    # Commit with dummy user details
    commit_cmd = [
        "git", "-C", git_dname, "-c", f"user.name={user_name}",
        "-c", f"user.email={user_email}", "commit", "-m", commit_message
    ]
    result_commit = subprocess.run(commit_cmd, capture_output=True, text=True)
    output = result_commit.stdout.strip() + "\n" + result_commit.stderr.strip()

    if result_commit.returncode != 0:
        if "nothing to commit" in output.lower():
            # No changes were committed; return current commit
            return get_git_commit_hash(git_dname)
        else:
            print(f"commit_repo error (commit): {result_commit.stderr}")
            return None

    # Extract the commit hash from stdout
    parts = result_commit.stdout.strip().split()
    if len(parts) >= 2:
        raw_hash = parts[1].strip("[]")
        return raw_hash
    else:
        return get_git_commit_hash(git_dname)

if __name__ == "__main__":
    # Get the current commit hash
    current_commit_hash = get_git_commit_hash()
    print(f"Current commit hash: {current_commit_hash}")

    # Get the diff of the current commit versus the previous commit
    previous_commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD^"]).decode().strip()
    print(f"Previous commit hash: {previous_commit_hash}")
    diff = diff_versus_commit(".", previous_commit_hash)
    print(diff)