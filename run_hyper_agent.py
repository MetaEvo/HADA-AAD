import argparse
import os
import asyncio
import sys

# Custom exception handler to suppress asyncio subprocess transport cleanup warnings
def _ignore_exception_handler(loop, context):
    msg = context.get("message", "")
    # Filter out subprocess transport cleanup warnings
    if "Event loop is closed" in msg or "BaseSubprocessTransport" in msg:
        return  # Ignore these warnings
    # Let the default handler deal with other exceptions
    loop.default_exception_handler(context)

from agent.llm import DEFAULT_MODEL
from hyper_agent import HyperAgent
from utils.git_utils import diff_versus_commit, reset_paths_to_commit


async def run_hyper_agent_async(args):
    """Run meta agent with proper asyncio exception handling."""
    # Set exception handler on the running loop
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_ignore_exception_handler)
    
    # Run meta agent
    hyper_agent = HyperAgent(
        model=args.model,
        chat_history_file=args.chat_history_file,
    )
    hyper_agent.forward(
        repo_path=args.repo_path,
        eval_path=args.evals_folder,
        iterations_left=args.iterations_left,
        domains=args.domains,
    )


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model to use for the agent",
    )
    parser.add_argument(
        "--chat_history_file",
        type=str,
        default="./outputs/chat_history.md",
        help="Path to chat history file",
    )
    parser.add_argument(
        "--repo_path", type=str, default="./", help="Path to the agent file"
    )
    parser.add_argument(
        "--evals_folder",
        type=str,
        default="./outputs/",
        help="Path to the folder containing the evaluation files",
    )
    parser.add_argument(
        "--iterations_left",
        type=int,
        default=None,
        help="The number of remaining iterations in which the meta agent will be invoked in future.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated list of domains being optimized (e.g., 'metabox_mo')",
    )
    parser.add_argument(
        "--git_dir", required=True, help="Path to git repository directory"
    )
    parser.add_argument(
        "--base_commit", required=True, help="Base commit hash to compare against"
    )
    parser.add_argument(
        "--outdir", required=False, default="./outputs/", help="Output directory"
    )
    args = parser.parse_args()

    # Run meta agent with proper asyncio handling
    asyncio.run(run_hyper_agent_async(args))

    # Reset unwanted diffs
    reset_paths_to_commit(
        git_dname=args.git_dir, commit=args.base_commit, paths=["domains/"]
    )

    # Save git diff
    hyper_agent_patch = diff_versus_commit(args.git_dir, args.base_commit)
    hyper_agent_patch_outfile = (
        os.path.join(args.outdir, "hyper_agent_patch.diff")
        if args.outdir
        else "hyper_agent_patch.diff"
    )
    with open(hyper_agent_patch_outfile, "w") as f:
        f.write(hyper_agent_patch)

    # Clean up asyncio resources to avoid "Event loop is closed" warnings
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(loop.shutdown_asyncgens())
    except:
        pass


if __name__ == "__main__":
    main()