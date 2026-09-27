import argparse
import os

from task_agent import TaskAgent
from utils.git_utils import diff_versus_commit
from agent.llm import QWEN_MODEL


def main():
    parser = argparse.ArgumentParser(description='Run task agent on POLYGLOT benchmark.')
    parser.add_argument('--problem_statement', required=True, help='The problem statement to process')
    parser.add_argument('--git_dir', required=True, help='Path to git repository directory')
    parser.add_argument('--base_commit', required=True, help='Base commit hash to compare against')
    parser.add_argument('--chat_history_file', required=True, help='Path to chat history file')
    parser.add_argument('--outdir', required=False, default="/dgm/", help='Output directory')
    parser.add_argument('--test_description', default=None, required=False, help='Description of how to test the repository')
    parser.add_argument('--language', default=None, required=False, help='Coding language of the repository')
    parser.add_argument('--model', required=False, default=QWEN_MODEL, help='LLM model to use')
    args = parser.parse_args()

    # Process the repository
    agentic_system = TaskAgent(
        model=args.model,
        chat_history_file=args.chat_history_file,
    )
    inputs = {
        "domain": "polyglot",
        "problem_statement": args.problem_statement,
        "git_tempdir": args.git_dir,
        "base_commit": args.base_commit,
        "test_description": args.test_description,
        "language": args.language,
    }

    # Run the agentic system to try to solve the problem
    agentic_system.forward(inputs)

    # Get code diff and save to hyper_agent_patch.diff
    hyper_agent_patch = diff_versus_commit(args.git_dir, args.base_commit)
    hyper_agent_patch_outfile = os.path.join(args.outdir, 'hyper_agent_patch.diff') if args.outdir else 'hyper_agent_patch.diff'
    with open(hyper_agent_patch_outfile, 'w') as f:
        f.write(hyper_agent_patch)

if __name__ == "__main__":
    main()