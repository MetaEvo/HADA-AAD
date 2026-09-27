import argparse
import os
import json
import sys

from utils.gl_utils import load_archive_data
from select_next_parent import select_next_parent


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_output_dir", type=str, required=True)
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",  # one or more domains
        choices=[
            "search_arena",
            "paper_review",
            "balrog_babyai",
            "balrog_babaisai",
            "balrog_minihack",
            "balrog_nle",
            "genesis_go2walking",
        ],
        required=True,
        help="One or more domains to evaluate (must be from the allowed list)",
    )
    args = parser.parse_args()

    # Set up the archive and output directories
    output_dir = args.generate_output_dir
    domains = args.domains
    archive = load_archive_data(os.path.join(output_dir, "archive.jsonl"), last_only=True)['archive']

    # Get next parent generation id
    next_parent_genid = select_next_parent(archive, output_dir, domains)

    # Print results
    print(next_parent_genid)
