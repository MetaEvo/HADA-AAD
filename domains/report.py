import os
import shutil
import sys

# Add the project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import importlib.util
import json
import pandas as pd


def report(
    dname,
    domain,
    suffix="",  # suffix to the predictions{suffix}.csv filename
):
    # Dynamically import functions based on the domain
    utils_prefix = domain.split("_", 1)[1] + "_" if domain.startswith("imo_") else ""
    domain_folder = domain.split('_')[0] if "imo_" in domain else domain
    utils_module_path = f"domains.{domain_folder}.{utils_prefix}utils"
    utils_module = importlib.import_module(utils_module_path)
    ground_truth_key = utils_module.GROUND_TRUTH_KEY
    question_id_col = utils_module.QUESTION_ID

    # Load and process the data
    path = os.path.join(dname, f"predictions{suffix}.csv")
    df = pd.read_csv(path, dtype=str)
    df = df[df["prediction"] != ""].copy()  # Filter out rows with NA predictions
    df["prediction"] = df["prediction"].str.strip().str.lower()
    df[ground_truth_key] = df[ground_truth_key].str.strip().str.lower()
    df["match"] = df[ground_truth_key] == df["prediction"]

    # Calculate overall accuracy
    accuracy = df["match"].mean()
    total_correct = int(df["match"].sum())
    total = len(df)
    print(f"Accuracy: {accuracy:.3f}, Total correct: {total_correct} / {total}")

    # Calculate Mean Absolute Error (MAE)
    if domain == "imo_grading":
        try:
            max_error = 7
            df["prediction_points"] = (
                df["prediction"]
                .map({"incorrect": 0, "partial": 1, "almost": 6, "correct": 7})
                .astype(float)
            )
            df['reward_points'] = (
                df[ground_truth_key]
                .map({"incorrect": 0, "partial": 1, "almost": 6, "correct": 7})
                .astype(float)
            )
            df["error"] = abs(df["prediction_points"] - df['reward_points'])
            df["error"] = df["error"].fillna(max_error)
            mae = df["error"].mean() / max_error
            print(f"Normalized Mean Absolute Error (MAE): {mae:.3f}")
        except Exception as e:
            mae = None
            print("Error: Could not calculate MAE for IMO grading")
            print(e)

    # Accuracy by label
    label_accuracies = df.groupby(ground_truth_key)["match"].mean()
    label_counts = df[ground_truth_key].value_counts()

    print("\nAccuracy by label:")
    label_report = {}
    labels = set(df[ground_truth_key].unique())
    for label in labels:
        # True Positive (TP): predicted = label and actual = label
        tp = ((df["prediction"] == label) & (df[ground_truth_key] == label)).sum()
        # False Positive (FP): predicted = label but actual ≠ label
        fp = ((df["prediction"] == label) & (df[ground_truth_key] != label)).sum()
        # False Negative (FN): predicted ≠ label but actual = label
        fn = ((df["prediction"] != label) & (df[ground_truth_key] == label)).sum()
        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        correct_label = tp
        total_label = int(label_counts.get(label, 0))

        print(
            f"  Label: {label} - Precision: {precision:.3f}, Recall: {recall:.3f}, Correct: {correct_label} / {total_label}"
        )
        label_report[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "correct": int(correct_label),
            "total": int(total_label),
        }

    # Compute label distributions for winner and prediction
    winner_distribution = df[ground_truth_key].value_counts(normalize=True).to_dict()
    prediction_distribution = df["prediction"].value_counts(normalize=True).to_dict()

    print("\nDistribution of ground truth labels:")
    for label, freq in winner_distribution.items():
        print(f"  {label}: {freq:.3f}")

    print("\nDistribution of prediction labels:")
    for label, freq in prediction_distribution.items():
        print(f"  {label}: {freq:.3f}")

    # Compute expected random guess accuracy using winner label distribution
    random_guess_accuracy = sum(p**2 for p in winner_distribution.values())
    print(
        f"\nExpected random guess accuracy (based on ground_truth_key label distribution): {random_guess_accuracy:.3f}"
    )

    # Build the report dictionary
    report = {
        "overall_accuracy": float(accuracy),
        **({"normalized_mean_absolute_error": mae} if domain == "imo_grading" else {}),
        "total_correct": total_correct,
        "total": total,
        "accuracy_by_ground_truth": label_report,
        "label_distribution": {
            "ground_truth": winner_distribution,
            "prediction": prediction_distribution,
        },
        "random_guess_accuracy": random_guess_accuracy,
        "question_ids_failed": [
            row[question_id_col] for _, row in df[df["match"] == False].iterrows()
        ],
        "question_ids_passed": [
            row[question_id_col] for _, row in df[df["match"] == True].iterrows()
        ],
    }

    # Save the report as a JSON file
    report_path = os.path.join(dname, f"report{suffix}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    return report, report_path

def report_imo_proof(dname):
    # Grade the generated proofs
    from domains.harness import harness
    output_folder = harness(
        domain="imo_proof_grading",
        agent_path="proofgrader.task_agent",
        proofs_dname=dname,
        output_dir=dname,
        run_id="gradings",
    )
    # Get report of grading
    from domains.imo.proof_eval import report_proof_grading
    report_proof_grading(dname=output_folder)
    # Move report to one dir above
    src = os.path.join(output_folder, "report.json")
    dst = os.path.join(os.path.dirname(output_folder), "report.json")
    shutil.move(src, dst)


def main():
    """Main entry point for report."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate report for the evaluation.")
    parser.add_argument("--dname", required=True, help="Path to harness outputs")
    parser.add_argument(
        "--domain",
        type=str,
        choices=[
            "search_arena",
            "paper_review",
            "balrog_babyai",
            "balrog_babaisai",
            "balrog_minihack",
            "balrog_nle",
            "genesis_go2walking",
            "genesis_go2walkback",
            "genesis_go2hop",
            "imo_grading",
            "imo_proof",
            "bbob_unconstrained",
            "bbob_constrained",
            "metabox_mo",
        ],
        required=True,
        help="Domain to evaluate",
    )
    # parser.add_argument("--proofgrader_repo", default="./proofgrader_repo", help="Path to repository containing the proof grader that we want to use for imo_proof domain")
    args = parser.parse_args()

    domain = args.domain

    # Human preferences domains
    if domain == "bbob_unconstrained":
        # Use the bbob-specific reporter
        from domains.bbob_unconstrained.report import report_bbob_unconstrained
        report_bbob_unconstrained(args.dname)
    elif domain == "bbob_constrained":
        # Use the bbob_constrained-specific reporter
        from domains.bbob_constrained.report import report_bbob_constrained
        report_bbob_constrained(args.dname)
    elif domain == "metabox_mo":
        # Use the metabox_mo-specific reporter
        from domains.metabox_mo.report import generate_report
        import pandas as pd
        predictions = pd.read_csv(os.path.join(args.dname, "predictions.csv"))
        generate_report(args.dname, predictions, mode="test")

    elif domain in ["search_arena", "paper_review", "imo_grading"]:
        report(dname=args.dname, domain=args.domain)

    elif domain == "imo_proof":
        report_imo_proof(dname=args.dname)

    # Balrog game domains
    elif "balrog" in domain:
        from domains.balrog.eval import report_balrog
        report_balrog(output_dir=args.dname)

    # Genesis Robotic Control Domains
    elif "genesis" in domain:
        from domains.genesis.eval import report_genesis
        report_genesis(output_dir=args.dname)


if __name__ == "__main__":
    main()