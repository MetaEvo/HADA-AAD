import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util

from domains.harness import get_dataset
from domains.report import report
from ensemble import ensemble


def get_ensemble_score(
    domain,
    generate_output_dir,
    num_samples=-1,
    max_workers=5,
    subset="",
):
    try:
        # Get dataset
        dataset = get_dataset(domain=domain, subset=subset)
        if num_samples > 0:
            dataset = dataset[:num_samples]
        split = subset.split("_")[-1]

        # Dynamically import functions based on the domain
        utils_module_path = f"domains.{domain}.utils"
        utils_module = importlib.import_module(utils_module_path)
        format_input_dict = utils_module.format_input_dict

        def predict_row(row):
            task = {"question_id": row["question_id"]}
            task.update(format_input_dict(row))
            return ensemble(domain, task, generate_output_dir, split=split)

        # Run ensemble in parallel
        predictions = [None] * len(dataset)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(predict_row, row): idx
                for idx, (_, row) in enumerate(dataset.iterrows())
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    predictions[idx] = future.result()
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    predictions[idx] = None

        # Save predictions
        dataset['prediction'] = predictions
        predictions_path = os.path.join(generate_output_dir, f'predictions_ensemble_{domain}_{split}.csv')
        dataset.to_csv(predictions_path, index=False)

        # Make report
        report_json, report_path = report(generate_output_dir, domain, suffix=f"_ensemble_{domain}_{split}")
        score = report_json.get("overall_accuracy", None)

    except Exception as e:
        print(f"Error in get_ensemble_score: {e}")
        predictions_path = None
        report_path = None
        score = None

    return score, predictions_path, report_path


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, choices=["search_arena", "paper_review"], required=True, help="Domain to evaluate")
    parser.add_argument("--generate_output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--subset", type=str, default="_filtered_100_train")
    args = parser.parse_args()

    # Get ensemble score
    score, predictions_path, report_path = get_ensemble_score(
        args.domain,
        args.generate_output_dir,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        subset=args.subset,
    )

    # Print results
    print(score)
    print(predictions_path)
    print(report_path)
