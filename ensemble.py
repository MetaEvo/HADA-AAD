
import os
import pandas as pd

from utils.gl_utils import load_archive_data, get_score


def ensemble(domain, task, generate_output_dir, split="train"):
    """
    Run ensemble on a single task.

    Args:
        domain (str): The domain of the task.
        task (dict): A task dictionary, with keys "question_id", and necessary input keys for the domain.
        generate_output_dir (str): The directory where the generated archive is stored.

    Returns:
        str: The prediction of the ensemble.
    """
    question_id = task["question_id"]

    # Get the best agent from archive
    archive_path = os.path.join(generate_output_dir, "archive.jsonl")
    archive_data = load_archive_data(archive_path, last_only=True)
    archive_genids = archive_data.get("archive", [])
    best_score, best_genid = -1, None
    for genid in archive_genids:
        score = get_score(domain, generate_output_dir, genid, split=split)
        if score is not None and score > best_score:
            best_score, best_genid = score, genid

    # Get the prediction from the best agent
    pred_dirname = f"{domain}_eval" if split == "train" else f"{domain}_eval_{split}"
    predictions_path = os.path.join(generate_output_dir, f"gen_{best_genid}/{pred_dirname}/predictions.csv")
    df = pd.read_csv(predictions_path)
    match = df.loc[df["question_id"] == question_id, "prediction"]
    if match.empty:
        prediction = None
    else:
        prediction = match.iloc[0]

    return prediction
