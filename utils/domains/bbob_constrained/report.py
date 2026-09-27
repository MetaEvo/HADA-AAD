import os
import json
import math


def report_bbob_constrained(dname):
    """Read `predictions.csv` created by the harness and produce a report.json.

    The predictions file is expected at `os.path.join(dname, 'predictions.csv')`
    with a column `prediction` containing the final gbest_f (as number or string).
    
    评分方式：
    - prediction 格式为 "score|gbest_f|initial_gbest_f|opt_val"
    - score = sigmoid(RI)，其中 RI = (f(x0) - f(x*)) / (f(xbest) - f(x*))
    """
    import pandas as pd

    path = os.path.join(dname, "predictions.csv")
    df = pd.read_csv(path, dtype=str)
    if df.empty:
        report = {"score": 0.0, "mean_gbest_f": None, "total": 0}
        print(json.dumps(report, indent=2))
        with open(os.path.join(dname, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        return report, os.path.join(dname, "report.json")

    # parse prediction values
    scores = []
    gbest_f_vals = []
    
    for pred in df['prediction']:
        # 尝试解析新格式 "score|gbest_f|initial_gbest_f|opt_val"
        parts = str(pred).split('|')
        if len(parts) == 4:
            score = float(parts[0])
            gbest_f = float(parts[1])
            scores.append(score)
            gbest_f_vals.append(gbest_f)

    total = len(df)
    if len(scores) == 0:
        report = {"score": 0.0, "mean_gbest_f": None, "total": total}
    else:
        mean_score = float(sum(scores) / len(scores))
        mean_gbest = float(sum(gbest_f_vals) / len(gbest_f_vals)) if gbest_f_vals else None
        report = {"score": float(mean_score), "mean_gbest_f": float(mean_gbest) if mean_gbest is not None else None, "total": total, "mean_score": float(mean_score)}

    report_path = os.path.join(dname, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return report, report_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dname', required=True)
    args = parser.parse_args()
    report_bbob_constrained(args.dname)