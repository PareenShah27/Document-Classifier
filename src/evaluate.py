import pandas as pd
import os
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import Any, Dict, Hashable, List

def evaluate_model(model: MultinomialNB | LogisticRegression | SVC, x_test: Any, y_test: Any, model_name: str, feature_type: str) -> Dict[str, Any]:
    """
    Predicts and calculates metrics.
    Returns a dictionary of results.
    """
    y_pred = model.predict(x_test)

    metrics = {
        "Model": model_name,
        "Feature Type": feature_type,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
        "F1 Score": round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4)
    }

    return metrics

def get_next_idx(output_dir: str, prefix: str = "metrics_") -> int:
    """
    Scans the output directory to find the next available run number.
    Example: if metrics_run_1.csv exists, returns 2.
    """
    max_run = 0
    pattern = re.compile(rf"{prefix}(\d+)\.csv")

    try:
        files = os.listdir(output_dir)
        for filename in files:
            match = pattern.match(filename)
            if match:
                run_num = int(match.group(1))
                if run_num > max_run:
                    max_run = run_num
    except FileNotFoundError:
        return 1

    return max_run + 1

def save_results(result_list: List[Dict[Hashable, Any]], output_dir: str = "results") -> None:
    """
    Saves list of metric dictionaries to metrics.csv
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.DataFrame(result_list)
    # Calculate next run number
    run_number = get_next_idx(output_dir)
    filename = f"metrics_run_{run_number}.csv"
    output_path = os.path.join(output_dir, filename)

    df.to_csv(output_path, index=False)

    print(f"\n[Run #{run_number}] Results saved to {output_path}")
    print("\n" + "="*80)
    print(df.to_string(index=False))
    print("="*80)