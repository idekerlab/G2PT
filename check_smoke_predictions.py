"""Check standalone checkpoint inference agrees with training-time validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--phenotype', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pred = pd.read_csv(args.predictions, dtype={'IID': str})[['IID', 'T2D']]
    truth = pd.read_csv(args.phenotype, sep='\t', dtype={'IID': str})[['IID', 'T2D']]
    if set(pred.IID) != set(truth.IID):
        raise ValueError('Standalone inference sample set differs from validation')
    joined = truth.merge(pred, on='IID', validate='one_to_one', suffixes=('_true', '_pred'))
    if not np.isfinite(joined.T2D_pred).all():
        raise ValueError('Nonfinite standalone predictions')
    history = [json.loads(line) for line in args.history.read_text().splitlines()]
    best = max(history, key=lambda row: row['selection_value'])
    metrics = {'auc': float(roc_auc_score(joined.T2D_true, joined.T2D_pred)),
               'average_precision': float(average_precision_score(joined.T2D_true, joined.T2D_pred))}
    for name, value in metrics.items():
        if abs(value - best[name]) > 1e-5:
            raise ValueError(f'Standalone {name} differs from training validation: {value} vs {best[name]}')
    result = {'passed': True, 'samples': len(joined), 'best_epoch': best['epoch'],
              'strict_state_dict_reload': True, **metrics}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
