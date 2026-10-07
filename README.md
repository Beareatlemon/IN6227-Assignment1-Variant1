_IN6227 Assignment 1 Variant 1 source_

The executable Jupyter notebook `IN6227-Assignment-1.ipynb` uses short section
notes and embedded charts for class balance, numeric distributions, parameter
stability, precision-recall/ROC curves, and confusion matrices. Its code uses
the same seed, preprocessing, model candidates, validation rule, test metrics,
and paired bootstrap as `assignment1_analysis.py`.

This script compares logistic regression and random forest classifiers across 12
documented parameter settings. It chooses a configuration within each family by
average precision on a stratified validation split, refits on the supplied
training file, and reports performance on the supplied test file. The notebook
shows all validation results and adds a five-fold stratified stability check of
the same 12 settings. It also shows precision, recall, F1 and error counts at
four illustrative thresholds on the original validation split; the supplied
test comparison remains fixed at 0.50. The archive includes `IN6227-parameter-search.csv`,
`IN6227-cross-validation.csv`, and `IN6227-cross-validation-folds.csv` so the
single-split and fold-level results can be inspected separately.

The expanded search and stability check were requested after the test results
from a smaller search had already been inspected. The selected configurations
did not change, but the test scores should be treated as descriptive rather
than a new independent confirmation of the extended analysis. The five-fold
check finds a different highest-mean random-forest leaf size by only 0.00108 AP;
the original selected setting is kept for the test comparison.

_Run_

Use Python 3.10 or newer and install dependencies:

```sh
python -m pip install -r requirements.txt
python assignment1_analysis.py --data-dir PATH_TO_DATASET_DIRECTORY --output results.json
python cross_validate_candidates.py --data-dir PATH_TO_DATASET_DIRECTORY
```

The dataset directory must contain `train.csv` and `test.csv`, each with a `label` column containing `yes` and `no`. The original data are not included in this archive. The printed and saved metrics are reproducible with random seed 42.
