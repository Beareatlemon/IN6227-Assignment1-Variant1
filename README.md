# IN6227 Assignment 1 Variant 1 source

This script compares logistic regression and random forest classifiers. It chooses a configuration within each family by average precision on a stratified validation split, refits on the supplied training file, and reports performance on the supplied test file.

## Run

Use Python 3.12 and install dependencies:

```sh
python -m pip install -r requirements.txt
python assignment1_analysis.py --data-dir PATH_TO_DATASET_DIRECTORY --output results.json
```

The dataset directory must contain `train.csv` and `test.csv`, each with a `label` column containing `yes` and `no`. The original data are not included in this archive. The printed and saved metrics are reproducible with random seed 42.

