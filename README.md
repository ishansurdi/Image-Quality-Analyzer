# SmartCity Image Quality Analyzer

A full-stack application that identifies common image-quality problems using
computer-vision features and a machine-learning model.

## Dataset generation

Place the BSDS500 images under `Data/images/train`, `Data/images/val`, and
`Data/images/test`. Install the dependencies and generate a small sample:

```bash
python -m pip install -r requirements.txt
python scripts/generate_dataset.py --max-images-per-split 10
```

Generated images and `metadata.csv` are written to `Data/generated`. Existing
outputs produced with the same seed and severity are reused. To process every
source image, omit `--max-images-per-split`:

```bash
python scripts/generate_dataset.py
```

Validate the dataset, extract features, and train the models:

```bash
python scripts/validate_dataset.py --dataset Data/generated
python scripts/extract_features.py --dataset Data/generated
python scripts/train_model.py
```

The training script creates:

- `artifacts/quality_model.joblib`: classifier, quality regressor, and model metadata
- `reports/model_evaluation.json`: classification, regression, confusion-matrix,
  and feature-importance results

## Current model results

The source-separated test set contains 1,200 images. No original BSDS500 image
appears in more than one dataset split.

| Metric | Result |
| --- | ---: |
| Classification accuracy | 0.8200 |
| Macro precision | 0.8225 |
| Macro recall | 0.8200 |
| Macro F1-score | 0.8183 |
| Quality-score MAE | 9.6628 |
| Quality-score RMSE | 13.7515 |
| Quality-score R² | 0.6972 |

The model uses 12 explainable image statistics. Its most important features are
bright-pixel ratio, Laplacian variance, blockiness, estimated noise, and mean
brightness.

Run the dataset test with:

```bash
python -m unittest discover -s tests -v
```
