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

Run the dataset test with:

```bash
python -m unittest discover -s tests -v
```
