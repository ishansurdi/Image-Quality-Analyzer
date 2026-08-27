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

## Backend API

Start the FastAPI development server from the project root:

```bash
uvicorn backend.app.main:app --reload
```

The API is available at `http://localhost:8000`, with interactive documentation
at `http://localhost:8000/docs`.

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service and loaded-model status |
| `POST` | `/api/v1/analyses` | Upload and analyze one image |
| `GET` | `/api/v1/analyses` | Paginated analysis history |
| `GET` | `/api/v1/analyses/{id}` | Retrieve one previous result |
| `GET` | `/uploads/{filename}` | Retrieve a stored uploaded image |

Upload example:

```bash
curl -X POST http://localhost:8000/api/v1/analyses \
  -F "image=@sample.jpg"
```

Supported formats are JPEG, PNG, and WebP. The default upload limit is 10 MB.
The API uses `201` for a successful analysis, `400` for unreadable images, `404`
for missing results, `413` for oversized uploads, `415` for unsupported media,
and `422` for invalid request parameters.

Configuration is read from `MODEL_PATH`, `DATABASE_PATH`, `UPLOAD_DIR`, and
`MAX_UPLOAD_MB`. Their defaults are shown in `.env.example`. SQLite tables are
created automatically when the application starts.

## Frontend

The frontend uses plain HTML, Tailwind CSS, and vanilla JavaScript. It has no
framework, package installation, or build step. Start it in a second terminal:

```bash
python -m http.server 5173 --directory frontend
```

Open `http://localhost:5173`. The page connects to port `8000` on the current
hostname and provides image preview, drag-and-drop upload, analysis results,
loading and error states, explainable statistics, and previous-analysis history.

For a different backend address, set `window.IMAGE_QUALITY_API_URL` before
`app.js` loads. The backend permits local frontend origins on ports `5173` and
`3000`.
