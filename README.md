# SmartCity Image Quality Analyzer

A simple full-stack application that uploads an image, detects quality problems, gives a 0–100 quality score, and stores previous results.

## Live application

- Frontend: https://analyzeimage.onrender.com/
- Backend API docs: https://smartcity-9eso.onrender.com/docs

## Detected image issues

The model detects acceptable images, blur, noise, compression, underexposure, and overexposure. It can return multiple issues when both model predictions and image statistics show defects.

## Datasets used

- **BSDS500:** 500 natural images stored in `Data/images`; these are used to create controlled synthetic degradations.
- **KADID-10k:** Real distorted images stored in `Data/kadid10k`; supported distortions and DMOS quality scores improve real-image performance.
- **BSDS ground truth:** Boundary annotations are stored in `Data/ground_truth`; they are not required by the quality-classification model.

## Dataset preparation

BSDS images are converted into six balanced classes using three severity levels. Source images remain in their original train, validation, or test split to prevent leakage.

```bash
python scripts/generate_dataset.py
python scripts/validate_dataset.py --dataset Data/generated
python scripts/extract_features.py --dataset Data/generated
```

KADID distortion IDs 01–03, 09–14, 16, and 17 map to blur, compression, noise, overexposure, and underexposure. Its 81 reference images are split 65/8/8 so variants of one reference never enter different splits.

```bash
python scripts/prepare_kadid.py
```

## Image features

Each image is represented by 12 explainable statistics: brightness mean/std, dark/bright pixel ratios, Laplacian variance, gradient strength, edge density, noise, entropy, saturation mean/std, and blockiness.

## Model

The saved model contains a Random Forest classifier for issue detection and a Random Forest regressor for the quality score. Both use 150 trees and fixed seed `42` for reproducible training.

Random Forest was selected because it works well with a small tabular feature set, supports nonlinear relationships, needs little preprocessing, and provides feature importance. It is also fast enough for CPU inference on Render.

Synthetic samples receive higher training weights to preserve clean/acceptable-image learning, while KADID adds realistic distortion patterns and DMOS score targets. The final training set contains 1,200 synthetic and 3,575 KADID rows.

```bash
python scripts/train_model.py
```

The model bundle is stored at `artifacts/quality_model.joblib`; Joblib uses Python pickle serialization efficiently for NumPy and scikit-learn objects. Evaluation details are stored in `reports/model_evaluation.json`.

## Current model results

| Held-out dataset | Accuracy | Macro F1 | Score MAE |
| --- | ---: | ---: | ---: |
| Synthetic BSDS | 75.58% | 0.7414 | 11.98 |
| KADID-10k | 65.91% | 0.5173 | 13.75 |

KADID macro F1 improved from 0.3384 with the previous synthetic-only model to 0.5173. The model artifact is about 46.76 MB and reports version `2.0.0`.

## Backend

The backend uses FastAPI, OpenCV, the trained Joblib model, and SQLite. It validates uploads, analyzes images, saves results, serves uploaded files, and returns standard HTTP status codes.

```bash
uvicorn backend.app.main:app --reload
```

Local API documentation is available at `http://localhost:8000/docs`.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check API and model version |
| `POST` | `/api/v1/analyses` | Upload and analyze one image |
| `GET` | `/api/v1/analyses` | List saved analyses |
| `GET` | `/api/v1/analyses/{id}` | Get one saved analysis |
| `GET` | `/uploads/{filename}` | Get an uploaded image |

JPEG, PNG, and WebP files up to 10 MB are supported. Important responses are `201` success, `400` unreadable image, `404` missing record, `413` too large, `415` unsupported type, and `422` invalid input.

## Backend configuration

Copy `.env.example` values into Render or your local environment. Available variables are `MODEL_PATH`, `DATABASE_PATH`, `UPLOAD_DIR`, `MAX_UPLOAD_MB`, and `FRONTEND_ORIGINS`.

## Frontend

The frontend uses plain HTML, Tailwind CSS through CDN, and vanilla JavaScript; React, Node.js, and a build step are not required. It provides drag-and-drop upload, preview, validation, results, statistics, API health, and history.

```bash
python -m http.server 5173 --directory frontend
```

Open `http://localhost:5173`; local mode uses `http://localhost:8000`, while deployed mode uses the Render backend. Set `window.IMAGE_QUALITY_API_URL` before `app.js` to override the API address.

## Local setup

Install Python dependencies, train the model if required, and run the backend and frontend in separate terminals.

```bash
python -m pip install -r requirements.txt
uvicorn backend.app.main:app --reload
python -m http.server 5173 --directory frontend
```

## Testing

The tests cover dataset generation, trained-model structure, API behavior, supported image formats, regression cases, and frontend files.

```bash
python -m pytest -q
```

## Render deployment

- Backend: create a Python Web Service with build command `pip install -r requirements.txt` and start command `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`.
- Frontend: create a Static Site with publish directory `frontend`; no build command is needed.
- CORS: set backend `FRONTEND_ORIGINS` to the deployed frontend URL, without a trailing slash.

## Project structure

- `backend/`: FastAPI application, model inference, SQLite storage, schemas, and configuration.
- `frontend/`: Static HTML, Tailwind styling, and vanilla JavaScript client.
- `scripts/`: Dataset generation, validation, feature extraction, KADID preparation, and training.
- `artifacts/`: Trained model bundle; generated feature CSV files remain ignored by Git.
- `reports/`: Dataset and model evaluation reports.
- `tests/`: Automated dataset, model, API, and frontend tests.

## Limitations

The model supports five defect types and is not a general object-detection system. Accuracy can decrease on unseen cameras, edits, mixed distortions, or image styles that differ greatly from BSDS500 and KADID-10k.
