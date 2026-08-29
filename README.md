# SmartCity Image Quality Analyzer

A simple full-stack application that uploads an image, detects quality problems, gives a 0–100 quality score, and stores previous results.

## Live application

- Frontend: https://analyzeimage.onrender.com/
- Backend API docs: https://smartcity-9eso.onrender.com/docs

## Detected image issues

The model detects acceptable images, blur, noise, compression, underexposure, and overexposure. It can return multiple issues when both model predictions and image statistics show defects.

## Severe degradation decision

An image receives `POTENTIALLY_DEFECTIVE` when its quality score is below 40; this represents severe quality degradation. Unreadable or structurally corrupted files are rejected with HTTP `400` instead of being classified.

The system evaluates whole-image quality and does not locate physical defects such as cracks or scratches. Compression is treated as a quality issue, while localized defect detection remains outside the current model scope.

## Datasets used

- **BSDS500:** 500 natural images stored in `Data/images`; these are used to create controlled synthetic degradations.
- **KADID-10k:** Real distorted images stored in `Data/kadid10k`; supported distortions and DMOS quality scores improve real-image performance.
- **BSDS ground truth:** Boundary annotations are stored in `Data/ground_truth`; they are not required by the quality-classification model.

## Dataset sources and citation

- BSDS500 download: [Berkeley Segmentation Dataset 500 on Kaggle](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500)
- KADID-10k download and details: [Official KADID-10k database](https://database.mmsp-kn.de/kadid-10k-database.html)

KADID-10k is freely available to the research community. Research using the database or distortion code should cite:

```bibtex
@inproceedings{kadid10k,
  title={KADID-10k: A Large-scale Artificially Distorted IQA Database},
  author={Lin, Hanhe and Hosu, Vlad and Saupe, Dietmar},
  booktitle={2019 Tenth International Conference on Quality of Multimedia Experience (QoMEX)},
  pages={1--3},
  year={2019},
  organization={IEEE}
}
```

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

A small histogram gradient-boosting classifier estimates compression severity directly. Its high, medium, and low predictions cap inconsistent compression scores at 35, 60, and 80.

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

KADID macro F1 improved from 0.3384 with the previous synthetic-only model to 0.5173. The model artifact is about 49.57 MB and reports version `2.1.0`.

## Incorrect and uncertain predictions

- A bright, soft image can be reported as both blur and overexposure because both conditions affect edge and brightness features.
- A clean image with little texture may look blurred to the model, while intentional dark or bright photography may be treated as an exposure problem.
- Noise and compression can be confused because both create high-frequency artifacts; low confidence indicates that the selected class is uncertain.
- Mixed or unfamiliar distortions may receive an inaccurate primary class because training covers only the five documented issue types.

Confidence is the Random Forest probability of the predicted class, not a guarantee of correctness. Predictions near competing class probabilities should be reviewed together with severity, score, and image statistics.

## Sample images

These held-out BSDS and KADID examples demonstrate the six conditions used by the application.

| Condition | Sample |
| --- | --- |
| Acceptable | [acceptable.jpg](samples/acceptable.jpg) |
| Blur | [blur.jpg](samples/blur.jpg) |
| Noise | [noise.jpg](samples/noise.jpg) |
| Underexposure | [underexposure.jpg](samples/underexposure.jpg) |
| Overexposure | [overexposure.jpg](samples/overexposure.jpg) |
| Compression | [compression.png](samples/compression.png) |

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

Upload an image with:

```bash
curl -X POST http://localhost:8000/api/v1/analyses \
  -F "image=@samples/blur.jpg"
```

Example response fields:

```json
{
  "quality_score": 55.2,
  "quality_label": "DEGRADED",
  "issues": [{"type": "blur", "severity": "medium", "confidence": 0.78}],
  "statistics": {"brightness_mean": 126.4, "laplacian_variance": 18.7}
}
```

## Backend configuration

Copy `.env.example` values into Render or your local environment. Available variables are `MODEL_PATH`, `DATABASE_PATH`, `UPLOAD_DIR`, `MAX_UPLOAD_MB`, and `FRONTEND_ORIGINS`.

## Database setup

SQLite is used because it is lightweight, serverless, and sufficient for a single-instance project. Python includes its driver, so no separate database server or package is required.

FastAPI initializes the database during startup. It creates the parent directory and the `analyses` table automatically with `CREATE TABLE IF NOT EXISTS`, so no manual migration command is needed.

Each row stores the original and saved filenames, content type, image dimensions, quality score, quality label, detected issues, image statistics, and creation time. Issues and statistics are stored as JSON text and converted back to JSON in API responses.

After successful inference, `POST /api/v1/analyses` saves the result and returns it. `GET /api/v1/analyses` provides paginated history, while `GET /api/v1/analyses/{id}` retrieves one record.

All SQL values use parameterized queries, and every connection is committed and closed automatically. Uploaded image files are stored in `UPLOAD_DIR`; SQLite stores their generated filenames rather than the image bytes.

`DATABASE_PATH` controls the SQLite file and defaults to `backend/data/analyses.db`. For local development, start the API normally and the database will be created on the first startup.

Render’s default filesystem is ephemeral, so the database and uploaded files can disappear after a restart or redeployment. For permanent history, attach a persistent disk mounted at `/var/data` and set:

```text
DATABASE_PATH=/var/data/analyses.db
UPLOAD_DIR=/var/data/uploads
```

This SQLite design is suitable for the current single Render instance. PostgreSQL should be used later if the application needs multiple backend instances or heavy concurrent writes.

## Model loading and inference

During FastAPI startup, `ImageAnalyzer` loads `artifacts/quality_model.joblib` once and keeps its models in application memory. Each request extracts the same 12 features, predicts issues and score, then applies learned compression calibration and severity rules.

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
- `samples/`: Held-out example images for every supported quality condition.
- `tests/`: Automated dataset, model, API, and frontend tests.

## Limitations

The model supports five defect types and is not a general object-detection system. Accuracy can decrease on unseen cameras, edits, mixed distortions, or image styles that differ greatly from BSDS500 and KADID-10k.
