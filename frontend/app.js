const LOCAL_HOSTNAMES = new Set(["localhost", "127.0.0.1"]);
const DEFAULT_API_URL = LOCAL_HOSTNAMES.has(window.location.hostname)
  ? `${window.location.protocol}//${window.location.hostname}:8000`
  : "https://smartcity-9eso.onrender.com";
const API_BASE_URL = window.IMAGE_QUALITY_API_URL || DEFAULT_API_URL;
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

const elements = {
  healthDot: document.querySelector("#health-dot"),
  healthText: document.querySelector("#health-text"),
  dropZone: document.querySelector("#drop-zone"),
  imageInput: document.querySelector("#image-input"),
  uploadPrompt: document.querySelector("#upload-prompt"),
  imagePreview: document.querySelector("#image-preview"),
  fileName: document.querySelector("#file-name"),
  uploadError: document.querySelector("#upload-error"),
  clearButton: document.querySelector("#clear-button"),
  analyzeButton: document.querySelector("#analyze-button"),
  analyzeButtonText: document.querySelector("#analyze-button-text"),
  loadingIcon: document.querySelector("#loading-icon"),
  emptyResult: document.querySelector("#empty-result"),
  analysisResult: document.querySelector("#analysis-result"),
  qualityLabel: document.querySelector("#quality-label"),
  qualityScore: document.querySelector("#quality-score"),
  scoreBar: document.querySelector("#score-bar"),
  issuesList: document.querySelector("#issues-list"),
  statisticsList: document.querySelector("#statistics-list"),
  historyLoading: document.querySelector("#history-loading"),
  historyEmpty: document.querySelector("#history-empty"),
  historyError: document.querySelector("#history-error"),
  historyList: document.querySelector("#history-list"),
  refreshHistory: document.querySelector("#refresh-history"),
};

const statisticLabels = {
  brightness_mean: "Brightness",
  brightness_std: "Contrast",
  laplacian_variance: "Sharpness",
  noise_estimate: "Noise estimate",
  entropy: "Entropy",
  blockiness: "Blockiness",
};

let selectedFile = null;
let previewUrl = null;

function showError(message) {
  elements.uploadError.textContent = message;
  elements.uploadError.classList.remove("hidden");
}

function clearError() {
  elements.uploadError.textContent = "";
  elements.uploadError.classList.add("hidden");
}

function validateFile(file) {
  if (!ALLOWED_TYPES.has(file.type)) {
    return "Choose a JPEG, PNG, or WebP image.";
  }
  if (file.size > MAX_FILE_SIZE) {
    return "The selected image is larger than 10 MB.";
  }
  return null;
}

function selectFile(file) {
  const error = validateFile(file);
  if (error) {
    clearSelection();
    showError(error);
    return;
  }

  clearError();
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  elements.imagePreview.src = previewUrl;
  elements.imagePreview.classList.remove("hidden");
  elements.uploadPrompt.classList.add("hidden");
  elements.fileName.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  elements.fileName.classList.remove("hidden");
  elements.clearButton.classList.remove("hidden");
  elements.analyzeButton.disabled = false;
}

function clearSelection() {
  selectedFile = null;
  elements.imageInput.value = "";
  elements.imagePreview.src = "";
  elements.imagePreview.classList.add("hidden");
  elements.uploadPrompt.classList.remove("hidden");
  elements.fileName.classList.add("hidden");
  elements.clearButton.classList.add("hidden");
  elements.analyzeButton.disabled = true;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
}

function setAnalyzing(isAnalyzing) {
  elements.analyzeButton.disabled = isAnalyzing || !selectedFile;
  elements.loadingIcon.classList.toggle("hidden", !isAnalyzing);
  elements.analyzeButtonText.textContent = isAnalyzing ? "Analyzing..." : "Analyze image";
}

function clearResult() {
  elements.analysisResult.classList.add("hidden");
  elements.emptyResult.classList.remove("hidden");
}

async function requestJson(url, options = {}) {
  const response = await fetch(`${API_BASE_URL}${url}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "The request could not be completed.");
  }
  return data;
}

function qualityColors(label) {
  if (label === "ACCEPTABLE") return ["text-emerald-700", "bg-emerald-500"];
  if (label === "DEGRADED") return ["text-amber-700", "bg-amber-500"];
  return ["text-red-700", "bg-red-500"];
}

function renderIssues(issues) {
  elements.issuesList.replaceChildren();
  if (issues.length === 0) {
    const message = document.createElement("p");
    message.className = "rounded-lg bg-emerald-50 px-4 py-3 text-sm font-medium text-emerald-700";
    message.textContent = "No significant quality issue detected.";
    elements.issuesList.append(message);
    return;
  }

  for (const issue of issues) {
    const item = document.createElement("div");
    item.className = "flex items-center justify-between rounded-lg border border-slate-200 px-4 py-3";

    const description = document.createElement("div");
    const name = document.createElement("p");
    name.className = "font-medium capitalize";
    name.textContent = issue.type.replaceAll("_", " ");
    const severity = document.createElement("p");
    severity.className = "mt-0.5 text-xs capitalize text-slate-500";
    severity.textContent = `${issue.severity} severity`;
    description.append(name, severity);

    const confidenceDetails = document.createElement("div");
    confidenceDetails.className = "text-right";
    const issueConfidence = document.createElement("p");
    issueConfidence.className = "text-sm font-semibold text-slate-700";
    issueConfidence.textContent = `${Math.round(issue.confidence * 100)}% issue confidence`;
    confidenceDetails.append(issueConfidence);

    if (Number.isFinite(issue.severity_confidence)) {
      const severityConfidence = document.createElement("p");
      severityConfidence.className = "mt-0.5 text-xs text-slate-500";
      severityConfidence.textContent = `${Math.round(issue.severity_confidence * 100)}% severity confidence`;
      confidenceDetails.append(severityConfidence);
    }

    item.append(description, confidenceDetails);
    elements.issuesList.append(item);
  }
}

function renderStatistics(statistics) {
  elements.statisticsList.replaceChildren();
  for (const [name, label] of Object.entries(statisticLabels)) {
    const item = document.createElement("div");
    item.className = "rounded-lg bg-slate-50 px-3 py-3";
    const term = document.createElement("dt");
    term.className = "text-xs text-slate-500";
    term.textContent = label;
    const value = document.createElement("dd");
    value.className = "mt-1 font-semibold text-slate-800";
    value.textContent = Number(statistics[name]).toFixed(2);
    item.append(term, value);
    elements.statisticsList.append(item);
  }
}

function renderResult(result) {
  elements.emptyResult.classList.add("hidden");
  elements.analysisResult.classList.remove("hidden");
  elements.qualityLabel.textContent = result.quality_label.replaceAll("_", " ");
  elements.qualityScore.textContent = Math.round(result.quality_score);

  const [textColor, barColor] = qualityColors(result.quality_label);
  elements.qualityLabel.className = `mt-1 text-xl font-bold ${textColor}`;
  elements.scoreBar.className = `h-full rounded-full transition-all duration-500 ${barColor}`;
  elements.scoreBar.style.width = `${result.quality_score}%`;
  renderIssues(result.issues);
  renderStatistics(result.statistics);
}

async function analyzeSelectedImage() {
  if (!selectedFile) return;
  clearError();
  clearResult();
  setAnalyzing(true);
  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const result = await requestJson("/api/v1/analyses", {
      method: "POST",
      body: formData,
    });
    renderResult(result);
    await loadHistory();
  } catch (error) {
    showError(error.message);
  } finally {
    setAnalyzing(false);
  }
}

function createHistoryItem(result) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "flex w-full items-center gap-4 px-5 py-4 text-left transition hover:bg-slate-50 sm:px-6";
  button.addEventListener("click", () => {
    renderResult(result);
    elements.analysisResult.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  const image = document.createElement("img");
  image.src = `${API_BASE_URL}${result.image_url}`;
  image.alt = "";
  image.loading = "lazy";
  image.className = "h-14 w-14 rounded-lg bg-slate-100 object-cover";

  const details = document.createElement("div");
  details.className = "min-w-0 flex-1";
  const filename = document.createElement("p");
  filename.className = "truncate text-sm font-semibold";
  filename.textContent = result.original_filename;
  const createdAt = document.createElement("p");
  createdAt.className = "mt-1 text-xs text-slate-500";
  createdAt.textContent = new Date(result.created_at).toLocaleString();
  details.append(filename, createdAt);

  const score = document.createElement("div");
  score.className = "text-right";
  const scoreValue = document.createElement("p");
  scoreValue.className = "text-lg font-bold";
  scoreValue.textContent = Math.round(result.quality_score);
  const label = document.createElement("p");
  label.className = "mt-0.5 text-xs text-slate-500";
  label.textContent = result.quality_label.replaceAll("_", " ");
  score.append(scoreValue, label);

  button.append(image, details, score);
  return button;
}

async function loadHistory() {
  elements.historyLoading.classList.remove("hidden");
  elements.historyEmpty.classList.add("hidden");
  elements.historyError.classList.add("hidden");
  elements.historyList.replaceChildren();

  try {
    const history = await requestJson("/api/v1/analyses?limit=10&offset=0");
    elements.historyEmpty.classList.toggle("hidden", history.items.length !== 0);
    for (const result of history.items) {
      elements.historyList.append(createHistoryItem(result));
    }
  } catch (error) {
    elements.historyError.textContent = error.message;
    elements.historyError.classList.remove("hidden");
  } finally {
    elements.historyLoading.classList.add("hidden");
  }
}

async function checkHealth() {
  try {
    const health = await requestJson("/health");
    elements.healthDot.className = "h-2.5 w-2.5 rounded-full bg-emerald-500";
    elements.healthText.textContent = `Service online · Model ${health.model_version}`;
  } catch {
    elements.healthDot.className = "h-2.5 w-2.5 rounded-full bg-red-500";
    elements.healthText.textContent = "Service unavailable";
  }
}

elements.imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) selectFile(file);
});
elements.clearButton.addEventListener("click", clearSelection);
elements.analyzeButton.addEventListener("click", analyzeSelectedImage);
elements.refreshHistory.addEventListener("click", loadHistory);

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("border-brand-500", "bg-brand-50");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("border-brand-500", "bg-brand-50");
  });
}
elements.dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) selectFile(file);
});

checkHealth();
loadHistory();
