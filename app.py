import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import os
import shutil
import tempfile
from PyPDF2 import PdfReader
import json
from datetime import datetime
from markupsafe import Markup  # <-- Add this import

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Register tojson filter for Jinja2
import jinja2
import json as pyjson
templates.env.filters["tojson"] = lambda value: Markup(pyjson.dumps(value))

# Load models and vectorizer
MODELS = {
    "knn": joblib.load("model/knn_model.joblib"),
    "naive_bayes": joblib.load("model/naive_bayes_model.joblib"),
    "svm": joblib.load("model/svm_model.joblib"),
    "logistic_regression": joblib.load("model/logistic_regression_model.joblib"),
    "random_forest": joblib.load("model/random_forest_model.joblib"),
}
VECTORIZER = joblib.load("model/tfidf_vectorizer.joblib")

# You may want to load label encoder if you want to map class indices to names
try:
    import pandas as pd
    # Load the dataset to get class names (assuming same as training)
    df = pd.read_csv("model/ProjectDataset.csv")
    class_names = sorted(df["Category"].unique())
except Exception:
    class_names = [str(i) for i in range(len(MODELS["knn"].classes_))]

RECENT_UPLOADS_PATH = "recent_uploads.json"

def load_recent_uploads():
    if os.path.exists(RECENT_UPLOADS_PATH):
        with open(RECENT_UPLOADS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_recent_uploads(recent_uploads):
    with open(RECENT_UPLOADS_PATH, "w", encoding="utf-8") as f:
        json.dump(recent_uploads, f, ensure_ascii=False, indent=2)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    recent_uploads = load_recent_uploads()
    return templates.TemplateResponse(
        "intelli_cv_classifier.html",
        {"request": request, "recent_uploads": recent_uploads}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    resume: UploadFile = File(...),
    model: str = Form(...)
):
    # Save uploaded file to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(resume.file, tmp)
        tmp_path = tmp.name

    filename = resume.filename

    # Extract text from PDF
    try:
        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        os.remove(tmp_path)
        recent_uploads = load_recent_uploads()
        return templates.TemplateResponse(
            "intelli_cv_classifier.html",
            {"request": request, "prediction": f"<div style='color:red;'>Failed to extract text: {str(e)}</div>", "recent_uploads": recent_uploads}
        )
    finally:
        os.remove(tmp_path)

    if not text.strip():
        recent_uploads = load_recent_uploads()
        return templates.TemplateResponse(
            "intelli_cv_classifier.html",
            {"request": request, "prediction": "<div style='color:red;'>No text found in PDF.</div>", "recent_uploads": recent_uploads}
        )

    # Preprocess text (should match training preprocessing)
    import re
    def clean(text):
        text = re.sub(r'http\S+\s*', ' ', text)
        text = re.sub(r'RT|cc', ' ', text)
        text = re.sub(r'#\S+', '', text)
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'[%s]' % re.escape("""!"#$&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7f]', r' ', text)
        return text

    clean_text = clean(text)
    X = VECTORIZER.transform([clean_text])

    clf = MODELS.get(model)
    if clf is None:
        prediction = "<div style='color:red;'>Invalid model selected.</div>"
        pred_label = "Invalid"
        confidence_scores = {}
    else:
        pred = clf.predict(X)[0]
        try:
            label = class_names[pred]
        except Exception:
            label = str(pred)
        pred_label = label
        # Beautified prediction output
        prediction = f"<div class='predicted-class-box'>{label}</div>"

        # For random forest, show confidence scores in a table
        confidence_scores = {}
        if model == "random_forest" and hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            scores = {class_names[i]: float(proba[i]) for i in range(len(proba))}
            confidence_scores = scores
            prediction += """
            <div style="margin-top:22px;">
                <table class="confidence-table">
                    <tr>
                        <th>Class</th>
                        <th>Confidence</th>
                    </tr>
            """
            for k, v in sorted(scores.items(), key=lambda x: -x[1]):
                prediction += f"<tr><td>{k}</td><td>{v:.2%}</td></tr>"
            prediction += '</table></div>'

    # Save to recent uploads
    recent_uploads = load_recent_uploads()
    upload_entry = {
        "filename": filename,
        "model": model,
        "predicted_class": pred_label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "confidence_scores": confidence_scores
    }
    recent_uploads.insert(0, upload_entry)
    recent_uploads = recent_uploads[:5]
    save_recent_uploads(recent_uploads)

    return templates.TemplateResponse(
        "intelli_cv_classifier.html",
        {"request": request, "prediction": prediction, "recent_uploads": recent_uploads}
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
