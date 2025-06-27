import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import os
import shutil
import tempfile
from PyPDF2 import PdfReader

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Extract text from PDF
    try:
        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        os.remove(tmp_path)
        return JSONResponse({"error": f"Failed to extract text: {str(e)}"})
    finally:
        os.remove(tmp_path)

    if not text.strip():
        return JSONResponse({"error": "No text found in PDF."})

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
        return JSONResponse({"error": "Invalid model selected."})

    # Predict
    pred = clf.predict(X)[0]
    try:
        label = class_names[pred]
    except Exception:
        label = str(pred)

    result = {
        "predicted_class": label,
        "extracted_text": clean_text[:3000]  # Limit for display
    }

    # For random forest, show confidence scores
    if model == "random_forest":
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            scores = {class_names[i]: float(proba[i]) for i in range(len(proba))}
            result["confidence_scores"] = scores

    return JSONResponse(result)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
