# 📝 JobMatcher PRO

Welcome to **JobMatcher Pro**, an end-to-end intelligent resume classification platform leveraging NLP and machine learning to automate and enhance the recruitment process.

---

## 🚀 Project Overview

JobMatcher Pro is designed to **automatically classify resumes** into predefined job categories, streamlining the screening process for recruiters and HR professionals. The system extracts text from uploaded resumes (PDF/DOCX), preprocesses it, and uses advanced ML models to predict the most suitable job category. The platform provides a modern web interface, supports multiple classification models, and displays confidence scores for predictions.

---

## 📺 [Video Demo](https://drive.google.com/file/d/1gv13ph3lGNSg6I8k3GayTkGJSaUdKUiS/view?usp=sharing)

---

## 🏗️ Project Architecture

**Architecture Highlights:**
- **Frontend:** HTML, CSS, JS (served via FastAPI/Jinja2)
- **Backend:** FastAPI (Python)
- **ML Models:** Trained with scikit-learn, loaded via joblib
- **PDF Parsing:** PyPDF2
- **Data Storage:** Recent uploads tracked in JSON
- **Deployment:** Docker-ready

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, CSS3, JavaScript (Vanilla), Jinja2 Templates
- **Backend:** Python 3, FastAPI, Uvicorn
- **NLP & ML:** scikit-learn, pandas, numpy, matplotlib, seaborn, wordcloud, nltk
- **PDF Extraction:** PyPDF2
- **Model Serialization:** joblib
- **Containerization:** Docker
- **Version Control:** Git

---

## 🤖 Machine Learning Models

The following models are trained and available for selection:
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes (MultinomialNB)**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Random Forest** (with per-class confidence scores)

**Feature Extraction:**  
All models use TF-IDF vectorization for transforming resume text into numerical features.

---

## 🔄 Workflow

1. **Data Preparation & Preprocessing**
   - Clean and normalize resume text (remove URLs, special chars, stopwords, etc.)
   - Visualize data distribution and word clouds for insights

2. **Feature Engineering**
   - Apply TF-IDF vectorization to convert text into feature vectors

3. **Model Training & Evaluation**
   - Train multiple classifiers (KNN, Naive Bayes, SVM, Logistic Regression, Random Forest)
   - Evaluate using accuracy, precision, recall, F1-score, confusion matrix
   - Save trained models and vectorizer with joblib

4. **API & Web Interface**
   - FastAPI backend serves a modern web UI for uploading resumes and selecting models
   - Extracts text from PDF/DOCX, preprocesses, predicts category, and displays results
   - For Random Forest, shows confidence scores for each class

5. **Recent Uploads**
   - Tracks last 5 uploads with filename, model, prediction, and timestamp

6. **Deployment**
   - Dockerfile provided for easy containerization and deployment

---

## 🌐 Web Interface Features

- **Upload Resume:** Supports PDF (and optionally DOCX)
- **Model Selection:** Choose from 5 ML models
- **Prediction Display:** Shows predicted class and, for Random Forest, confidence scores per class
- **Recent Uploads:** Sidebar with last 5 predictions
- **Responsive Design:** Clean, modern, and mobile-friendly

---

## 🐳 Docker Support

Easily containerize and deploy the app:

```bash
docker build -t intellicv-app .
docker run -p 8000:8000 intellicv-app
```

---

## 📝 How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Resume_Classification-app.git
   cd Resume_Classification-app
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   uvicorn app:app --reload
   ```
   Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

4. **Upload a resume, select a model, and view the prediction!**

---

## 🧠 Model Training

- See `Resume_Analyzer.ipynb` for full data exploration, preprocessing, model training, evaluation, and export.
- Models and vectorizer are saved in the `model/` directory for use by the FastAPI app.

---

## 🗂️ Repository Structure

```
Resume_Classification-app/
│
├── app.py                  # FastAPI backend
├── requirements.txt
├── Dockerfile
├── model/
│   ├── knn_model.joblib
│   ├── naive_bayes_model.joblib
│   ├── svm_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   ├── tfidf_vectorizer.joblib
│   └── ProjectDataset.csv
│
├── templates/
│   ├── intelli_cv_classifier.html
│   └── index.html
│
├── static/
│   ├── style.css
│   ├── app.js
│   └── resume_banner.png
│
├── Resume_Analyzer_Project/
│   └── model/
│       └── Resume_Analyzer.ipynb
│
└── recent_uploads.json
```

---

## 🛡️ Git & Version Control

- All code, models, and assets are tracked via Git.
- Use feature branches for development and submit pull requests for review.
- Commit messages should be clear and descriptive.

---

## 📦 Requirements

See `requirements.txt` for all dependencies.

---


## 📚 References

- [scikit-learn documentation](https://scikit-learn.org/)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [PyPDF2 documentation](https://pypdf2.readthedocs.io/)
- [Jinja2 documentation](https://jinja.palletsprojects.com/)

---

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📧 Contact

For questions or suggestions, please contact [shivam2048sinha@gmail.com].

---

**Empowering smarter recruitment with AI!** 🚀

