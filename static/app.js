document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const form = e.target;
    const fileInput = document.getElementById('pdf-upload');
    const modelSelect = document.getElementById('model-select');
    const resultSection = document.getElementById('result-section');
    const predictedClass = document.getElementById('predicted-class');
    const confidenceScores = document.getElementById('confidence-scores');
    const extractedText = document.getElementById('extracted-text');

    if (!fileInput.files.length) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', modelSelect.value);

    predictedClass.textContent = '';
    confidenceScores.textContent = '';
    extractedText.textContent = '';
    resultSection.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            predictedClass.textContent = "Error: " + data.error;
            resultSection.style.display = 'block';
            return;
        }

        predictedClass.textContent = "Predicted Class: " + data.predicted_class;
        extractedText.textContent = data.extracted_text || '';

        if (modelSelect.value === 'random_forest' && data.confidence_scores) {
            let html = '<strong>Confidence Scores:</strong><ul>';
            for (const [label, score] of Object.entries(data.confidence_scores)) {
                html += `<li>${label}: ${(score*100).toFixed(2)}%</li>`;
            }
            html += '</ul>';
            confidenceScores.innerHTML = html;
        } else {
            confidenceScores.innerHTML = '';
        }

        resultSection.style.display = 'block';
    } catch (err) {
        predictedClass.textContent = "Error: " + err.message;
        resultSection.style.display = 'block';
    }
});
