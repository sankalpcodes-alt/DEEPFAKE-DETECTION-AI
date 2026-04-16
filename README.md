Deepfake Detection Using Multimodal Deep Learning

Models
1. Baseline CNN
2. Proposed Model (CNN + FFT)

Frontend
- Streamlit web app for image upload and model comparison
- Side-by-side prediction cards for both models
- Saved evaluation chart preview

Run The Frontend
```bash
pip install -r requirements.txt
streamlit run app.py
```

Dataset Structure

dataset/
   train/
      real/
      fake/
   test/
      real/
      fake/

Evaluation Metrics
Accuracy
F1 Score

Tools
Python
PyTorch
OpenCV
NumPy
Scikit-learn
Matplotlib
Streamlit
