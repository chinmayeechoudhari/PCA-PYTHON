# PCA Studio — Python Edition
## A full Model Training project built from scratch

---

## STEP-BY-STEP: How to run this in VS Code

### Step 1 — Make sure Python is installed
Open a terminal (in VS Code: Terminal → New Terminal) and type:
```
python --version
```
You should see Python 3.8 or higher. If not, download Python from https://python.org

---

### Step 2 — Open the project folder in VS Code
Go to: File → Open Folder → select the `pca-python` folder

---

### Step 3 — Create a virtual environment (recommended)
In the VS Code terminal:
```
python -m venv venv
```
Then activate it:

On Windows:
```
venv\Scripts\activate
```
On Mac/Linux:
```
source venv/bin/activate
```
You should see `(venv)` appear at the start of your terminal line.

---

### Step 4 — Install dependencies
```
pip install -r requirements.txt
```
This installs Flask (the web server) and Pillow (image processing).

---

### Step 5 — Run the app
```
python app.py
```
You will see output like:
```
 * Running on http://127.0.0.1:5000
```

---

### Step 6 — Open in browser
Go to: http://127.0.0.1:5000

The app opens with 8 tabs. Follow the workflow left to right.

---

## The 8-Tab Workflow

| Tab | What to do |
|-----|------------|
| 01 Data      | Upload 10–30 images. Set size (32×32 is fine). Choose grayscale or RGB. |
| 02 Split     | Set train/test split (80% train is standard). Click Confirm Split. |
| 03 Train     | Set k (try 10). Click Run PCA Training. |
| 04 Results   | See scree plot, variance charts, and eigenfaces. |
| 05 Evaluate  | Click Run Evaluation to see train vs test error and the k-curve. |
| 06 Reconstruct | Pick any image, drag the slider to see how k affects quality. |
| 07 Anomaly   | Upload a very different image and see if the model flags it. |
| 08 Projection| See all images plotted in 2D PC space. |

---

## What is "Model Training" in this project?

In machine learning, "model training" means teaching an algorithm using data so it can make decisions on new, unseen data. Here's exactly how that works in this PCA project:

### The Model
The PCA model is defined by:
- **Eigenvectors** (principal components) — directions of maximum variation in the training images
- **Eigenvalues** — how much variation each direction captures
- **Mean vector** — the average training image

These are the things that get "learned" from data.

### The Training Process (what happens when you click Train)
1. **Vectorise** — each training image is resized and flattened into a 1D array of pixel values (0 to 1)
2. **Mean-centre** — subtract the average image from every image vector (removes brightness bias)
3. **Covariance matrix** — build a matrix measuring how correlated every pair of pixels is across training images
4. **Power Iteration** — find the top-k eigenvectors of the covariance matrix (directions of most variation)
5. **Project** — dot-product every image against each eigenvector to get its "scores"

The model is stored as these eigenvectors + mean. No training images are needed after this.

### Train / Test Split
This is the most important concept. We split images into two groups:
- **Training set (80%)** — the model sees this data and learns from it
- **Test set (20%)** — the model never sees this during training

After training, we measure reconstruction error on both sets:
- Low train error + Low test error = Good generalisation
- Low train error + High test error = Overfitting (memorised training data)

### What is Overfitting?
If you trained on all images including the test set, the model would "memorise" them and score perfectly. But it would fail on any genuinely new image. Overfitting means the model learned the specific details of training data instead of the underlying patterns.

### Anomaly Detection
Once trained, the model can only reconstruct images that look like its training data. Give it an unusual image and the reconstruction will be poor — the error (MSE) will be high. We compare every image's error to a threshold:
- Error > threshold = ANOMALY
- Error ≤ threshold = Normal

This is used in real industry for: defect detection in manufacturing, fraud detection in banking, unusual activity detection in networks.

### Save / Load Model
The trained eigenvectors are saved to a .json file. Later you can load this file and use the model on new images without retraining. This simulates deploying a trained ML model.

---

## File Structure

```
pca-python/
├── app.py                  ← Flask server + all PCA maths (Python)
├── requirements.txt        ← Python package list
├── templates/
│   └── index.html          ← The 8-tab web interface
├── static/
│   ├── css/
│   │   └── style.css       ← Dark theme styling
│   └── js/
│       └── main.js         ← Frontend: API calls, charts, UI
├── uploads/                ← Temporary (auto-created)
└── models/                 ← Temporary (auto-created)
```

---

## No major libraries used
- All PCA maths is written from scratch in `app.py`
- Pillow is used only for image resizing and pixel reading
- Flask is only the web server (routing)
- Chart.js is only for drawing the charts

No NumPy, no scikit-learn, no pandas.
