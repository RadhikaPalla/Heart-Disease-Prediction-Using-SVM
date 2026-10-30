# Heart Disease Prediction System - Streamlit App

## Files
- `app.py` - dashboard and patient prediction interface
- `requirements.txt` - required Python packages
- `ML Dataset.csv` - copy your existing dataset into this folder

## Run in VS Code (Windows)
1. Extract/open this folder in VS Code.
2. Copy your `ML Dataset.csv` into this same folder.
3. Open Terminal in VS Code.
4. Run:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Run:
   ```bash
   python -m streamlit run app.py
   ```
6. The browser opens. If it does not, copy the Local URL shown in the terminal into your browser.

The CSV must contain a `target` column with values 0 and 1, and these 11 features in the same order/names as your training dataset:
`age`, `sex`, `chest pain type`, `resting bp s`, `cholesterol`, `fasting blood sugar`, `resting ecg`, `max heart rate`, `exercise angina`, `oldpeak`, `ST slope`.

The app trains SVM, Logistic Regression and KNN using the same 80/20 stratified split (`random_state=42`) and scaling approach, then shows model comparisons, SVM confusion matrix, ROC curve/AUC, prediction distribution and a patient prediction form.

Educational demonstration only; not a medical diagnosis.
