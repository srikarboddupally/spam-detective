# 🕵️ Spam Detective

A Streamlit web app that uses machine learning to classify email text as **Spam** or **Not Spam (Ham)**. Built using the Spambase dataset and an XGBoost model.

🔗 **Live App**: [https://spam-detective-project.streamlit.app](https://spam-detective-project.streamlit.app)

---

## 🚀 Features

- Paste any email content into the app
- Extracts 57 Spambase features (word/character frequencies, capitalization patterns)
- Scales and selects features using pre-trained `StandardScaler` and `SelectKBest`
- Classifies emails using a trained **XGBoost** model
- Displays prediction + model confidence

---

## 🧠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- XGBoost
- Joblib
- Pandas, NumPy, RegEx

---

## 📦 Installation

To run locally:

```bash
git clone https://github.com/srikarboddupally/spam-detective.git
cd spam-detective
pip install -r requirements.txt
streamlit run app.py
