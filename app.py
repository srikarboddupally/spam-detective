import streamlit as st
import numpy as np
import joblib
import re
import pandas as pd

# --- Load Pre-trained Model and Processors ---
@st.cache_resource
def load_model_components():
    model = joblib.load("spam_xgb_model.joblib")
    scaler = joblib.load("scaler.joblib")
    selector = joblib.load("selector.joblib")
    return model, scaler, selector

try:
    model, scaler, selector = load_model_components()
except Exception as e:
    st.error(f"Failed to load model components: {e}")
    st.stop()


# --- Feature Extraction Function ---
def extract_features_from_text(text):
    """
    Analyzes raw email text and computes the 57 features required by the Spambase model.
    """
    # Spambase feature names (order is crucial)
    word_features = [
        'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet',
        'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses',
        'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000',
        'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet',
        '857', 'data', '415', '85', 'technology', '1999', 'parts', 'pm',
        'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu',
        'table', 'conference'
    ]
    char_features = [';', '(', '[', '!', '$', '#']

    # Normalize text to lowercase for word frequency counting
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    num_words = len(words)

    # 1. Calculate word frequencies (first 48 features)
    word_freq = []
    if num_words > 0:
        for feature in word_features:
            count = words.count(feature)
            word_freq.append(100 * count / num_words)
    else:
        word_freq = [0.0] * 48

    # 2. Calculate character frequencies (next 6 features)
    num_chars = len(text)
    char_freq = []
    if num_chars > 0:
        for char in char_features:
            count = text.count(char)
            char_freq.append(100 * count / num_chars)
    else:
        char_freq = [0.0] * 6

    # 3. Calculate capital run-length features (last 3 features)
    capital_runs = re.findall(r'[A-Z]+', text)
    if capital_runs:
        run_lengths = [len(run) for run in capital_runs]
        capital_run_length_average = np.mean(run_lengths)
        capital_run_length_longest = np.max(run_lengths)
        capital_run_length_total = np.sum(run_lengths)
    else:
        capital_run_length_average = 0
        capital_run_length_longest = 0
        capital_run_length_total = 0

    # Combine all features into a single list
    all_features = word_freq + char_freq + [
        capital_run_length_average,
        capital_run_length_longest,
        capital_run_length_total
    ]
    
    return np.array(all_features).reshape(1, -1)

# --- Streamlit UI ---
st.set_page_config(page_title="Spam Detective", layout="wide")

st.title("🕵️ Spam Detective: AI-Powered Email Classifier")
st.markdown("Is that email suspicious? Paste its content below to find out if it's **Spam** or **Not Spam**.")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Email Content Here")
    email_text = st.text_area("Paste the full body of the email:", height=300, 
                              placeholder="Example: Congratulations! You've won a $1,000,000 prize. Click here to claim your money now! Free offer...")

if st.button("🔍 Analyze Email"):
    if email_text.strip() == "":
        st.warning("Please paste some email content to analyze.")
    else:
        # 1. Extract features from the input text
        with st.spinner('Analyzing content and extracting features...'):
            feature_vector = extract_features_from_text(email_text)
        
        # 2. Preprocess features using saved scaler and selector
        with st.spinner('Applying scaling and feature selection...'):
            features_scaled = scaler.transform(feature_vector)
            features_selected = selector.transform(features_scaled)
        
        # 3. Make a prediction
        with st.spinner('Running prediction model...'):
            prediction = model.predict(features_selected)[0]
            prediction_proba = model.predict_proba(features_selected)[0]

        # 4. Display the result
        with col2:
            st.subheader("Analysis Result")
            if prediction == 1:
                st.error(f"**Result: SPAM**")
                st.metric(label="Confidence", value=f"{prediction_proba[1]*100:.2f}%")
                st.markdown("Our model has flagged this email as potential spam. Be cautious with any links or attachments.")
            else:
                st.success(f"**Result: NOT SPAM (HAM)**")
                st.metric(label="Confidence", value=f"{prediction_proba[0]*100:.2f}%")
                st.markdown("This email appears to be legitimate.")
        
    # Optional: Show a few extracted feature values for insight
    with st.expander("Peek Behind the Scenes: Extracted Feature Insights"):
        # Use original 57 feature names here!
        word_features = [
            'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet',
            'order', 'mail', 'receive', 'will', 'people', 'report', 'addresses',
            'free', 'business', 'email', 'you', 'credit', 'your', 'font', '000',
            'money', 'hp', 'hpl', 'george', '650', 'lab', 'labs', 'telnet',
            '857', 'data', '415', '85', 'technology', '1999', 'parts', 'pm',
            'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu',
            'table', 'conference'
        ]
        char_features_names = [
            'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#'
        ]
        capital_features = [
            'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'
        ]
        feature_names = word_features + char_features_names + capital_features

        df_features = pd.DataFrame(feature_vector, columns=feature_names)

        st.markdown("**Capitalization Patterns:**")
        st.write(f"- Average Capital Sequence Length: `{df_features['capital_run_length_average'].values[0]:.2f}`")
        st.write(f"- Longest Capital Sequence: `{df_features['capital_run_length_longest'].values[0]}`")
        
        st.markdown("\n**Key Word Frequencies (% of total words):**")
        st.write(f"- 'free': `{df_features['free'].values[0]:.2f}%`")
        st.write(f"- 'money': `{df_features['money'].values[0]:.2f}%`")
        st.write(f"- 'remove': `{df_features['remove'].values[0]:.2f}%`")
        
        st.markdown("\n**Special Character Frequencies (% of total characters):**")
        st.write(f"- '!': `{df_features['char_freq_!'].values[0]:.2f}%`")
        st.write(f"- '$': `{df_features['char_freq_$'].values[0]:.2f}%`")

st.markdown("---")
st.info("This app uses a machine learning model trained on the Spambase Dataset to classify emails based on word/character frequencies and capitalization patterns.")
