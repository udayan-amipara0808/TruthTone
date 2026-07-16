import streamlit as st
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import time
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title = "Sarcasm & Sentiment Analysis",
    layout = "wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        background: linear-gradient(135deg, #1E3A8A, #3b5fd4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 30px;
        opacity: 0.7;
    }
    .metric-card {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.2);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .sarcasm-text { color: #DC2626; font-weight: bold; }
    .sentiment-pos { color: #16A34A; font-weight: bold; }
    .sentiment-neg { color: #EA580C; font-weight: bold; }
    </style>
    """
,unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading BERT models — takes ~30 seconds on first run...")
def get_weights_path(filename):
    return hf_hub_download(
        repo_id="udayan0808/truthtone-weights",
        filename=filename
    )
def load_models_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def build_multitask_bert(max_len=128):
        input_ids      = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')

        bert_base     = TFBertModel.from_pretrained('bert-base-uncased')
        bert_output   = bert_base({"input_ids": input_ids, "attention_mask": attention_mask})
        pooled_output = bert_output[1]

        shared_dropout = tf.keras.layers.Dropout(0.3)(pooled_output)

        sarcasm_dense = tf.keras.layers.Dense(64, activation="relu")(shared_dropout)
        sarcasm_out   = tf.keras.layers.Dense(1, activation="sigmoid", name="sarcasm_output")(sarcasm_dense)

        sentiment_dense = tf.keras.layers.Dense(64, activation="relu")(shared_dropout)
        sentiment_out   = tf.keras.layers.Dense(1, activation="sigmoid", name="sentiment_output")(sentiment_dense)

        return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=[sarcasm_out, sentiment_out])

    def load_model_weights(model, npz_path):
        dummy_ids  = tf.zeros((1, 128), dtype=tf.int32)
        dummy_mask = tf.zeros((1, 128), dtype=tf.int32)
        model({"input_ids": dummy_ids, "attention_mask": dummy_mask})

        loaded = np.load(npz_path)
        weights_list = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]

        expected = len(model.get_weights())
        if len(weights_list) != expected:
            raise ValueError(
                f"Weight count mismatch for {npz_path}: "
                f"file has {len(weights_list)} arrays, model expects {expected}. "
                "This means the architecture built here does not match the one "
                "used when the weights were originally saved."
            )

        model.set_weights(weights_list)
        return model

    news_model   = build_multitask_bert()
    reddit_model = build_multitask_bert()

    models_loaded = False
    try:
        news_model_path = get_weights_path("bert_multitask_news_weights.npz")
        reddit_model_path = get_weights_path("bert_multitask_reddit_weights.npz")

        news_model = load_model_weights(news_model, news_model_path)
        reddit_model = load_model_weights(reddit_model, reddit_model_path)
        models_loaded = True
    except FileNotFoundError as e:
        st.error(f"Weight file not found: {e}")
    except Exception as e:
        st.error(f"Weight loading error: {e}")

    return tokenizer, news_model, reddit_model, models_loaded

tokenizer , news_model , reddit_model, is_loaded = load_models_and_tokenizer()

def predict_text(text,model,tokenizer):

    if not is_loaded:
        time.sleep(1)
        return np.random.rand(), np.random.rand()

    encoded  = tokenizer([text],padding="max_length",truncation=True,max_length=128,return_tensors="tf")
    pred = model.predict({"input_ids":encoded["input_ids"], "attention_mask":encoded["attention_mask"]}, verbose=0)
    return pred[0][0][0],pred[1][0][0]

with st.sidebar:
    st.image("https://placehold.co/75x75/1E3A8A/FFF?text=UA", width='content')

    st.markdown("### About Me")
    st.markdown("**Name:** Udayan Amipara")
    st.markdown("ML & Deep Learning Enthusiast , Looking for great opportunity and work in the domain of Software Engineering & AIML")

    st.markdown("### 🔗 Links")
    st.markdown(
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/udayan-amipara-637b21324/)")
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/udayan-amipara0808)")
    st.markdown(
        "[![Repo](https://img.shields.io/badge/Project_Repo-gray?style=for-the-badge&logo=git)](https://github.com/udayan-amipara0808/TruthTone)")

    st.divider()

    st.markdown("### Project Description")
    st.info(
        "This application demonstrates **Domain Adaptation** in NLP. "
        "It features a Multi-Task BERT model trained simultaneously on Sarcasm and Sentiment. "
        "Because sarcasm looks different in a news headline versus a social media post, "
        "two distinct models were trained to capture domain-specific nuances."
    )

    st.divider()
    st.caption(f"Model status: {' Weights loaded' if is_loaded else 'Weights NOT loaded — showing random demo values'}")

st.markdown('<h1 class="main-title">TruthTone - Multi-Task Sarcasm & Sentiment AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Detecting the true meaning behind the text using BERT</p>', unsafe_allow_html=True)

st.markdown("### 1. Select the Domain Context")
domain = st.radio(
    "Choose the type of text you are analyzing to load the appropriate model weights:",
    options=["News Headline (Formal)", " Social Media / Reddit (Informal)"],
    horizontal=True
)

st.markdown("### 2. Enter Text")
user_input = st.text_area("Type or paste a sentence here...", height=100,
                          placeholder="e.g., Oh great, another flight delay. Just what I wanted today!")

if st.button("Analyze Text", type="primary", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("BERT is reading between the lines..."):
            # Select the right brain based on user toggle
            active_model = news_model if "News" in domain else reddit_model

            sarc_score, sent_score = predict_text(user_input, active_model, tokenizer)

            # Interpret the scores
            is_sarcastic = sarc_score > 0.5
            literal_sentiment = "Positive" if sent_score > 0.5 else "Negative"

            # The Sarcasm Adjustment Logic
            if is_sarcastic:
                true_meaning = "Negative" if literal_sentiment == "Positive" else "Positive"
                interpretation = f"The literal words are **{literal_sentiment}**, but the model detects heavy sarcasm. The *true* intent is likely **{true_meaning}**."
            else:
                interpretation = f"The model detects no sarcasm. The literal meaning holds: **{literal_sentiment}**."

        st.markdown("### Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Sarcasm Confidence", f"{sarc_score * 100:.1f}%",
                      delta="Sarcastic" if is_sarcastic else "Literal",
                      delta_color="inverse" if is_sarcastic else "normal")
            st.progress(float(sarc_score))
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Literal Sentiment Score", f"{sent_score * 100:.1f}%",
                      delta="Positive" if sent_score > 0.5 else "Negative")
            st.progress(float(sent_score))
            st.markdown('</div>', unsafe_allow_html=True)

        st.success(f"**Pipeline Conclusion:** {interpretation}")

st.divider()
st.markdown("### Model Performance Metrics")
st.markdown("Compare how the two models perform on their respective domains:")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)

# Replace these placeholder numbers with your actual training accuracies!
with m_col1:
    st.metric(label="News Sarcasm Accuracy", value="93.9%")
with m_col2:
    st.metric(label="News Sentiment Accuracy", value="83.5%")
with m_col3:
    st.metric(label="Social Sarcasm Accuracy", value="81.2%")
with m_col4:
    st.metric(label="Social Sentiment Accuracy", value="86.7%")