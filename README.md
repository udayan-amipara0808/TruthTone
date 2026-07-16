# TruthTone - https://truthtone.streamlit.app/
 
**Detecting the true meaning behind the text — sarcasm and sentiment analysis powered by domain-adapted BERT.**

TruthTone is a multi-task NLP application that analyzes text for both **sarcasm** and **sentiment** simultaneously, then combines the two to infer the speaker's *actual* intended meaning. For example, a sentence like *"Oh great, another flight delay"* reads as literally positive but is flagged as sarcastic — so TruthTone reports the true underlying sentiment as negative.

## Why two models?
 
Sarcasm doesn't look the same everywhere. A news headline signals sarcasm very differently than a Reddit comment or tweet. To capture this, TruthTone uses **two separate fine-tuned BERT models**:
 
- **News model** — trained on formal news headlines
- **Social/Reddit model** — trained on informal social media text
You pick the domain in the app, and the corresponding model is used for inference.

## How it works
 
1. Input text is tokenized using `bert-base-uncased`.
2. A shared BERT encoder produces a pooled representation of the text.
3. Two separate task-specific heads branch off the shared representation:
   - **Sarcasm head** — binary sigmoid output (sarcastic / literal)
   - **Sentiment head** — binary sigmoid output (positive / negative)
4. The app combines both scores: if sarcasm is detected, the literal sentiment is flipped to infer the true intended meaning.
## Tech stack
 
- **Streamlit** — web UI
- **TensorFlow / Keras** — model architecture and inference
- **Hugging Face Transformers** — BERT tokenizer and base model
- **NumPy** — array handling

 ## Running locally
 
```bash
git clone https://github.com/yourusername/truthtone.git
cd truthtone
pip install -r requirements.txt
streamlit run app.py
```
## Model performance
 
| Domain | Sarcasm Accuracy | Sentiment Accuracy |
|---|---|---|
| News | 93.9% | 83.4% |
| Social / Reddit | 81.2% | 86.7% |
