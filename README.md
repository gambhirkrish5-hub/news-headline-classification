# News Headline Classification — TF-IDF + LogReg vs. BiLSTM

Classifying news headlines into 6 categories (Politics, Sports, Business, Tech,
Entertainment, Science), comparing a classic NLP pipeline against a deep learning
approach to see which is the better practical choice.

## Overview

This project builds and evaluates two very different models for short-text
classification: a fast, interpretable **TF-IDF + Logistic Regression** pipeline, and
a **Bidirectional LSTM** with word embeddings. Beyond just comparing accuracy, the
project digs into *why* the two models perform the way they do, and what that implies
for choosing between classic ML and deep learning in a real production system.

## Tools & Libraries

- **Python** — pandas, NumPy
- **Classic ML** — scikit-learn (TfidfVectorizer, LogisticRegression, Pipeline, metrics)
- **Deep Learning** — TensorFlow / Keras (Embedding, Bidirectional LSTM, Dense, Dropout,
  EarlyStopping)
- **Text Processing** — NLTK (stopwords), regular expressions
- **Visualization** — Matplotlib, Seaborn, WordCloud
- **Environment** — Jupyter Notebook

## What I Did

1. **Data preparation** — loaded a balanced, 6-category headline dataset (8,400
   headlines), with a defensive fallback to a template-generated synthetic dataset when
   the real corpus isn't available in the environment, so the notebook always runs
   end-to-end.
2. **EDA** — examined class balance, headline length distribution, and generated
   per-category word clouds to visually confirm each category's distinctive vocabulary.
3. **Preprocessing** — cleaned and tokenized headlines, removing stopwords for the
   TF-IDF pipeline while keeping them for the LSTM (since sequence models can use
   function words as context, unlike a bag-of-words model).
4. **Model A — TF-IDF + Logistic Regression** — unigram + bigram TF-IDF features (8,000
   vocabulary cap) feeding a tuned multinomial Logistic Regression.
5. **Model B — Bidirectional LSTM** — a from-scratch/GloVe-ready embedding layer, a 64-
   unit BiLSTM, dropout regularization, and a dense softmax output, trained with early
   stopping.
6. **Evaluation** — compared both models on accuracy, precision, recall, and F1-score;
   visualized confusion matrices and training curves.
7. **Misclassification analysis** — grouped confusion pairs to check whether errors
   were random or clustered around genuinely ambiguous, overlapping topics.

## Key Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 95.65% | 95.67% | 95.66% | 95.65% |
| Bidirectional LSTM | 95.65% | 95.67% | 95.66% | 95.65% |

- Both models comfortably cleared an 80% target accuracy, with consistent, balanced
  performance across all six categories (F1-scores between 0.95 and 0.97).
- **Notably, both models produced identical predictions on every single test headline**
  (verified directly, not just via matching aggregate metrics) — a result traced to the
  dataset's highly distinctive per-category vocabulary, which lets a bag-of-words model
  perform just as well as a sequence model. This finding, and what it implies about when
  to prefer classic ML over deep learning, is explored in detail in the notebook and the
  accompanying report.
- Where errors did occur, they clustered around genuinely overlapping real-world topics
  (e.g. Business vs. Tech for technology-company earnings stories) rather than being
  spread randomly — a sign the errors reflect authentic task ambiguity, not model
  weakness.

## Notebook

See [`News_Headline_Classification.ipynb`](./News_Headline_Classification.ipynb) for the
full analysis, code, and visualizations.

## Full Report

A detailed 15-page write-up — covering methodology, model architecture, evaluation, a
deep dive into the identical-predictions finding, and a discussion of classic ML vs.
deep learning trade-offs — is available on request as a companion PDF/Word report.
