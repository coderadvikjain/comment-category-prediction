# Comment Category Prediction using NLP & ML

A multi-class text classification project that predicts the **category of online comments** using TF-IDF feature extraction, 80+ engineered features, and classical ML models. Built for the Comment Category Prediction Challenge on Kaggle.

---

## Overview

This project tackles a multi-class classification problem on user-generated comments. It combines NLP-based text vectorization (word + character level TF-IDF) with extensive hand-crafted features, then compares multiple models before selecting **LinearSVC** as the best performer based on Macro F1 score.

---

## Dataset

- **Source:** [Comment Category Prediction Challenge](https://www.kaggle.com/competitions/comment-category-prediction-challenge)
- **Target:** `label` (multi-class — 4 classes)
- **Text Feature:** `comment`
- **Metadata Features:** `post_id`, `upvote`, `downvote`, `if_1`, `if_2`, `emoticon_1/2/3`, `race`, `religion`, `gender`, `disability`, `created_date`, and more
- **Split:** 80/20 train-validation with stratification (`random_state=42`)

---

## Pipeline

### 1. Data Cleaning
- Missing values in `race`, `religion`, `gender` handled with **most frequent** imputation
- Empty/whitespace comments replaced with empty string
- Duplicates removed

### 2. EDA & Chart Analysis
- **Label Distribution** — heavy class imbalance (class 0 dominates, class 3 rarest)
- **Word Count Distribution** — right-skewed, most comments are short
- **Word Count by Label** — label 0 = longer discussions, label 3 = shorter messages
- **Activity by Hour** — peak in evening, low during work/college hours
- **Correlation Heatmap** — `if_2` is the strongest predictor of labels

### 3. Feature Engineering (80+ features)

| Category | Features |
|----------|----------|
| **Time Features** | `hour`, `weekday`, `is_weekend`, `is_night`, `is_morning`, `is_afternoon` |
| **Text Statistics** | `comment_len`, `caps_ratio`, `exclamation`, `question`, `punct_count`, `digit_count`, `space_count` |
| **Social Signals** | `url_count`, `mention_count`, `hashtag_count`, `has_emoticon` |
| **Sentiment Keywords** | `has_negative_words`, `has_positive_words` (keyword-based detection) |

### 4. Text Vectorization
- **Word-level TF-IDF:** unigrams to trigrams, max features with sublinear TF
- **Character-level TF-IDF:** 2-6 char n-grams (captures slang, misspellings, partial words)

### 5. Preprocessing
- Numerical features scaled with **MinMaxScaler**
- Categorical features (`race`, `religion`, `gender`) encoded with **OneHotEncoder**
- Combined via `ColumnTransformer`

### 6. Feature Selection
- **SelectKBest** with chi-squared test, top 155,000 features retained

### 7. Model Training & Comparison

| Model | Macro F1 |
|-------|----------|
| **Linear SVC** | **0.830** |
| Logistic Regression | 0.823 |
| SGD Classifier | 0.822 |
| MLP Classifier | 0.780 |

### 8. Dimensionality Reduction
- **TruncatedSVD** (500 components) used for MLP Classifier to reduce compute

### 9. Final Model
- **LinearSVC** (C=0.2, `class_weight='balanced'`) retrained on full data (train + validation merged)

---

## Results

| Approach | Macro F1 |
|----------|----------|
| Best on leaderboard | **0.87** |
| **This project (LinearSVC)** | **0.836** |

> Achieved **0.836 Macro F1** using classical ML with TF-IDF and hand-crafted features — no transformers, no pretrained language models. Only 3.4% behind the top score of 0.87.

---

## Project Structure

```
├── notebook.ipynb          # Main Kaggle notebook
└── README.md               # This file
```

---

## How to Run

1. **On Kaggle / Google Colab (recommended):**
   - Open the [notebook on Kaggle](https://www.kaggle.com/) and click **"Copy & Edit"**
   - Or upload `notebook.ipynb` to [Google Colab](https://colab.research.google.com/)
   - Run all cells

2. **Locally:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn scipy
   jupyter notebook notebook.ipynb
   ```

---

## Tech Stack

- Python 3
- Scikit-learn (TF-IDF, LinearSVC, SelectKBest, TruncatedSVD, pipelines)
- Pandas / NumPy
- Matplotlib / Seaborn
- SciPy

---

## Key Features

- Dual TF-IDF extraction: word-level (1-3 grams) + character-level (2-6 grams)
- 80+ hand-crafted features from text, metadata, time, and sentiment signals
- Chi-squared feature selection (155K features retained)
- 4 model comparison benchmarked on Macro F1
- TruncatedSVD for dimensionality reduction with MLP
- Class weighting (`balanced`) to handle label imbalance
- Full train+val merge for maximum training data before final submission

---

## Author

**Jatin Nahata Jain** — [Kaggle Profile](https://www.kaggle.com/jatinnahatajain)

---

## License

This project is open source and available under the [MIT License](LICENSE).