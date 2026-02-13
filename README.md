# 🏟️📊 Sports vs Politics Text Classification  
**Problem 4 – NLU Assignment 1**

---

## 1. Problem Statement

The objective of this project is to design a text classification system that classifies a given document into one of the following categories:

- **SPORTS**
- **POLITICS**

The system uses traditional machine learning techniques with different feature representations and compares their performance quantitatively.

---

## 2. Dataset Collection

### 2.1 Data Sources

The dataset consists of news articles collected from publicly available sources.

**Sports Articles Sources:**
- ESPN  
- Cricbuzz  
- BBC Sports  
- Sports sections of newspapers  

**Politics Articles Sources:**
- BBC Politics  
- The Hindu (Politics)  
- Indian Express (Politics)  
- Government policy news portals  

Each document belongs strictly to **one class only**.

---

### 2.2 Dataset Structure

```
data/
├── sports/
│   ├── sport_1.pdf
│   ├── sport_2.txt
│   └── ...
└── politics/
    ├── politics_1.pdf
    ├── politics_2.txt
    └── ...
```

Each file contains **one full article** or **one document**.

---

### 2.3 Dataset Statistics (To be filled)

| Class    | Number of Documents |
|----------|---------------------|
| Sports   | 9 |
| Politics | 7 |
| Total    | 16 |

---

## 3. Text Preprocessing

Before feature extraction, the text is cleaned using the following steps:

1. Convert text to lowercase  
2. Remove punctuation and special characters  
3. Tokenize text into words  
4. Remove stopwords  
5. Optional: remove very short tokens (length < 2)  

**Note:**  
Stemming and lemmatization are intentionally avoided to preserve interpretability.

---

## 4. Feature Engineering

### 4.1 Bag of Words (BoW)

- Represents documents as word frequency vectors  
- Simple and interpretable  
- High-dimensional sparse vectors  

**Advantages:**
- Easy to implement  
- Works well with Naive Bayes  

**Limitations:**
- Ignores word importance  
- Treats all words equally  

---

### 4.2 TF-IDF (Term Frequency – Inverse Document Frequency)

- Penalizes frequent but uninformative words  
- Highlights discriminative terms  

TF-IDF(w,d) = TF(w,d) × log(N / DF(w))

**Advantages:**
- Improves classification accuracy  
- Reduces effect of stopwords  

---

### 4.3 N-grams (Optional Enhancement)

- Uses unigrams and bigrams  
- Captures contextual phrases like:  
  - "prime minister"  
  - "world cup"  

---

## 5. Machine Learning Models

### 5.1 Naive Bayes
- Probabilistic model  
- Assumes feature independence  
- Fast and memory efficient  

### 5.2 Logistic Regression
- Linear classifier  
- Outputs probabilities  
- Works well with TF-IDF features  

### 5.3 Support Vector Machine (SVM)
- Maximizes margin between classes  
- Performs well in high-dimensional spaces  
- Typically achieves best accuracy for text tasks  

---

## 6. Experimental Setup

- Dataset split: **80% training / 20% testing**  
- Same train-test split used for all models  
- Hyperparameters kept minimal for fair comparison  

---

## 7. Evaluation Metrics

The models are evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  

---

## 8. Results and Comparison

| Model | Feature Type | Accuracy | Precision | Recall | F1 |
|------|-------------|----------|-----------|--------|----|
| Naive Bayes | BoW | TBD | TBD | TBD | TBD |
| Logistic Regression | TF-IDF | TBD | TBD | TBD | TBD |
| SVM | TF-IDF + Bigram | TBD | TBD | TBD | TBD |

---

## 9. Limitations

- Overlap between sports and political vocabulary  
- Dataset bias depending on source  
- Models do not capture deep semantics  
- Performance depends heavily on feature engineering  

---

## 10. Conclusion and Future Work

- TF-IDF with SVM performs best overall  
- Feature representation plays a crucial role  
- Future improvements may include:
  - Word embeddings  
  - Deep learning models  
  - Larger and more diverse datasets  

---

## 11. Repository Structure

```
sports-politics-classifier/
│
├── data/
│   ├── sports/
│   └── politics/
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   └── metrics.csv
│
├── report.pdf
└── README.md
```

---
## 12. Author

- **Name:** <Your Name>  
- **Roll Number:** <Your Roll Number>  
- **Course:** Natural Language Understanding  

