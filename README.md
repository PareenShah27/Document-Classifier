# Sports vs Politics Document Classifier 
# (NLU Assignment – Problem 4)

## Course Information

* **Course**: CSL 7640 – Natural Language Understanding
* **Assignment**: Assignment 1 – Problem 4
* **Task**: Classify documents as **Sports** or **Politics** using machine learning

---

## 1. Problem Statement

The objective of this task is to design a document-level text classifier that automatically categorizes an input document as either **Sports** or **Politics**. The system must:

* Read a text document (.txt or .pdf)
* Extract and preprocess textual content
* Represent documents using suitable feature representations
* Train and compare **at least three machine learning models**
* Quantitatively evaluate and compare model performance

---

## 2. Dataset Description

### 2.1 Data Collection

The dataset was manually curated using publicly available reports and articles related to:

* **Sports**: match reports, sports policy documents, tournament summaries
* **Politics**: political analysis reports, governance documents, policy briefs

Documents were stored in the following directory structure:

```
data/
 ├── sports/
 │    ├── *.txt / *.pdf
 └── politics/
      ├── *.txt / *.pdf
```

### 2.2 Dataset Characteristics

* **Classes**: 2 (Sports, Politics)
* **Formats supported**: `.txt`, `.pdf`
* **Class balance**: Approximately balanced
* **Document length**: Medium to long-form reports

---

## 3. Preprocessing Pipeline

Each document undergoes the following preprocessing steps:

1. Text extraction from `.txt` or `.pdf` files
2. Lowercasing
3. Removal of numbers and special characters
4. Whitespace normalization

Only Python standard libraries and minimal utilities were used for preprocessing.

---

## 4. Feature Representation

The following feature representations were explored:

* **Bag of Words (BoW)**
* **TF-IDF (Unigrams)**
* **TF-IDF with Bigrams**

Vectorizers were trained **only on training data** to avoid data leakage.

---

## 5. Machine Learning Models

Three supervised machine learning models were implemented and compared:

1. **Naive Bayes (MultinomialNB)** with Bag of Words
2. **Logistic Regression** with TF-IDF
3. **Support Vector Machine (Linear Kernel)** with TF-IDF + Bigrams

All models were trained using scikit-learn.

---

## 6. Evaluation Strategy

### 6.1 Cross-Validation

Due to limited dataset size, **Stratified 5-Fold Cross-Validation** was used to obtain reliable performance estimates while preserving class distribution.

### 6.2 Evaluation Metrics

The following metrics were reported:

* Accuracy
* Precision (weighted)
* Recall (weighted)
* F1-score (weighted)

---

## 7. Results Summary

| Model                      | Accuracy  | Precision | Recall    | F1-score  |
| -------------------------- | --------- | --------- | --------- | --------- |
| Naive Bayes                | ~0.75     | ~0.79     | ~0.75     | ~0.72     |
| Logistic Regression        | ~0.82     | ~0.79     | ~0.82     | ~0.78     |
| **SVM (TF-IDF + Bigrams)** | **~0.87** | **~0.82** | **~0.87** | **~0.83** |

The SVM model achieved the best average performance across folds.

---

## 8. Final Model Selection

The model with the highest mean cross-validation accuracy was automatically selected and retrained on the **full dataset**. This final model was then used for interactive document classification.

---

## 9. How to Run

### 9.1 Install Dependencies

```bash
pip install scikit-learn pypdf pandas
```

### 9.2 Run the Classifier

```bash
python classifier.py
```

### 9.3 Interactive Mode

After training, the system enters an interactive mode:

```
Enter file path: <path_to_document>
```

---

## 10. Limitations

* Small dataset size limits generalization
* Overlap between sports governance and political documents
* No semantic understanding beyond lexical features

---

## 11. Conclusion

This project demonstrates a complete NLP pipeline for document classification using classical machine learning techniques. The results align with theoretical expectations, with SVM performing best on high-dimensional sparse text features.

---

## 12. Repository Structure

```
├── classifier.py
├── preprocessor.py
├── features.py
├── train.py
├── evaluate.py
├── data/
├── results/
└── README.md
```

---

*This repository is part of NLU Assignment 1 (Problem 4) and is intended for academic evaluation only.*




