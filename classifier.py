import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from src import preprocessor, features, train, evaluate

def classify_doc(model: MultinomialNB | LogisticRegression | SVC, vectorizer: CountVectorizer | TfidfVectorizer) -> None:
    """
    Interactive loop to classify user-submitted files.
    """

    print("\n" + "="*80)
    print(f"INTERACTIVE DOCUMENT CLASSIFIER")
    print("="*80)
    print("Submit a file (.txt or .pdf) to test if it's Sports or Politics.")
    print("Type 'exit' to quit.\n")

    while True:
        filepath = input("Enter file path: ").strip()

        if filepath.lower() in ['exit', 'quit', 'bye', 'end']:
            print("Exiting Classifier...")
            return
        
        # 1. Read and clean
        raw_content = preprocessor.read_doc(filepath)
        if not raw_content:
            print("Warning: File was Empty.")
            continue
        
        cleaned_text = preprocessor.clean_text(raw_content)
        if not cleaned_text:
            print("Warning: File was empty or unreadable.")
            continue

        # 2. Vectorize (Must use the SAME vectorizer from training)
        features_vec = vectorizer.transform([cleaned_text])

        # 3. Predict
        prediction = model.predict(features_vec)[0]
        print(f"\n Classification Result: {prediction.upper()}\n")

if __name__ == "__main__":

    # 1. Load and Preprocess
    data_dir = "data"
    x, y = preprocessor.load_data(data_dir)

    if len(x) == 0:
        print("Error: No data found. Please populate 'data/sports' and 'data/politics'.")
        sys.exit(1)
    
    # 2. KFold for better model evaluation (5-fold)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. K-Fold Training and Evaluation

    all_results = []

    fold_no = 1

    for train_idx, test_idx in kfold.split(x, y):
        print(f"\n========== Fold {fold_no} ==========")

        # Split manually
        x_train = [x[i] for i in train_idx]
        x_test  = [x[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test  = [y[i] for i in test_idx]

        # -------------------------
        # MODEL 1: Naive Bayes
        # -------------------------
        x_train_BoW, x_test_BoW, vec_BoW = features.get_BoW(x_train, x_test)
        model_nb = train.train_naive_bayes(x_train_BoW, y_train)
        results_nb = evaluate.evaluate_model(
            model_nb, x_test_BoW, y_test,
            "Naive Bayes", "Bag of Words"
        )
        all_results.append(results_nb)

        # -------------------------
        # MODEL 2: Logistic Regression
        # -------------------------
        x_train_tfidf, x_test_tfidf, vec_tfidf = features.get_TF_IDF(x_train, x_test)
        model_lr = train.train_logisitc_regression(x_train_tfidf, y_train)
        results_lr = evaluate.evaluate_model(
            model_lr, x_test_tfidf, y_test,
            "Logistic Regression", "TF-IDF"
        )
        all_results.append(results_lr)

        # -------------------------
        # MODEL 3: SVM
        # -------------------------
        x_train_svm, x_test_svm, vec_svm = features.get_TF_IDF(
            x_train, x_test, ngram_range=(1,2)
        )
        model_svm = train.train_svm(x_train_svm, y_train)
        results_svm = evaluate.evaluate_model(
            model_svm, x_test_svm, y_test,
            "SVM", "TF-IDF + Bigrams"
        )
        all_results.append(results_svm)

        fold_no += 1    

    # 4. Export Result Comparision
    df = pd.DataFrame(all_results)

    mean_results = df.groupby("Model").mean(numeric_only=True).reset_index()

    print("\n================ MEAN K-FOLD RESULTS ================")
    print(mean_results) 

    evaluate.save_results(mean_results.to_dict(orient='records'))
    # 5. Select Best Model
    print("\n" + "="*60)
    print("MODEL SELECTION")
    print("="*60)   

    best_model = mean_results.loc[mean_results["Accuracy"].idxmax()]
    best_model_name = best_model["Model"]
    best_model_acc = best_model["Accuracy"]

    print(f"Selecting best model: {best_model_name}")
    print(f"Performance (Accuracy): {best_model_acc * 100:.4f}%")

    # 5. Start Interactive Classifier using best model

    print(f"Best model selected from K-Fold: {best_model_name}")

    if best_model_name == "Naive Bayes":
        x_all, _, vec = features.get_BoW(x, x)
        final_model = train.train_naive_bayes(x_all, y)

    elif best_model_name == "Logistic Regression":
        x_all, _, vec = features.get_TF_IDF(x, x)
        final_model = train.train_logisitc_regression(x_all, y)

    else:  # SVM
        x_all, _, vec = features.get_TF_IDF(x, x, ngram_range=(1, 2))
        final_model = train.train_svm(x_all, y)

    print("Final model trained on full dataset.")


    classify_doc(final_model, vec)




