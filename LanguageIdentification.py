import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from typing import Tuple
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_and_preprocess(train_data: str, dev_data: str, test_data: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = pd.read_csv(train_data, sep='\t', header=None, names=['label','text'])
    X_dev = pd.read_csv(dev_data, sep='\t', header=None, names=['label','text'])
    X_test = pd.read_csv(test_data, sep='\t', header=None, names=['label','text'])

    X_train['Processed_Text'] = X_train['text'].apply(preprocessing_data)
    X_dev['Processed_Text'] = X_dev['text'].apply(preprocessing_data)
    X_test['Processed_Text'] = X_test['text'].apply(preprocessing_data)

    return X_train, X_dev, X_test

def preprocessing_data(text: str) -> dict[Tuple[str, str], int]:
    text_pre = text[:100]
    char_bigrams = [text_pre[i:i + 2] for i in range(len(text_pre) - 1)]
    dict_features = Counter(char_bigrams)
    #unigrams = text_pre.split()
    #dict_features = Counter(unigrams)
    return dict_features

def feature_extraction(train: dict, dev: dict, test: dict):
    vectorizer = DictVectorizer()

    train = vectorizer.fit_transform(train)
    dev = vectorizer.transform(dev)
    test = vectorizer.transform(test)

    return train, dev, test

def train_modelNB(X_train, labels_train):
    clf = MultinomialNB(alpha = 0.1)
    clf_model = clf.fit(X_train, labels_train)
    return clf_model

def train_modelLG(X_train, labels_train):
    clf_log_reg = LogisticRegression(random_state=0, C = 0.5)
    clf_log_model = clf_log_reg.fit(X_train, labels_train)
    return clf_log_model

def validate_model(X_dev, labels_dev, model):
    y_pred = model.predict(X_dev)
    accuracy = accuracy_score(labels_dev, y_pred)
    f1 = f1_score(labels_dev, y_pred, average = 'macro')
    return accuracy, f1

def test_model(X_test, labels_test, model, title):
    y_predict = model.predict(X_test)
    confusion = confusion_matrix(labels_test, y_predict)
    languages = ['kin', 'ind', 'som', 'hau', 'lin', 'aze', 'uzb', 'nbe', 'por', 'orm', 'spa', 'swh', 'fra', 'eng',
                 'hat', 'tur', 'sna']
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels= languages)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    macro_f1 = f1_score(labels_test, y_predict, average = "macro")
    accuracy = accuracy_score(labels_test, y_predict)
    print(f"Accuracy:{accuracy} and F1: {macro_f1} ")
    report = classification_report(labels_test, y_predict)
    print(report)

def grid_search(X_train, labels_train, X_dev, labels_dev):
    alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    for value in alpha_values:
        clf_naive = MultinomialNB(alpha = value)
        clf_model = clf_naive.fit(X_train, labels_train)
        accu, f1 = validate_model(X_dev, labels_dev, clf_model)
        print(f"Accuracy:{accu} and F1: {f1} for alpha={value}")
        clf_log_reg = LogisticRegression(random_state=0, C = value)
        clf_log_model = clf_log_reg.fit(X_train, labels_train)
        accu2, f1 = validate_model(X_dev, labels_dev, clf_log_model)
        print(f"Accuracy:{accu2} and F1: {f1} for C={value}")

def main() -> None:
    df_train, df_dev, df_test = load_and_preprocess("data/mot_train.tsv", "data/mot_dev.tsv", "mot_test.tsv")


    Train_extract, Dev_extract, Test_extract= feature_extraction(df_train['Processed_Text'], df_dev['Processed_Text'], df_test['Processed_Text'])

    #grid_search(Train_extract, df_train['label'], Dev_extract, df_dev['label'])

    naive_bayes_model = train_modelNB(Train_extract, df_train['label'])
    test_model(Test_extract, df_test["label"], naive_bayes_model, "Naive Bayes Confusion Matrix")

    log_reg_model = train_modelLG(Train_extract, df_train['label'])
    test_model(Test_extract, df_test["label"], log_reg_model, "Logistic Regression Confusion Matrix")

if __name__ == "__main__":
    main()