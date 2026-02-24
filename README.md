## 1. Introduction
Language Identification (LI) is the task of identifying the language of a given text segment. While it may seem straightforward, it presents significant challenges when languages share common words, structures, and alphabets. 

In this project, I worked with **17 different languages** that all use the **Latin script**. To differentiate them, I built models that analyze character patterns and bigram distributions rather than relying on script detection. This project was designed to explore the challenges of breaking down languages in a meaningful way for machine learning.

---

## 2. Data Overview

### 2.1 Dataset Description & Origin
The dataset was obtained from the **Multilingual Open Text (MOT) corpus**. It contains multilingual text snippets obtained from news articles, labeled by their respective ISO language codes.

> **Source:** [MOT Corpus on GitHub](https://github.com/bltlab/mot)

### 2.2 Data Format & Examples
All instances within the dataset contain a language label and a paragraph of text. Below are representative examples from the dataset (truncated for display):

* **aze:** `Birl¸smi¸s ¸Statların 2018-ci ild Iranla beynlxalq nüv sazi¸sindn...`
* **spa:** `“Este es uno de los resultados más emocionantes de la recientemente...`
* **swh:** `Aidha wakili huyo alisema kuwa juhudi za kutaka kujua mahali anaposhik- iliwa...`

### 2.3 Data Distribution
The training set contains **48,000 examples**, with the dev and test sets containing **10,000 examples** each. The labels are evenly distributed to prevent model bias.

| Language | Count | Language | Count |
| :--- | :--- | :--- | :--- |
| Kinyarwanda (`kin`) | 2,817 | Portuguese (`por`) | 2,738 |
| Indonesian (`ind`) | 2,790 | Oromo (`orm`) | 2,728 |
| Somali (`som`) | 2,782 | Spanish (`spa`) | 2,727 |
| Hausa (`hau`) | 2,775 | Swahili (`swh`) | 2,727 |
| Lingala (`lin`) | 2,771 | French (`fra`) | 2,725 |
| Azerbaijani (`aze`) | 2,770 | English (`eng`) | 2,719 |

---

## 3. Methodology

### Feature Extraction
* **Truncation:** I tested truncating text to the first 50 vs. 100 characters to evaluate the effect of input length on performance.
* **Vectorization:** Used **Character Bigrams** and **Unigrams** with a `DictVectorizer` to convert text frequencies into training vectors.

### Models & Hyperparameters
I compared two primary classifiers using a **Grid Search** for optimization:
1. **Multinomial Naive Bayes:** Tuned the smoothing parameter $\alpha$.
2. **Logistic Regression:** Tuned the regularization strength $C$.

---

## 4. Results

### Final Test Scores
Based on the development set results, the best-performing models used **100-character truncation**.

| Model | Hyperparameters | Accuracy | Macro F1 |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | $C = 0.5$ | **99.63%** | **99.63%** |
| **Naive Bayes** | $\alpha = 0.1$ | 99.59% | 99.59% |

### Confusion Matrix Comparison
![Model Evaluation Results](https://github.com/user-attachments/assets/469283bc-99aa-4007-b876-417cf6fd7a35)

### Key Insights
* **Character Bigrams vs. Unigrams:** Interestingly, Naive Bayes performed significantly better with **unigrams**, while Logistic Regression reached its peak performance using **character bigrams**.
* **Context Length:** Increasing truncation from 50 to 100 characters provided the models with more information to learn from, resulting in higher precision.
* **Error Analysis:** Most errors occurred between closely related languages (e.g., English and Spanish) due to shared character patterns in shorter text segments.

---

## 5. Conclusion & Future Work
This project demonstrates that character bigram features are highly effective for language identification in Latin-script texts. For future iterations, I would like to explore:
* **Word-level features** combined with n-grams to disambiguate very short texts.
* **Neural Models:** Transitioning from traditional classifiers to Transformer-based architectures.
* **Broader Scope:** Including non-Latin script languages to test the model's versatility.

---
