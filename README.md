# Sentiment Analysis IMDB Movie Reviews (TF-IDF Text Embedding)
This project performs binary sentiment classification (positive/negative) on IMDB movie reviews using classical machine learning methods (Logistic Regression and SVM) with TF-IDF embeddings

## 🗂️ Dataset 
- [IMDB Movie Reviews](https://github.com/troyhunterz/imdb-moviereviews/dataset)
- Labels:
  - `reviews` - raw movie review text
  - `sentiment` - label (`positive` or `negative`)

## 📚 Technologies & Libraries
- Python 3.12.4
- pandas, numpy
- scikit-learn (Pipeline, GridSearchCV, TfidfVectorizer, LogisticRegression, LinearSVC)
- Custom library: [`preprocess_tr`](https://github.com/troyhunterz/preprocess_tr)
- Pickle (model saving)

## 🧹 Preprocessing
Each review is cleaned using the `preprocess_tr` library:
- Expand contractions
- Remove accents, HTML, emails, URLs
- Lemmatization
- Lowercasing

## 🤖 Model Training
### 1. Logistic Regression + TF-IDF
- Pipeline with TF-IDF + LogisticRegression
- Grid search over:
  - n-gram range
  - IDF 
  - analyzer: word / char / char_wb
  - regularization (L1/L2, C values)

### 2. SVM (LinearSVC) + TF-IDF
- Same pipeline as above, with SVM classifier
- Grid search over hyperparameters

## 🔍 Evaluation
- Accuracy
- F1-score
- Precision / Recall
- Confusion matrix

## 💾 Model Saving & Inference
The final model is saved to `model/model.pkl` using `pickle`

Example usage:
```python
x = ['This is a great movie. I loved it', 'i have watched this movie. plot is straight. return my money']
```
## ⚙️ Setup Instructions
```bash
git clone https://github.com/troyhunterz/imdb-sentiment-tfidf.git
cd imdb-sentiment-tfidf
```

## 🧾 License
This project is licensed under the MIT License.

## 👤 Author
troyhunterz

email: ann0nfolder@gmail.com