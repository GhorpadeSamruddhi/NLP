# Bus Complaint Analyser

This project classifies passenger complaints into the department that should handle them:
Operations, Safety, Billing, Infrastructure, Maintenance, or Customer Service.

## NLP concepts demonstrated

- Text cleaning: lowercasing, URL/email masking, punctuation and whitespace normalization
- Tokenization and stopword removal using NLTK with an offline fallback
- Porter stemming and WordNet lemmatization
- POS tagging when the NLTK tagger data is installed
- Bag-of-Words with unigram and bigram features
- TF-IDF with sublinear term frequency
- Supervised classification using Logistic Regression
- Naive Bayes model comparison
- Accuracy, classification report, and confusion matrix
- Confidence ranking, sentiment lexicon, and cosine similarity to known complaints

## Run

```text
python -m pip install -r requirements.txt
python main.py
```

Type a complaint at the prompt, for example:

```text
The bus charged me twice for the same ticket
```

The included CSV is a teaching dataset. For a stronger real-world project, replace or extend it with anonymized complaints labelled by the correct department. Keep the same `complaint` and `department` columns.