# Bus Complaint Analyser

An NLP-based machine learning system that reads passenger complaints and classifies them into the department responsible for handling them.

This project was created for an NLP subject assignment and demonstrates a complete text-classification workflow using Python and scikit-learn.

## Problem Statement

Bus transport organizations receive complaints about delays, fares, safety, vehicle condition, bus stops, and customer support. Manually forwarding every complaint is slow and inconsistent.

The Bus Complaint Analyser automatically predicts the most appropriate department for each complaint.

## Departments

The current teaching dataset contains six complaint categories:

- Operations
- Safety
- Billing
- Infrastructure
- Maintenance
- Customer Service

## NLP Concepts Demonstrated

- Text cleaning and normalization
- Lowercasing and punctuation removal
- URL and email masking
- Tokenization using NLTK
- Stopword removal
- Porter stemming
- WordNet lemmatization
- Part-of-speech tagging
- Bag-of-Words representation
- Unigrams and bigrams
- TF-IDF feature extraction
- Logistic Regression classification
- Multinomial Naive Bayes comparison
- Accuracy and classification report
- Confusion matrix
- Prediction confidence ranking
- Sentiment scoring with a complaint lexicon
- Cosine similarity with known complaints

## Project Structure

```text
NLP/
├── main.py                       # Main training and prediction application
├── bus_complaints.csv            # Labeled complaint dataset
├── requirements.txt              # Python dependencies
├── BUS_COMPLAINT_ANALYSER.md     # Detailed project notes
├── Assignment_02.py              # Earlier preprocessing assignment
├── assignment_3.py               # Earlier TF-IDF and similarity assignment
├── resumes.csv                   # Dataset for the earlier assignment
└── academic_submissions.json     # Dataset for the earlier assignment
```

## Requirements

- Python 3.9 or newer
- pip

## Installation

Clone the repository and open its directory:

```bash
git clone https://github.com/your-username/bus-complaint-analyser.git
cd bus-complaint-analyser
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python main.py
```

The program trains the models, prints evaluation results, and opens an interactive complaint prompt.

Example input:

```text
The ticket machine charged my card twice for one journey
```

Example output:

```text
Department: Billing
Confidence: 28.91%
Sentiment: neutral
```

Press Enter on an empty prompt to exit.

## Model Workflow

```text
Complaint text
      |
      v
Text cleaning and normalization
      |
      v
Tokenization, stopword removal, and lemmatization
      |
      v
TF-IDF with unigram and bigram features
      |
      v
Logistic Regression classifier
      |
      v
Predicted department and confidence
```

The application also trains a Multinomial Naive Bayes model with Bag-of-Words features as a baseline comparison.

## Dataset Format

The classifier expects a CSV file with these columns:

| Column | Description |
|---|---|
| `id` | Unique complaint identifier |
| `complaint` | Passenger complaint text |
| `department` | Correct department label |

Example:

```csv
id,complaint,department
1,"The bus arrived late and I missed my class",Operations
2,"The driver was driving dangerously",Safety
3,"My card was charged twice",Billing
```

The included dataset contains 48 anonymized teaching examples, with eight examples per department. For a production-quality model, use a larger real-world dataset with consistently reviewed labels.

## Evaluation

The application automatically prints:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Naive Bayes comparison accuracy

Because the included dataset is intentionally small, evaluation results are suitable for demonstrating the NLP workflow rather than measuring production performance. A larger and more varied labeled dataset should be used for reliable deployment.

## Future Improvements

- Add more real and anonymized complaint records
- Support Hindi and other regional languages
- Add a web interface using Streamlit or Flask
- Save and load the trained model with `joblib`
- Add complaint priority and urgency classification
- Detect duplicate complaints
- Add database storage for complaint history
- Add cross-validation and hyperparameter tuning
- Create charts for department trends and complaint volume

## Academic Use

This repository is intended for educational use and demonstrates how classical NLP techniques can be combined with supervised machine learning for text classification.
