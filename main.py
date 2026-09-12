"""Bus complaint analyser.

Run ``python main.py`` to train and evaluate the classifier, then classify a
new complaint interactively. The script demonstrates preprocessing,
tokenization, stopword removal, stemming, lemmatization, POS tagging,
bag-of-words, TF-IDF, supervised classification, sentiment, and similarity.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

try:
	from nltk.corpus import stopwords
	from nltk.stem import PorterStemmer, WordNetLemmatizer
	from nltk.tokenize import word_tokenize
except ImportError:  # pragma: no cover - requirements.txt installs NLTK.
	stopwords = PorterStemmer = WordNetLemmatizer = word_tokenize = None


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "bus_complaints.csv"


def _english_stopwords() -> set[str]:
	"""Use NLTK stopwords when downloaded, with a small offline fallback."""
	fallback = {
		"a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
		"for", "from", "had", "has", "have", "he", "her", "his", "i",
		"in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
		"that", "the", "their", "them", "there", "they", "this", "to",
		"was", "we", "were", "with", "you", "your",
	}
	if stopwords is None:
		return fallback
	try:
		return set(stopwords.words("english"))
	except LookupError:
		return fallback


STOP_WORDS = _english_stopwords()
STEMMER = PorterStemmer() if PorterStemmer else None
LEMMATIZER = WordNetLemmatizer() if WordNetLemmatizer else None


def clean_text(text: str) -> str:
	"""Normalize complaint text while preserving useful transport words."""
	text = str(text).lower()
	text = re.sub(r"https?://\S+|www\.\S+", " ", text)
	text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " email ", text)
	text = re.sub(r"[^a-z0-9\s]", " ", text)
	return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
	"""Tokenize with NLTK when available, otherwise use a regex tokenizer."""
	if word_tokenize:
		try:
			return word_tokenize(text)
		except LookupError:
			pass
	return re.findall(r"[a-z0-9]+", text)


def lemmatize_tokens(tokens: Iterable[str]) -> list[str]:
	result = []
	for token in tokens:
		if token in STOP_WORDS or len(token) < 2:
			continue
		if LEMMATIZER:
			try:
				token = LEMMATIZER.lemmatize(token)
			except LookupError:
				pass
		result.append(token)
	return result


def stem_tokens(tokens: Iterable[str]) -> list[str]:
	if not STEMMER:
		return list(tokens)
	return [STEMMER.stem(token) for token in tokens]


def preprocess(text: str) -> str:
	"""Return the lemmatized corpus used by the machine-learning models."""
	return " ".join(lemmatize_tokens(tokenize(clean_text(text))))


def bag_of_words_demo(texts: list[str]) -> tuple[int, tuple[int, int]]:
	vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1)
	matrix = vectorizer.fit_transform(texts)
	return len(vectorizer.vocabulary_), matrix.shape


def sentiment_score(text: str) -> str:
	"""Small transparent lexicon sentiment feature for complaint urgency."""
	negative = {"angry", "bad", "delay", "dirty", "late", "lost", "rude", "unsafe", "worst"}
	positive = {"helpful", "quick", "safe", "thank", "thanks", "good", "resolved"}
	tokens = set(tokenize(clean_text(text)))
	score = len(tokens & positive) - len(tokens & negative)
	return "negative" if score < 0 else "positive" if score > 0 else "neutral"


def pos_tag_demo(text: str) -> list[tuple[str, str]]:
	"""Return NLTK POS tags when the optional tagger data is available."""
	try:
		from nltk import pos_tag
		return pos_tag(tokenize(clean_text(text)))
	except (ImportError, LookupError):
		return []


def train_system(data: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
	data = data.copy()
	data["processed_text"] = data["complaint"].map(preprocess)
	train_x, test_x, train_y, test_y = train_test_split(
		data["processed_text"], data["department"], test_size=0.25,
		random_state=42, stratify=data["department"],
	)

	model = Pipeline([
		("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
		("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
	])
	model.fit(train_x, train_y)
	predictions = model.predict(test_x)
	print("\nMODEL EVALUATION")
	print(f"Accuracy: {accuracy_score(test_y, predictions):.2%}")
	print(classification_report(test_y, predictions, zero_division=0))
	print("Confusion matrix labels:", list(model.classes_))
	print(confusion_matrix(test_y, predictions, labels=model.classes_))

	comparison = Pipeline([
		("bow", CountVectorizer(ngram_range=(1, 2))),
		("classifier", MultinomialNB()),
	])
	comparison.fit(train_x, train_y)
	print(f"Naive Bayes comparison accuracy: {comparison.score(test_x, test_y):.2%}")
	return model, data, test_x.to_frame(name="processed_text").assign(actual=test_y, predicted=predictions)


def classify(model: Pipeline, data: pd.DataFrame, complaint: str) -> None:
	processed = preprocess(complaint)
	department = model.predict([processed])[0]
	probabilities = model.predict_proba([processed])[0]
	ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)

	corpus_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
	corpus_matrix = corpus_vectorizer.fit_transform(data["processed_text"])
	query_vector = corpus_vectorizer.transform([processed])
	similarities = (corpus_matrix @ query_vector.T).toarray().ravel()
	nearest = int(np.argmax(similarities))

	print("\nPREDICTION")
	print(f"Department: {department}")
	print(f"Confidence: {max(probabilities):.2%}")
	print(f"Sentiment: {sentiment_score(complaint)}")
	print(f"Closest known complaint: {similarities[nearest]:.2%} similarity")
	print("Department ranking:", ", ".join(f"{label} ({score:.1%})" for label, score in ranked))
	print("Tokens:", tokenize(clean_text(complaint)))
	print("Stemmed tokens:", stem_tokens(tokenize(clean_text(complaint))))
	tags = pos_tag_demo(complaint)
	if tags:
		print("POS tags:", tags)


def main() -> None:
	data = pd.read_csv(DATA_FILE)
	required = {"complaint", "department"}
	if not required.issubset(data.columns):
		raise ValueError(f"Dataset must contain columns: {sorted(required)}")
	print("BUS COMPLAINT ANALYSER")
	print(f"Records: {len(data)} | Departments: {data['department'].nunique()}")
	vocabulary_size, bow_shape = bag_of_words_demo(data["complaint"].map(preprocess).tolist())
	print(f"Bag-of-Words vocabulary: {vocabulary_size} | matrix shape: {bow_shape}")
	print("Department distribution:")
	print(data["department"].value_counts().to_string())
	model, processed_data, _ = train_system(data)

	print("\nEnter a complaint to classify, or press Enter to exit.")
	while True:
		complaint = input("Complaint: ").strip()
		if not complaint:
			break
		classify(model, processed_data, complaint)


if __name__ == "__main__":
	main()
