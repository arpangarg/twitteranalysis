import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score


def readData(file_path):
	raw_data = dict()

	with open(file_path, 'rb') as f:
		file_contents = csv.reader(f)
		for line in file_contents:
			raw_data[line[0]] = line[1]

	return raw_data


class Preprocess(object):

	def __init__(self, raw_data):
		self.all_feature_words = self.buildFeatures(raw_data.keys())
		self.X, self.y = self.processData(raw_data.keys(), raw_data.values())

	def getFeatures(self, tweet):
		feature_words = []
		sentences = sent_tokenize(tweet)	

		for sentence in sentences:
			words = word_tokenize(sentence)
			for word in words:
				w = re.search(r'^[A-Za-z0-9-]+$', word.lower())
				if (w and w.group() not in ENGLISH_STOP_WORDS and 
					w.group() not in feature_words):
					feature_words.append(w.group())

		return feature_words

	def buildFeatures(self, raw_tweets):
		all_features = []

		for tweet in raw_tweets:
			feature_words = self.getFeatures(tweet)
			for feature in feature_words:
				if feature not in all_features:
					all_features.append(feature)

		return all_features

	def vectorizeTweet(self, tweet):
		return_vector = [0] * len(self.all_feature_words)
		feature_words = self.getFeatures(tweet)

		for word in feature_words:
			if word in self.all_feature_words:
				return_vector[self.all_feature_words.index(word)] = 1

		return return_vector

	def processData(self, tweets, labels=None):
		X = []
		y = []
		for i in range(len(tweets)):
			X.append(self.vectorizeTweet(tweets[i]))
			if labels:
				y.append(int(labels[i]))

		if labels:
			return X, y
		else:
			return X


if __name__ == '__main__':
	'''
	try naive bayes, logistic regression and SVM
	'''
	# process training data
	raw_data = readData('apple_labeled.csv')
	pp = Preprocess(raw_data)

	# split labeled tweets to training and testing data
	X_train, X_test, y_train, y_test = train_test_split(
		pp.X, pp.y, test_size=0.2, random_state=0
	)

	# test scores with basic models without hyperparameter tuning
	models = {
		'Naive Bayes': MultinomialNB,
		'Logistic Regression': LogisticRegression,
		'State Vector Machine': SVC
	}

	for name, model in models.items():
		classifier = model()
		classifier.fit(X_train, y_train)
		print ''.join(
			[
				name,
				' F1 Score: ',
				str(f1_score(y_test, classifier.predict(X_test)))
			]
		)
