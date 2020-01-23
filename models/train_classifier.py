# import libraries
import sys
import time
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

import re

import nltk
nltk.download(['punkt','wordnet','averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

def load_data(database_filepath):
	"""
	load data from database_filepath
	Args:
		database_filepath: full or relative path to the database file
		
	Returns:
		X: feature
		Y: target
		category_names: labels	
	"""
	# create an engine
	engine = create_engine('sqlite:///{}'.format(database_filepath)) 
	# read the data into a pandas dataframe
	df = pd.read_sql_table('DisasterResponse', engine)
	# define feature
	X = df['message']
	
	# define target
	Y = df.iloc[:,4:]
	
	# define labels
	category_names = Y.columns
	
	return X, Y, category_names


def tokenize(text):
    """
	process text data: text extraction, tokenization, lemmatization, & stopwords removal
	Args:
		text: input the
	
	Returns:
		clean tokens
	
	"""
	 
    # extrac text based on the pattern
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # get tokens and remove stopwords
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
	# text normalization: get the root stem
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
	"""
	build a machine learning pipeline
	
	"""
	pipeline = Pipeline([("vect",CountVectorizer(tokenizer=tokenize)),
                    ("tfidf",TfidfTransformer()),
                    ("clf", MultiOutputClassifier(LogisticRegression()))])
	parameters = {
    "clf__estimator__C" : [0.1,1,10]
    }
	model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=3, verbose=2)
	return model

def metrics(y_true, y_pred):
	"""
	define metrics to evaluate the machine learning pipeline
	
	Args:
		y_true: the actual target, as y_test
		y_pred: the predicted target
	
	Returns:
		a pandas dataframe of precision, recall, fbeta-score, and accuracy
	"""

	labels = []
	precision=[]
	recall=[]
	fbeta_score=[]
	accuracy=[]
    
	target_names=['class0','class1']      
    
	for i in range(y_true.shape[1]):
		labels.append(y_true.columns.tolist()[i])
        
		report=classification_report(y_true.iloc[:,i],y_pred[:,i], target_names=target_names)
        
		precision.append(float(report[-40:-30].strip()))
		recall.append(float(report[-30:-20].strip()))
		fbeta_score.append(float(report[-20:-10].strip()))
        
		accuracy.append(accuracy_score(y_true.iloc[:,i],y_pred[:,i]))    
    
	metrics_df = pd.DataFrame(data = {'Precision':precision,'Recall':recall,'F1-Score':fbeta_score, 'Accuracy': accuracy}, index = labels)
	return metrics_df


def evaluate_model(model, X_test, Y_test, category_names):
	"""
	evaluate the performance of the machine learning pipeline on test data
	
	Args:
		model: traning model
		X_test: test feature
		Y_test: test target
		category_names: labels
	"""
	# predict target using the training model
	Y_pred = model.predict(X_test)
    
	# print classification report
	metrics_df = metrics(Y_test, Y_pred)
	print(metrics_df)
	print("F1 score mean : ", metrics_df['F1-Score'].mean())


def save_model(model, model_filepath):
	pickle.dump(model, open(model_filepath, 'wb'))


def main():
	if len(sys.argv) == 3:
		database_filepath, model_filepath = sys.argv[1:]
		print('Loading data...\n    DATABASE: {}'.format(database_filepath))
		X, Y, category_names = load_data(database_filepath)
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
		print('Building model...')
		model = build_model()
        
		print('Training model...')
		model.fit(X_train, Y_train)
        
		print('Evaluating model...')
		evaluate_model(model, X_test, Y_test, category_names)

		print('Saving model...\n    MODEL: {}'.format(model_filepath))
		save_model(model, model_filepath)

		print('Trained model saved!')

	else:
		print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()