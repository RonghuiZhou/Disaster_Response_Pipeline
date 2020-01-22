import json
import plotly
import pandas as pd
import re
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet','stopwords'])

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
# create an engine
engine = create_engine('sqlite:///../data/DisasterResponse.db')
# read the data into a pandas dataframe
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    

    # Get occurence of each type
    df_copy = df.copy()
    counts_percentage = df_copy.iloc[:,4:].sum()/len(df)
    counts_top10 = counts_percentage.sort_values(ascending = False)[:10]
    df_copy_cols = df_copy.columns													  
   
    # Get top 10 words
       
    top_words = {}

    stop_words = stopwords.words('english')
    punct = [p for p in string.punctuation]


    for message in df['message']:
        for word in message.split():           
            if word.lower() not in stop_words and word.lower() not in punct:
                if word in top_words:
                    top_words[word] += 1
                else:
                    top_words[word] = 1
    
    ax_2 = pd.DataFrame.from_dict(top_words, orient = 'index')
    ax_2.columns = ['Counts']
    top10_words_pct = ax_2.sort_values('Counts', ascending = False)[:10]['Counts']/len(df)
    top10_words = list(ax_2.sort_values('Counts', ascending = False)[:10].index)   



    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

            {
            'data': [
                Bar(
                    x=df_copy_cols,
                    y=counts_top10
                )
            ],

            'layout': {
                'title': 'Top 10 Categories',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45
                }
            }
        }
        
     
        
        
     , {
            'data': [
                Bar(
                    x=top10_words,
                    y=top10_words_pct
                )
            ],

            'layout': {
                'title': 'Top 10 Most Used Words',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }

        
      
        
        
    ]		  

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()