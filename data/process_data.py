import sys
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	"""
	This function load two csv files and merge them together to a pandas dataframe
	
	Args:
		messages_filepath: full or relative path to the messages file
		categories_filepath: full or relative path to the categories file
	
	Return:
		df: a merged pandas dataframe contain messages and categories		
	"""
	messages = pd.read_csv(messages_filepath)    
	categories = pd.read_csv(categories_filepath)
	df = messages.merge(categories, on='id')
	return df
	
	


def clean_data(df):
	"""
	This function clean the dataframe and return a tidy dataframe ready for machine learning pipeline
	
	Args:
		df: a dirty dataframe
		
	Returns:
		df: a tidy dataframe ready for machine learning pipeline
	"""
	# create a dataframe of the 36 individual category columns
	categories = df['categories'].str.split(';',expand=True)
	
	# select the first row of the categories dataframe
	row = categories.iloc[0]
	# use this row to extract a list of new column names for categories.
	# one way is to apply a lambda function that takes everything 
	# up to the second to last character of each string with slicing
	category_colnames = row.apply(lambda x : x[:-2]).values.tolist()
	
	# rename the columns of `categories`
	categories.columns = category_colnames
	
	# Convert category values to just numbers 0 or 1
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].astype(str).str[-1:]
    
		# convert column from string to numeric
		categories[column] = pd.to_numeric(categories[column])
	
	# drop the original categories column from `df`
	df.drop(['categories'],inplace=True, axis=1)
	
	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df,categories], axis=1,join='inner')
	
	# drop duplicates
	df.drop_duplicates(inplace=True)
	
	# drop rows if the 'related' column is 2 (should be either 0 or 1)
	df=df[df.related != 2]
	
	# drop the 'child_alone' column since it has the same value for all rows
	df.drop(['child_alone'],axis=1,inplace=True)
	return df


def save_data(df, database_filename):
    """
	Save the clean dataset into an sqlite database
	
	Args:
		df: the pandas dataframe to be saved
		database_filename: the name for the sqlite database file		
	"""
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()