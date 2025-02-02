# Disaster Response Pipeline
### Data Scientist Nanodegree on [Udacity](https://www.udacity.com/); Training Data from [Figure Eight](https://www.figure-eight.com/)


#### Author: [Ronghui Zhou](https://www.linkedin.com/in/ronghuizhou/); Jan. 23, 2020     

## **Motivation**
Following natural disasters, there are usually tons of overwhelming messages. The goal of this project is to build a data processing and interpretation pipeline to speed up the resource allocation process. Time is life!

## **1. Installation**
   1.1 Install [Anaconda](https://www.anaconda.com/) if it is not installed, go to step 1.2 otherwise  
   1.2 Update all libraries in Anaconda Prompt (see library_requirement.txt for a complete list of libraries)  
				```
                conda update -all                           
                ```  
   1.3 Clone this GIT repository:

          git clone https://github.com/RonghuiZhou/Disaster_Response_Pipeline.git

   1.4 Delete (or move to some other place) these two files: data/DisasterResponse.db & models/classifier.pkl
      

## **2. How to run:**

- **Step 1:** Run the followihng commands in the project's root directory to set up your database and model.      

1.1: Run ETL pipeline which cleans the data and stores in the database:     
                  ```python
                     data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db                        
                  ```                              
<p>The two data files (messages.csv and categories.csv) will be merged into one dataset, categories are separated into each column, unrelated information is dropped and values converted to numeric values either 0 or 1. Duplicates will be dropped and the cleaned data will be saved to a sqlite database to feed into the machine learning pipeline.</p>      
		  
1.2: Run ML pipeline (Logistic Regression and grid search to optimize hyperparameters from Scikit-learn) which trains classifier and saves the model:            
                  ```python
                     models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
                  ```            
<p>The cleaned data from previous step (ETL pipeline) will be loaded and messages will be used as the features. Natural language processing toolkit (nltk) will be employed to understand the message after some cleaning, including dropped unrelated characters other than alphabets and numbers, separate into individual words, drop common meaningless stopwords. The goal is to interpret the message and assign related categories to the message so relavant teams will respond promptly. Logistic regression classification model is used here together with TF-IDF technique.</p> 
				
- **Step 2:** Run the following command to run your web app:         
        ```python
        app/run.py
        ```

- **Step 3:** Go to http://localhost:2020



## **3. File structure**
<pre>

├── README.md---------------------------------------# Read this file for all details
│
├── library_requirement.txt-------------------------# A list of libraries in the environment
│
├── app
│   ├── run.py--------------------------------------# flask file to run app
│   ├── screenshots
│   │	├── overview_of_training_dataset.png--------# screenshot of web app: overview of training dataset
│   │ 	└── classify_message.png--------------------# screenshot of web app: classify message
│   └── templates
│       ├── go.html---------------------------------# classification result page of web app
│       └── master.html-----------------------------# main page of web app
├── data
│   ├── DisasterResponse.db-------------------------# database to save cleaned data
│   ├── categories.csv------------------------------# raw data to process: categories
│   ├── messages.csv--------------------------------# raw data to process: messages
│   └── process_data.py-----------------------------# perform ETL pipline
├── models
│   ├── train_classifier.py-------------------------# perform classification pipeline
│   └── classifier.pkl------------------------------# optimized ML model saved
├── notebook
│   ├── ETL Pipeline Preparation.ipynb--------------# Jupyter notebook for ETL 
│   └── ML Pipeline Preparation.ipynb---------------# Jupyter notebook for ML



</pre>

## **4. Screenshots:**

The screenshots of the web app are below:

**_Screenshot 1: overview of training dataset_**

![Overview of training dataset](/app/screenshots/overview_of_training_dataset.png)

**_Screenshot 2: classify message_**

![Classify message](/app/screenshots/classify_message.png)


