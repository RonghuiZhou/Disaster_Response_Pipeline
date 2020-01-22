# Disaster Response Pipeline
### Disaster Response Pipelines Built for Data Scientist Nanodegree on Udacity

#### Ronghui Zhou; zhou.uf@gmail.com; Jan. 2020 
[LinkedIn](https://www.linkedin.com/in/ronghuizhou/)


The screen shots of the web app are below:

**_Screenshot 1_**

![screenshot1](/app/screenshots/overview_of_training_dataset.png)

![Overview of training dataset](/app/screenshots/overview_of_training_dataset.png)

**_Screenshot 2_**

![screenshot2](/app/screenshots/classify_message.png)
![Classify message](/app/screenshots/classify_message.png)

<a id='files'></a>

<pre>
.
├── app
│   ├── run.py--------------------------------------# flask file to run app
│   ├── screenshots
│ 	│	├── overview_of_training_dataset.png--------# screenshot of web app: overview of training dataset
│   │ 	└──	classify_message.png--------------------# screenshot of web app: classify message
│   └── templates
│       ├── go.html---------------------------------# classification result page of web app
│       └── master.html-----------------------------# main page of web app
├── data
│   ├── DisasterResponse.db-------------------------# database to save cleaned data
│   ├── categories.csv------------------------------# raw data to process: categories
│   ├── messages.csv--------------------------------# raw data to process: messages
│   └── process_data.py-----------------------------# perform ETL pipline
├── model
│   ├── train_classifier.py-------------------------# perform classification pipeline
│   └── classifier.pkl------------------------------# optimized ML model saved
├── notebook
│   ├── ETL Pipeline Preparation.ipynb--------------# Jupyter notebook for ETL 
│   └── ML Pipeline Preparation.ipynb---------------# Jupyter notebook for ML

</pre>


## Instructions:
