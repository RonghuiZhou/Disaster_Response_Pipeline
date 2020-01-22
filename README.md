# Disaster Response Pipeline
### Disaster Response Pipelines Built for Data Scientist Nanodegree on Udacity

#### Ronghui Zhou; zhou.uf@gmail.com; [LinkedIn](https://www.linkedin.com/in/ronghuizhou/); Jan. 2020 

## **2. Instructions:**

- Step 1: Run the followihng commands in the project's root directory to set up your database and model.
        
      - Step 1.1: Run ETL pipeline and clean the data and store in the databse: 
                  python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
      
      - Step 1.2: Run ML pipeline that trains classifier and saves the model:
                  python models/strain_classifier.py data/DisasterResponse.db models/classifier.pkl

- Step 2: Run the following command to run your web app:
       python app/run.py

- Step 3: Go to http://localhost:2020



## **3. File structure**
<pre>

├── app
│   ├── run.py--------------------------------------# flask file to run app
│   ├── screenshots
│ 	│	├── overview_of_training_dataset.png----------# screenshot of web app: overview of training dataset
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

## **4. Screenshots:**

The screen shots of the web app are below:

**_Screenshot 1: overview of training dataset_**

![Overview of training dataset](/app/screenshots/overview_of_training_dataset.png)

**_Screenshot 2: classify message_**

![Classify message](/app/screenshots/classify_message.png)


