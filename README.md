# End to End Machine Learning Project Krish Naik

## Software and Tools Requirements

1. [Github Account](https://github.com)
2. [HerokuAccount](https://heroku.com)
3. [VSCodeIDE](https://code.visualstudio.com/)
4. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)

Create a new environment

```
conda create -p venv python==3.7 -y
```

### Introduction About the Data :

This dataset is about students' performance.

**The dataset** The goal is to predict `math_score` of given students.

There are 10 independent variables (including `id`):

* `gender`
* `race_ethnicity` 
* `parental_level_of_education` 
* `lunch`
* `test_preparation_course` 
* `reading_score`
* `writing_score`

Target variable:
* `math_score`

Dataset Source Link :
[https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)


# AWS Deployment Link :

AWS Elastic Beanstalk link : [http://gemstonepriceutkarshgaikwad-env.eba-7zp3wapg.ap-south-1.elasticbeanstalk.com/](http://gemstonepriceutkarshgaikwad-env.eba-7zp3wapg.ap-south-1.elasticbeanstalk.com/)

<!-- This is a comment and won't be displayed in the final output.
# Screenshot of UI

![HomepageUI](./Screenshots/HomepageUI.jpg)  -->

# YouTube Video Link

Link for YouTube Video : Click the below thumbnail to open 

[![https://www.youtube.com/watch?v=1m3CPP-93RI](https://i.ytimg.com/an_webp/1m3CPP-93RI/mqdefault_6s.webp?du=3000&sqp=CMrfr6EG&rs=AOn4CLBYEtIidgQZES-Lo2OPtQyR9iXI4Q)](https://www.youtube.com/watch?v=1m3CPP-93RI)

<!--
# AWS API Link

API Link : [http://gemstonepriceutkarshgaikwad-env.eba-7zp3wapg.ap-south-1.elasticbeanstalk.com/predictAPI](http://gemstonepriceutkarshgaikwad-env.eba-7zp3wapg.ap-south-1.elasticbeanstalk.com/predictAPI)

# Postman Testing of API :

![API Prediction](./Screenshots/APIPrediction.jpg) -->

# Approach for the project 

1. Data Ingestion : 
    * In Data Ingestion phase the data is first read as csv. 
    * Then the data is split into training and testing and saved as csv file.

2. Data Transformation : 
    * In this phase a ColumnTransformer Pipeline is created.
    * for Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed on numeric data.
    * for Categorical Variables SimpleImputer is applied with most frequent strategy, then ordinal encoding performed , after this data is scaled with Standard Scaler.
    * This preprocessor is saved as pickle file.

3. Model Training : 
    * In this phase base model is tested . The best model found was catboost regressor.
    * After this hyperparameter tuning is performed on catboost and knn model.
    * A final VotingRegressor is created which will combine prediction of catboost, xgboost and knn models.
    * This model is saved as pickle file.

4. Prediction Pipeline : 
    * This pipeline converts given data into dataframe and has various functions to load pickle files and predict the final results in python.

5. Flask App creation : 
    * Flask app is created with User Interface to predict the math score of a student inside a Web Application.

## Exploratory Data Analysis Notebook

Link : [EDA Notebook](./notebook/1_EDA_studentPerformance.ipynb)

## Model Training Approach Notebook

Link : [Model Training Notebook](./notebook/2_ModelTraining.ipynb)

<!--
# Model Interpretation with LIME 

Link : [LIME Interpretation](./notebook/3_Explainability_with_LIME.ipynb)-->