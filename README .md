
# Employee Turnover Prediction

The goal is to find out the employees who stay and leave the company in the upcoming year.If an employee that you have invested so much time and money leaves for other companies, then this would mean that you would have to spend even more time and money to hire somebody else. Making use of data science and predictive modeling capabilities, if we can predict employee turnover rate will save the company from loss.


# Dataset link
https://drive.google.com/drive/folders/1fmCqKr6DNqy8g3tvlFjMpXPix6f-sLAi


# Importing Libraries

#for Manipulations
import pandas as pd
import numpy as np

#for Data visualizations
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#for Scientific calculations
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

#To Ignore warnings
import warnings
warnings.filterwarnings("ignore")

Programming Language : Python V3.9.7

# Analysis

Task 1: Data cleaning and statistical analysis

Task 2: Feature Engineering

Task 3: Feature Selection

Task 4: Model Building


# Data Sets

After performed Feature engineering, a training data set and a testing data set were made.Training data set was used to train the Machine Learning models, and testing data set was used to find out accuracy of our model.

# Running Tests

To Run the tests, model was evaluated using confusion matrix, accuracy, precision, recall, f1_score.

# Conclusion

In this Project we find out the various insights through four different tasks.

In Task-1,We perform EDA and seperate features from the target value and perform descriptive statistics to find out the mean, median, standara deviation,variance,etc...

In Task-2, We perform the feature engineering technique and findout the outliers in the columns and evaluate target variable.Then 
finding the patterns by using matplotlib and seaborn libraries,we can find out the employees who stay and leave the company in the upcoming year.

In Task-3,Perform feature selection on numerical and categorical features separately.Removing features with zero variance, make use of visualization.

In Task-4,We find out the outliers and detect them and then split the data.Then we take three ensemble models which are RandomForestClassifier, ExtraTreeClassifier Model,GradientBoostingClassifer Model and compare the accuracy and evaluate the confusion matrix,finally we come to know that the RandomForestClassifier Model gives the best accuracy than other two Models.The accuracy of RandomForestClassifier Model is 0.842911877394636








