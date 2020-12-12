import numpy as np
import pandas as pd
import time
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold
from sklearn import svm
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample 
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import remove_stopwords
sys.path.insert(0, '..')
from assignment8.my_evaluation import my_evaluation



class my_model():
    def fit(self, X, y):
        # do not exceed 29 mins
        X = self.clean_all_data(X)
        
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', 
                                            use_idf=True, smooth_idf=True, ngram_range=(1,4))
        XX = self.preprocessor.fit_transform(X["description"], X["requirements"])
        
        #self.pac = PassiveAggressiveClassifier(class_weight="balanced", random_state=10, C = 0.5)
        
        self.pac = PassiveAggressiveClassifier(random_state=5)
        
        pac_grid = {'class_weight' : ["balanced"],
                    'random_state': [10,5,15],
                    'C':[0.5,1,0.25,0.75],
                    'shuffle':[True,False]
        }
        
        self.rfc = RandomForestClassifier(class_weight="balanced",random_state=5)
        rf_grid = {"max_depth": [10, 15, 25],
                    "criterion": ['gini', 'entropy'],
                    "min_samples_split": [2, 3, 4, 5],
                     "n_estimators": [10]
                    }
        
        self.rsv = RandomizedSearchCV(self.pac, pac_grid, random_state=0, n_jobs=-1)
        
        self.rsv.fit(XX, y)
        
        return
    
    
    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X = self.clean_all_data(X)
        XX = self.preprocessor.transform(X["description"])
        predictions = self.rsv.predict(XX)
        
        return predictions
    
    
    def clean_all_data(self, data_frame):
        #warnings.filterwarnings(action='ignore')

        #fillna to location column
        data_frame['location'] = data_frame.location.fillna('none')

        #fillna to description column
        data_frame['description'] = data_frame.description.fillna('not specified')

        #fillna to requirements column
        data_frame['requirements'] = data_frame.description.fillna('not specified')
        
        #drop unnecassary columns
        data_frame.drop(['telecommuting','has_questions'],axis = 1, inplace = True)  
        
        #mapping fraudulent to T and F, where there is  0 and 1 respectively
        data_frame['has_company_logo'] = data_frame.has_company_logo.map({1 : 't', 0 : 'f'})
        
        #remove any unnecassary web tags in the data set
        data_frame['title'] = data_frame.title.str.replace(r'<[^>]*>', '')
        data_frame['description'] = data_frame.description.str.replace(r'<[^>]*>', '')
        data_frame['requirements'] = data_frame.requirements.str.replace(r'<[^>]*>', '')
        
        
        # removing the characters in data set that are not words and has white spaces 
        for column in data_frame.columns:
            data_frame[column] = data_frame[column].str.replace(r'\W', ' ').str.replace(r'\s$','')
            
        
        # mapping back the columns to original binary values
        #data_frame['has_company_logo'] = data_frame.has_company_logo.map({'t': 1, 'f':0})
        
        self.all_genism_stop_words = STOPWORDS
        
        text_columns = list(data_frame.columns.values)
        
        for columns in text_columns:
            self.remove_stopwords_from_data_train(data_frame,columns)
        
        return data_frame
    
    def remove_stopwords_from_data_train(self,data_frame, column_name):
        data_frame[column_name] = data_frame[column_name].apply(lambda x: " ".join([i for i in x.lower().split() if i not in self.all_genism_stop_words]))