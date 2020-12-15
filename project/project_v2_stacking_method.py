import pandas as pd
import time
import sys

import time
from sklearn.metrics import classification_report,  f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from scipy.sparse import hstack
import xgboost as xgb


from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.parsing.preprocessing import preprocess_string



sys.path.insert(0, '..')

class NumberSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

class TextSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class my_model():
    
          
    
    def fit(self, X, y):
        
        text_column = X['title'] + ' ' + X['location'] + ' ' + X['requirements'] + ' ' + X['description']
        X['text'] = text_column
        
        text = Pipeline([
                ('selector', TextSelector(key='text')),
                ('tfidf', TfidfVectorizer(stop_words='english', norm='l2', use_idf=True,
                                            smooth_idf=True, max_df=0.3, sublinear_tf=True, lowercase=False)) ])

        XX= text.fit_transform(X)
        
        
        other1 =  Pipeline([
                ('selector', NumberSelector(key='has_questions')),
                ('no-transform', None )
            ])

        other1.fit_transform(X)
        
        other2 =  Pipeline([
                ('selector', NumberSelector(key='has_company_logo')),
                ('no-transform', None )
            ])

        other2.fit_transform(X)
        
        
        other3 =  Pipeline([
                ('selector', NumberSelector(key='telecommuting')),
                ('no-transform', None )
            ])

        other3.fit_transform(X)
        
        
        
        feats = FeatureUnion([('text', text), 
                      ('questions', other1), ('logo', other2), ('telecommute', other3)])
        
        feature_processing = Pipeline([('feats', feats)])
        feature_processing.fit_transform(X)

        param_xgb={'subsample': 1.0, 'min_child_weight': 5, 'max_depth': 20, 'gamma': 0.5, 'colsample_bytree': 1.0, 'learning_rate' :0.02, 'n_estimators':300}


        estimators_stacking = [
                ('XGB', xgb.XGBClassifier(**param_xgb)),
                
                ('MLPClassifier', MLPClassifier(
                    hidden_layer_sizes=(3,),
                    early_stopping=True,
                    n_iter_no_change=5,
                )),
                ('KNeighborsClassifier', KNeighborsClassifier(
                    n_neighbors=3,
                    metric='cosine',
                    n_jobs=3,
                )),
                ('RandomForestClassifier', RandomForestClassifier(
                    n_estimators=125,
                    min_samples_leaf=3,
                    class_weight='balanced_subsample',
                    n_jobs=3,
                )),
                ('SGDClassifier', SGDClassifier(
                    alpha=1e-05,
                    max_iter=500,
                    tol=1e-4,
                    learning_rate='adaptive',
                    eta0=0.5,
                    early_stopping=True,
                    class_weight={1:0.8, 0:0.2},
                )),
            ]

        clf = StackingClassifier(
            estimators=estimators_stacking,
            final_estimator=GradientBoostingClassifier(
                subsample=0.75,
                min_samples_leaf=3,
                n_iter_no_change=5
            )
        )
        #sgd = SGDClassifier( max_iter=3000, random_state=42, class_weight= {1:0.8, 0:0.2}, penalty='l2', loss='hinge', learning_rate='optimal', eta0=500)       
           
        self.pipeline = Pipeline([('features',feats),('classifier', clf )])

        self.pipeline.fit(X, y)   
        
        #self.clf = SGDClassifier(class_weight="balanced", max_iter=3000, random_state=42)
        #self.clf.fit(XX, y)
              
        return 

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions

        text_column = X['title'] + ' ' + X['location'] + ' ' + X['requirements'] + ' ' + X['description']
        X['text'] = text_column
                      
        predictions =self.pipeline.predict(X)

        
        #predictions = self.clf.predict(featuresval)
        return predictions


if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    # split the date
    
    def split_by_fractions(df:data, fracs:list, random_state:int=42):
        assert sum(fracs)==1.0, 'fractions sum is not 1.0 (fractions_sum={})'.format(sum(fracs))
        remain = df.index.copy().to_frame()
        res = []
        for i in range(len(fracs)):
            fractions_sum=sum(fracs[i:])
            frac = fracs[i]/fractions_sum
            idxs = remain.sample(frac=frac, random_state=random_state).index
            remain=remain.drop(idxs)
            res.append(idxs)
        return [df.loc[idxs] for idxs in res]
    
    train,val = split_by_fractions(data, [0.8,0.2]) # e.g: [ train, validation]

    print("\n\nSplit ratio of Train: Validation : ",  train.shape, val.shape, "\n")
    
    # split the data
    y=train["fraudulent"]
    X= train.drop(['fraudulent'], axis=1)
    
    print("\nShape of Train X and Y :", X.shape, y.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    print("\nShape of Train Split into train and test : ", X_train.shape, y_train.shape, X_test.shape, y_test.shape )
    
    
    
    yvaldn=val["fraudulent"]
    Xvaldn=val.drop(['fraudulent'], axis=1)
    #Xvaldn['text'] = Xvaldn['title'] + ' ' + Xvaldn['location'] + ' ' + Xvaldn['requirements'] + ' ' + Xvaldn['description']
    print("\n\n\nShape of Validation X and Y :", Xvaldn.shape, yvaldn.shape)
    
    clf = my_model()
    clf.fit(X_train, y_train)
    print("fitted")

    predictions = clf.predict(Xvaldn)
    #print(predictions[:3])
    print(f1_score(yvaldn, predictions))

    runtime = (time.time() - start) / 60.0
    print(runtime)