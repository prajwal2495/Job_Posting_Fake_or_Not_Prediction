import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import remove_stopwords

class my_model():
    def fit(self, X, y):
        # do not exceed 29 mins
        x_train, x_test, y_train, y_test = self.pre_process(X,y)
        self.tree = DecisionTreeClassifier(X,y)
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        x_train, x_test, y_train, y_test = self.pre_process(X)
        predictions = self.tree.predict(x_test)
        return predictions

    def pre_process(self, X, y):
        pass

        return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("../data/job_train.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)
    # Train model
    clf = my_model()
    clf.fit(X, y)
    runtime = (time.time() - start) / 60.0
    print(runtime)
    predictions = clf.predict(X)
    print(predictions)


"""

Hello Professor @azhe825,

Can we make changes to the skeleton code provided for the project.py ?
I was planning to add an __init__ method and it might take some parameters which would change the line
`clf = my_model()`

will this be alright ? or should we follow what is given ?

Thanks
Prajwal

"""