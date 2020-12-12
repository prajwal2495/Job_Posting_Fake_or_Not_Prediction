import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import resample 
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import remove_stopwords

class my_model():
    def fit(self, X, y):
        # do not exceed 29 mins
        self.y_data = y
        
        #count vectorizing description
        description_X = X.description
        
        X_train, X_test, y_train, y_test = train_test_split(description_X, y, test_size = 0.33, shuffle = True)
        
        count_vector = CountVectorizer(stop_words='english').fit(X_train)
        
        data_frame_desc_train = pd.DataFrame(count_vector.transform(X_train).todense(), 
                                       columns = count_vector.get_feature_names())
        
        data_frame_desc_test = pd.DataFrame(count_vector.transform(X_test).todense(),
                                      columns=count_vector.get_feature_names())
        
        ######################Count vectorizing description end ########################
        
        #count vectorizing requirements
        req_x = X.requirements
        
        X_train, X_test, y_train, y_test = train_test_split(req_x, y, test_size = 0.33, shuffle = True)
        
        data_frame_req_train = pd.DataFrame(count_vector.transform(X_train).todense(),
                                             columns=count_vector.get_feature_names())
        
        data_frame_req_test = pd.DataFrame(count_vector.transform(X_test).todense(),
                                            columns=count_vector.get_feature_names())
        
                ######################Count vectorizing requirements end ########################
        
        #count vectorizing requirements
#         title_x = X.title
#         X_train, X_test, y_train, y_test = train_test_split(title_x, y, test_size = 0.33, shuffle = True)
        
#         data_frame_title_train = pd.DataFrame(count_vector.transform(X_train).todense(),
#                                              columns=count_vector.get_feature_names())
        
#         data_frame_title_test = pd.DataFrame(count_vector.transform(X_test).todense(),
#                                             columns=count_vector.get_feature_names())
         
            ######################Count vectorizing requirements end ########################
    
        
        #concatenate all the vectorized data frames 
        training = pd.concat([data_frame_desc_train, data_frame_req_train], axis=1)
        testing = pd.concat([data_frame_desc_test,data_frame_req_test], axis=1)
        
        print("after balancing the data:")
        print(training.shape)
        print(y_train.shape)
        print()
        print(testing.shape)
        print(y_test.shape)
        
        #set_trace()
        
#         tree = DecisionTreeClassifier(max_depth=15)
#         tree.fit(training, y_train)
#         tree.score(testing, y_test)
#         predictions = tree.predict(testing)
        

        boost = AdaBoostClassifier(n_estimators=100,random_state=0)
        boost.fit(training, y_train)
        boost.score(testing, y_test)
        predictions = boost.predict(testing)
        
        print("Classification report:")
        print(metrics.classification_report(y_test,predictions))
        
        return

    def predict(self, X):
        #count vectorizing description
        description_X = X.description
        
        X_train, X_test, y_train, y_test = train_test_split(description_X, y, test_size = 0.33, shuffle = True)
        
        count_vector = CountVectorizer(stop_words='english').fit(X_train)
        
        data_frame_desc_train = pd.DataFrame(count_vector.transform(X_train).todense(), 
                                       columns = count_vector.get_feature_names())
        
        data_frame_desc_test = pd.DataFrame(count_vector.transform(X_test).todense(),
                                      columns=count_vector.get_feature_names())
        
        ######################Count vectorizing description end ########################
        
        #count vectorizing requirements
        req_x = X.requirements
        
        X_train, X_test, y_train, y_test = train_test_split(req_x, y, test_size = 0.33, shuffle = True)
        
        data_frame_req_train = pd.DataFrame(count_vector.transform(X_train).todense(),
                                             columns=count_vector.get_feature_names())
        
        data_frame_req_test = pd.DataFrame(count_vector.transform(X_test).todense(),
                                            columns=count_vector.get_feature_names())
        
                ######################Count vectorizing requirements end ########################
        
        #count vectorizing requirements
#         title_x = X.title
#         X_train, X_test, y_train, y_test = train_test_split(title_x, y, test_size = 0.33, shuffle = True)
        
#         data_frame_title_train = pd.DataFrame(count_vector.transform(X_train).todense(),
#                                              columns=count_vector.get_feature_names())
        
#         data_frame_title_test = pd.DataFrame(count_vector.transform(X_test).todense(),
#                                             columns=count_vector.get_feature_names())
         
            ######################Count vectorizing requirements end ########################
    
        
        #concatenate all the vectorized data frames 
        training = pd.concat([data_frame_desc_train, data_frame_req_train], axis=1)
        testing = pd.concat([data_frame_desc_test,data_frame_req_test], axis=1)

#         tree = DecisionTreeClassifier(max_depth=15)
#         tree.fit(training, y_train)
#         tree.score(testing, y_test)
#         predictions = tree.predict(testing)
        
        boost = AdaBoostClassifier(n_estimators=50,random_state=0)
        boost.fit(training, y_train)
        boost.score(testing, y_test)
        predictions = boost.predict(testing)
        
        return predictions
    
    def clean_data_frame(self,data_frame):
                
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
        data_frame['fraudulent'] = data_frame.fraudulent.map({1 : 't', 0 : 'f'})
        
        #remove any unnecassary web tags in the data set
        data_frame['title'] = data_frame.title.str.replace(r'<[^>]*>', '')
        data_frame['description'] = data_frame.description.str.replace(r'<[^>]*>', '')
        data_frame['requirements'] = data_frame.requirements.str.replace(r'<[^>]*>', '')
        
        # removing the characters in data set that are not words and has white spaces 
        for column in data_frame.columns:
            data_frame[column] = data_frame[column].str.replace(r'\W', ' ').str.replace(r'\s$','')
            
        # mapping back the columns to original binary values
        #data_frame['has_company_logo'] = data_frame.has_company_logo.map({'t': 1, 'f':0})
        data_frame['fraudulent'] = data_frame.fraudulent.map({'t': 1, 'f':0})
        
        #add all STOPWORDS from genism to a list
        self.all_gensim_stop_words = STOPWORDS

        #adding all the columns in the data_frame to a list
        text_columns = list(data_frame.columns.values)
        text_columns.remove('fraudulent')

        #cleaning all the columns by removing the stopwords in each of them
        for columns in text_columns:
            self.clean_all_columns(data_frame,columns)
        
        # as 1 and 0 values in the fraudulent class is highly unbalanced
        # true = 0 and fake = 1
        # 0 : 1 == 8484 : 456
        Class_1 = data_frame[data_frame.fraudulent == 1]
        Class_0 = data_frame[data_frame.fraudulent == 0]

        Class_0_count, Class_1_count = data_frame.fraudulent.value_counts()

        Class_0_undersampling = Class_0.sample(Class_1_count-140)
        #Class_0_undersampling = Class_0.sample(400)


        data_frame_undersample = pd.concat([Class_0_undersampling, Class_1], axis=0)
        
        
        return data_frame_undersample
    
    def clean_all_columns(self,data_frame,column_name):
        data_frame[column_name] = data_frame[column_name].apply(lambda x: " ".join([i for i in x.lower().split() if i not in self.all_gensim_stop_words]))



if __name__ == "__main__":
    start = time.time()

    # Load data
    data = pd.read_csv("../data/job_train.csv")
    
    clf = my_model()

    # Replace missing values with empty strings
    data = data.fillna("")

    data = clf.clean_data_frame(data)

    y = data["fraudulent"]
    X = data.drop(['fraudulent'], axis=1)

    # Train model
    clf.fit(X, y)

    runtime = (time.time() - start) / 60.0
    print("Total Runtime:",runtime)

    print("Predictions:")
    predictions = clf.predict(X)
    print(predictions)