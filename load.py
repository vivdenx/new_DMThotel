import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import preprocessing as prep
from down_sampling import negative_down_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


#cat training_set_VU_DM.csv | parallel --header : --pipe -N499999 'cat >file_{#}.csv'
#splits into 10 files
cols_to_drop = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_query_affinity_score',
                    'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate_percent_diff',
                    'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff',
                    'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff', 'comp2_rate',
                    'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate', 'comp2_inv',
                    'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv']



def read():
   #data = pd.read_csv(open("./training_set_VU_DM.csv"))

    data = pd.read_csv(open("./file_1.csv"))
    data['month'] = pd.to_datetime(data['date_time']).dt.month
    data['hour'] = pd.to_datetime(data['date_time']).dt.hour
    data.drop('date_time',axis=1,inplace=True)
    data.drop(cols_to_drop, axis=1, inplace=True)
    #print(data.columns)
    down = negative_down_sampling(data)
    down.to_csv("./down_1.csv")



def treeModel():
    data = pd.read_csv(open("./down_1.csv"))
    data.drop(['Unnamed: 0','gross_bookings_usd','position'],axis=1,inplace=True)
    test_data = pd.read_csv(open("./Testy_2.csv"))
    test_data['month'] = pd.to_datetime(test_data['date_time']).dt.month
    test_data['hour'] = pd.to_datetime(test_data['date_time']).dt.hour
    test_data.drop('date_time',axis=1,inplace=True)
    test_data.drop(cols_to_drop, axis=1, inplace=True)


    #THIS IS THE TARGET VARIABLE 
    y = data.click_bool
    data.drop(['click_bool','booking_bool'],axis=1,inplace=True)
    data = data.fillna(0)
    test_data = test_data.fillna(0)
    #data = reset_index(data)
    #X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
    
    rf = RandomForestClassifier(n_estimators=101, min_samples_leaf=5, min_samples_split=3)
    #feature_importance(X_train,y_train,rf)
    sc = StandardScaler()
    rf.fit(data,y)
    #prediction = rf.predict(test_data)
    pred_prob = rf.predict_proba(test_data)
    #print(pred_prob[:,1])
    test_data = test_data[['srch_id','prop_id']]
    s_ids = np.unique(test_data['srch_id'])
    test_data['prediction'] = pred_prob[:,1]
    test_data.sort_values(by=['srch_id','prediction'],ascending=[True,False],inplace=True)
    test_data.drop('prediction', axis=1,inplace=True)
    pred = pd.DataFrame(test_data)
    pred.to_csv("./results.csv",index=False)


    #for second roudn
    #pred.to_csv('results.csv',mode='a',header=False)



    '''
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    rf.fit(X_train, y_train)

    y_pred=rf.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    y_pred_prob = rf.predict_proba(X_test)
    pred = pd.DataFrame(y_pred_prob,y_test)
    pred.to_csv("./hmm.csv")
    '''
def treeModel2():
    data = pd.read_csv(open("./cleaned_data.csv"))
    data.drop(['position'],axis=1,inplace=True)
    test_data = pd.read_csv(open("./Testy_2.csv"))
    test_data.drop('date_time',axis=1,inplace=True)
    #test_data['month'] = pd.to_datetime(test_data['date_time']).dt.month
    #test_data['hour'] = pd.to_datetime(test_data['date_time']).dt.hour
    data.drop(['number_of_comp','aggr_comp_rate_dif','aggr_comp_availability','aggr_comp_price_dif'],axis=1,inplace=True)
    test_data.drop(cols_to_drop, axis=1, inplace=True)
    
    y = data.click_bool
    data.drop(['click_bool','booking_bool'],axis=1,inplace=True)
    data = data.fillna(0)
    test_data = test_data.fillna(0)
    #data = reset_index(data)
    #X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
    
    rf = RandomForestClassifier(n_estimators=101, min_samples_leaf=5, min_samples_split=3)
    #feature_importance(X_train,y_train,rf)
    sc = StandardScaler()
    rf.fit(data,y)
    #prediction = rf.predict(test_data)
    pred_prob = rf.predict_proba(test_data)
    #print(pred_prob[:,1])
    test_data = test_data[['srch_id','prop_id']]
    s_ids = np.unique(test_data['srch_id'])
    test_data['prediction'] = pred_prob[:,1]
    test_data.sort_values(by=['srch_id','prediction'],ascending=[True,False],inplace=True)
    test_data.drop('prediction', axis=1,inplace=True)
    pred = pd.DataFrame(test_data)
    pred.to_csv("./results4.csv",index=False,header=False)  

    #pred.to_csv('./results2.csv',mode='a',header=False)


def treeRegressModel():
    data = pd.read_csv(open("./file_1.csv"))
    data['date_time'] = pd.to_datetime(data['date_time']).dt.month
    data['hour'] = pd.to_datetime(data['date_time']).dt.hour
    data.drop(cols_to_drop, axis=1, inplace=True)
    #print(data.columns)
    data = negative_down_sampling(data)
    

    y = data.click_bool
    data.drop(['click_bool','booking_bool'],axis=1,inplace=True)
    data = data.fillna(0)
    #data = reset_index(data)
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
    rf = RandomForestClassifier(n_estimators=51, min_samples_leaf=5, min_samples_split=3)
    #feature_importance(X_train,y_train,rf)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    rf.fit(X_train, y_train)

    y_pred=rf.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    y_pred_prob = rf.predict_proba(X_test)
    pred = pd.DataFrame(y_pred_prob,y_test)
    pred.to_csv("./hmm.csv")

def feature_importance(X_train,y_train,rf):
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()




if __name__ == "__main__":
    #read()
    treeModel2()
