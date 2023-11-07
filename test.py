import pandas as pd

test=pd.read_csv('newdataset.csv')
test_label=test['Class']
test_features=test.drop(['Class'],axis=1)
test_features=test.drop(['Class'],axis=1)
test_labels=test['Class']
print("Testing accuracy = ",qsvc.score(test_features,test_labels))