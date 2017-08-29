# coding: utf-8
'''
代码来源《Python机器学习及实践-从零开始通往Kaggle竞赛之路》
'''
import pandas as pd
train=pd.read_csv('../datasets/train.csv')
test=pd.read_csv('../datasets/test.csv')

# print train.info()
# print test.info()

selected_features=['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']

X_train=train[selected_features]
X_test=test[selected_features]

y_train=train['Survived']

# print X_train['Embarked'].value_counts()
# print X_test['Embarked'].value_counts()

X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)

X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)

X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

# X_train.info()

from sklearn.feature_extraction import DictVectorizer
dict_vec=DictVectorizer(sparse=False)

# print X_train.to_dict(orient='recored')

X_train=dict_vec.fit_transform(X_train.to_dict(orient='recored'))
# print dict_vec.feature_names_
# print X_train[0]
# print X_train[1]

X_test=dict_vec.transform(X_test.to_dict(orient='record'))

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()

from xgboost import XGBClassifier
xgbc=XGBClassifier()

from sklearn.cross_validation import cross_val_score
print cross_val_score(rfc, X_train, y_train, cv=5).mean()
print cross_val_score(xgbc, X_train, y_train, cv=5).mean()

rfc.fit(X_train, y_train)
rfc_y_predict=rfc.predict(X_test)
rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':rfc_y_predict})
rfc_submission.to_csv('../rst/rfc_submission.csv', index=False)




