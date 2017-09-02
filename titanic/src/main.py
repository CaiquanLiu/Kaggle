# coding: utf-8
'''
代码来源《Python机器学习及实践-从零开始通往Kaggle竞赛之路》
'''
import pandas as pd

# 数据录入
train = pd.read_csv('../datasets/train.csv')
test = pd.read_csv('../datasets/test.csv')
# print train.info()
# print test.info()

# 特征选取
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['Survived']
# print X_train['Embarked'].value_counts()
# print X_test['Embarked'].value_counts()

# 数据预处理
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)
# X_train.info()

# 特征向量化
from sklearn.feature_extraction import DictVectorizer

dict_vec = DictVectorizer(sparse=False)
# print X_train.to_dict(orient='recored')
X_train = dict_vec.fit_transform(X_train.to_dict(orient='recored'))
# print dict_vec.feature_names_
# print X_train[0]
# print X_train[1]
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

# 引入随机森林分类器
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

# 引入boosting分类器
from xgboost import XGBClassifier

xgbc = XGBClassifier()

# 引入预测评估
from sklearn.cross_validation import cross_val_score

# print cross_val_score(rfc, X_train, y_train, cv=5).mean()
# print cross_val_score(xgbc, X_train, y_train, cv=5).mean()

# 使用随机森林模型进行训练和预测
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
rfc_submission.to_csv('../rst/rfc_submission.csv', index=False)

# 使用boosting进行训练和预测
xgbc.fit(X_train, y_train)
xgbc_y_predict = xgbc.predict(X_test)
xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_y_predict})
xgbc_submission.to_csv('../rst/xgbc_submission.csv', index=False)

# 使用并行网络搜索的方式提高XGBClassifier的预测性能
from sklearn.grid_search import GridSearchCV

params = {'max_depth': range(2, 7), 'n_estimators': range(100, 1100, 200), 'learning_rate': [0.05, 0.1, 0.25, 0.5, 1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
gs.fit(X_train, y_train)
print gs.best_score_
print gs.best_params_
xgbc_best_y_predict = gs.predict(X_test)
xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': xgbc_best_y_predict})
xgbc_best_submission.to_csv('../rst/xgbc_best_submission.csv', index=False)
