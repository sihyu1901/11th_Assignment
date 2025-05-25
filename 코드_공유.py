import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

target = wine.target
feature = wine.data

train_input, test_input, train_target, test_target = train_test_split(feature,
                                                    target, 
                                                    test_size=0.2, 
                                                    random_state=42)



from xgboost.sklearn import XGBClassifier

params_XGB = { 'max_depth' : [3, 5, 7, 9, 15],'learning_rate' : [0.1, 0.01, 0.001], 'n_estimators' : [50, 100, 200, 300]}
gs_XGB = GridSearchCV(XGBClassifier(random_state = 42), params_XGB, scoring='accuracy')
gs_XGB.fit(train_input, train_target)

XGB = gs_XGB.best_estimator_

importances_XGB = XGB.feature_importances_

importances_XGB_df = pd.DataFrame({'features': df.columns, 'importance': importances_XGB})
importances_XGB_df.sort_index(inplace = True)

print(f"Beest Hyper-parameters : {gs_XGB.best_params_}") 
print(f"Best Score : {XGB.score(train_input, train_target)}")
plt.figure(figsize = (14,5))
plt.bar(importances_XGB_df.features, importances_XGB_df.importance, width = 0.4)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation = 45,size = 9)
plt.title('Feature Importance')
