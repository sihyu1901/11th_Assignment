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


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
params_DC = {'criterion' : ['gini', 'entropy'], 'max_depth' : [2,3,4,5], 'min_samples_split' : [2,5,10], 'min_samples_leaf' : [1,2,4]}
gs_DT = GridSearchCV(DecisionTreeClassifier(random_state = 42), params_DC, scoring='accuracy') 


gs_DT.fit(train_input, train_target)

DT = gs_DT.best_estimator_ 

importances_DT = DT.feature_importances_

importances_DT_df = pd.DataFrame({'features': df.columns, 'importance': importances_DT})
importances_DT_df.sort_index(inplace = True)

print(f"Best Hyper-parameters : {gs_DT.best_params_}")  
print(f"Best Score : {DT.score(train_input, train_target)}") 
plt.figure(figsize = (14,5))
plt.bar(importances_DT_df.features, importances_DT_df.importance, width = 0.4)
plt.xticks(rotation = 45, size = 9)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
