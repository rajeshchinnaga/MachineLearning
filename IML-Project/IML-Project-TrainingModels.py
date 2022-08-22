#!/usr/bin/env python
# coding: utf-8

# # **IML Project - Fraudulent website Detection**
# 
# 
# **Team**
# 1. ADARSH REDDY
# 2. VENKATA B
# 3. RAJESH CHINNAGA

# ## **Step 1. Loading Data:**

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


urlData = pd.read_csv('phishingProjectDataset.csv')
urlData.head()


# In[3]:


print("Dataset Features: ",urlData.columns)


# ## **Step 2. Visualization of Trends between Data Features:**

# In[4]:


urlData.hist(bins = 40,figsize = (20,20))
plt.show()


# In[5]:


plt.figure(figsize=(20,15))
sns.heatmap(urlData.corr())
plt.show()


# ## **Step 3. Data Preprocessing :**

# In[6]:


urlData.describe()


# The majority of the binary values in the aforementioned data, with the exception of the features "Domain" and "URL Depth," are 0s and 1s, as can be seen. Domain feature is not required for the training of the machine learning model or for predicting the target values for the output data set. Additionally, URL Depth values range from 0 to 20, but we feel that because this won't be an issue while training the model, we can leave the values at their current levels.

# In[7]:


urlTestData = urlData.drop(['Domain'], axis = 1).copy()


# In[8]:


urlTestData.isnull().sum()


# In[9]:


# To prevent overfitting during training the model, we must shuffle the data before dividing it into test and training sets.
urlTestData = urlTestData.sample(frac=1).reset_index(drop=True)
urlTestData.head()


# In[10]:


y = urlTestData['Label']
X = urlTestData.drop('Label',axis=1)
X.shape, y.shape


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.33, random_state = 42)
X_train.shape, X_test.shape
y_train.shape, y_test.shape


# ## **Step 4. Machine Learning Models & Training**
# 

# ### **1. Decision Tree**

# In[12]:


from sklearn.metrics import accuracy_score
Models = []
traningSetResult = []
testSetReslt = []

#function to store the performace metrics of individual models
def storeMetrics(model, trainResult, testResult):
  Models.append(model)
  traningSetResult.append(round(trainResult, 3))
  testSetReslt.append(round(testResult, 3))


# In[13]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 5)
tree.fit(X_train, y_train)


# In[14]:


y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)


# In[15]:


train_accuracy = accuracy_score(y_train,y_train_tree)
test_accuracy = accuracy_score(y_test,y_test_tree)
print("Decision Tree:")
print("Accuracy on training Data: {}".format(train_accuracy))
print("Accuracy on test Data: {}".format(test_accuracy))


# In[16]:


#Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_test_tree)
classificationReport = classification_report(y_test, y_test_tree, labels=tree.classes_,
                                   target_names=tree.classes_,
                                   output_dict=True)


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted label');ax.set_ylabel('True label'); 
ax.set_title('Decision Tree - Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Legit', 'Fraud']); ax.yaxis.set_ticklabels(['Legit', 'Fraud']);


# In[18]:


import seaborn as sns
ax= plt.subplot()
sns.heatmap(pd.DataFrame(classificationReport).iloc[:-1, :].T, annot=True)
ax.set_title('Decision Tree - Calssification Report'); 


# In[19]:


#Visualizing the decision tree using GraphViz.
from sklearn.tree import export_graphviz
import pydotplus
import IPython.display as display
from IPython.display import Image
export_graphviz(tree, out_file="tree-structure.dot", feature_names=X_train.columns,class_names=['0','1'], filled=True, rounded=True)
g = pydotplus.graph_from_dot_file(path="tree-structure.dot")
display.display(Image(g.create_png()))


# In[20]:


plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()


# In[21]:


storeMetrics('Decision Tree', train_accuracy, test_accuracy)


# ### **2. Random Forest Classifier**

# In[22]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=5)
forest.fit(X_train, y_train)


# In[23]:


y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)


# In[24]:


train_accuracy = accuracy_score(y_train,y_train_forest)
test_accuracy = accuracy_score(y_test,y_test_forest)
print("Random forest:")
print("Accuracy on training Data: {}".format(train_accuracy))
print("Accuracy on test Data: {}".format(test_accuracy))


# In[25]:


#Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_test_forest)
classificationReport = classification_report(y_test, y_test_forest, labels=forest.classes_,
                                   target_names=forest.classes_,
                                   output_dict=True)


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted label');ax.set_ylabel('True label'); 
ax.set_title('Random Forest Classifier - Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Legit', 'Fraud']); ax.yaxis.set_ticklabels(['Legit', 'Fraud']);


# In[27]:


import seaborn as sns
ax= plt.subplot()
sns.heatmap(pd.DataFrame(classificationReport).iloc[:-1, :].T, annot=True)
ax.set_title('Random Forest Classifier - Calssification Report'); 


# In[28]:


estimator = forest.estimators_[5]
from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names=X_train.columns,class_names=['0','1'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)
g = pydotplus.graph_from_dot_file(path="tree.dot")
display.display(Image(g.create_png()))


# In[29]:


plt.figure(figsize=(9,7))
n_features = X_train.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()


# In[30]:


storeMetrics('Random Forest', train_accuracy, test_accuracy)


# ### **3. Multilayer Perceptrons (MLPs) :**

# In[31]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100,100,100]))
mlp.fit(X_train, y_train)


# In[32]:


y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)


# In[33]:


train_accuracy = accuracy_score(y_train,y_train_mlp)
test_accuracy = accuracy_score(y_test,y_test_mlp)

print("Multilayer Perceptrons:")
print("Accuracy on training Data: {}".format(train_accuracy))
print("Accuracy on test Data: {}".format(test_accuracy))


# In[34]:


#Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_test_mlp)
classificationReport = classification_report(y_test, y_test_mlp, labels=mlp.classes_,
                                   target_names=mlp.classes_,
                                   output_dict=True)


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted label');ax.set_ylabel('True label'); 
ax.set_title('MultiLayer Perceptrons - Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Legit', 'Fraud']); ax.yaxis.set_ticklabels(['Legit', 'Fraud']);


# In[36]:


import seaborn as sns
ax= plt.subplot()
sns.heatmap(pd.DataFrame(classificationReport).iloc[:-1, :].T, annot=True)
ax.set_title('MultiLayer Perceptrons - Calssification Report'); 


# In[37]:


storeMetrics('Multilayer Perceptrons', train_accuracy, test_accuracy)


# ### **4. XGBoost Classifier**

# In[38]:


from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate=0.4,max_depth=7)
xgb.fit(X_train, y_train)


# In[39]:


y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)


# In[40]:


train_accuracy = accuracy_score(y_train,y_train_xgb)
test_accuracy = accuracy_score(y_test,y_test_xgb)

print("XGBoost:")
print("Accuracy on training Data: {}".format(train_accuracy))
print("Accuracy on test Data: {}".format(test_accuracy))


# In[41]:


#Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_test_xgb)
classificationReport = classification_report(y_test, y_test_xgb, labels=xgb.classes_,
                                   target_names=xgb.classes_,
                                   output_dict=True)


# In[42]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted label');ax.set_ylabel('True label'); 
ax.set_title('XGBoost Classifier - Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Legit', 'Fraud']); ax.yaxis.set_ticklabels(['Legit', 'Fraud']);


# In[43]:


import seaborn as sns
ax= plt.subplot()
sns.heatmap(pd.DataFrame(classificationReport).iloc[:-1, :].T, annot=True)
ax.set_title('XG Boost Classifier - Calssification Report'); 


# In[44]:


storeMetrics('XGBoost', train_accuracy, test_accuracy)


# ### **5. Support Vector Machines**

# In[45]:


from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=12)
svm.fit(X_train, y_train)


# In[46]:


y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)


# In[47]:


train_accuracy = accuracy_score(y_train,y_train_svm)
test_accuracy = accuracy_score(y_test,y_test_svm)

print("SVM:")
print("Accuracy on training Data: {}".format(train_accuracy))
print("Accuracy on test Data: {}".format(test_accuracy))


# In[48]:


#Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
confusionMatrix = confusion_matrix(y_test, y_test_svm)
classificationReport = classification_report(y_test, y_test_svm, labels=svm.classes_,
                                   target_names=svm.classes_,
                                   output_dict=True)


# In[49]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted label');ax.set_ylabel('True label'); 
ax.set_title('Support Vector Machines - Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Legit', 'Fraud']); ax.yaxis.set_ticklabels(['Legit', 'Fraud']);


# In[50]:


import seaborn as sns
ax= plt.subplot()
sns.heatmap(pd.DataFrame(classificationReport).iloc[:-1, :].T, annot=True)
ax.set_title('Support Vector Machines - Calssification Report'); 


# In[51]:


storeMetrics('SVM', train_accuracy, test_accuracy)


# ## **Step 5. Models Comparison**

# In[52]:


#creating dataframe
results = pd.DataFrame({ 'ML Model': Models,    
    'Train Accuracy': traningSetResult,
    'Test Accuracy': testSetReslt})
results


# In[53]:


results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)


# In[54]:


import plotly.express as px


# In[55]:


fig = px.bar(
    x = Models,
    y = testSetReslt,
    color = Models,
    labels = {'x': "Model", 'y': "Accuracy"},
    title = "Model Accuracy Comparision"
)
fig.show()


# In[56]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# In[57]:


decisionTree = DecisionTreeClassifier(random_state=29)


# In[58]:


# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}


# In[59]:


grid_search = GridSearchCV(estimator=decisionTree, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")


# In[60]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)')


# In[61]:


grid_search.best_estimator_


# In[62]:


decisionTree_Best = grid_search.best_estimator_


# In[63]:


train_accuracy = accuracy_score(y_train,decisionTree_Best.predict(X_train))
test_accuracy = accuracy_score(y_test,decisionTree_Best.predict(X_test))
print("Decision Tree with HyperParameter Tuning:")
print("Accuracy on training Data: {}".format(train_accuracy))
print("Accuracy on test Data: {}".format(test_accuracy))


# In[64]:


#Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
confusionMatrix = confusion_matrix(y_test, decisionTree_Best.predict(X_test))
classificationReport = classification_report(y_test,  decisionTree_Best.predict(X_test), labels=decisionTree_Best.classes_,
                                   target_names=decisionTree_Best.classes_,
                                   output_dict=True)


# In[65]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(confusionMatrix, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted label');ax.set_ylabel('True label'); 
ax.set_title('Decision Tree with HyperParameter Tuning - Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Legit', 'Fraud']); ax.yaxis.set_ticklabels(['Legit', 'Fraud']);


# In[66]:


import seaborn as sns
ax= plt.subplot()
sns.heatmap(pd.DataFrame(classificationReport).iloc[:-1, :].T, annot=True)
ax.set_title('Decision Tree - Calssification Report'); 


# In[67]:


storeMetrics('Decision Tree HPT', train_accuracy, test_accuracy)


# In[68]:


fig = px.bar(
    x = Models,
    y = testSetReslt,
    color = Models,
    labels = {'x': "Model", 'y': "Accuracy"},
    title = "Model Accuracy Comparision"
)
fig.show()


# ## **9. References**
# 1. https://en.wikipedia.org/wiki/Phishing
# 2. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0258361
# 3. https://www.phishtank.com/developer_info.php
# 4. https://www.unb.ca/cic/datasets/url-2016.html

# In[ ]:




