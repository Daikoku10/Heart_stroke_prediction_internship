#!/usr/bin/env python
# coding: utf-8

# # Heart Stroke Prediction
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ### Here we will be experimenting with 2 algorithms
# 
# 1.KNeighborsClassifier
# 2.RandomForestClassifier

# In[2]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[3]:


df = pd.read_csv('datasetheart.csv')


# In[4]:


df.info()


# In[5]:


df.describe()


# ## Feature Selection

# In[27]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="PuBu")


# In[25]:


df.hist(color="orange",edgecolor="red")
plt.rcParams['figure.figsize']=(12,10)


# In[28]:


sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdPu',edgecolor="darkblue")
plt.rcParams['figure.figsize']=(5,3)


# ### Data Processing

# In[12]:


dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[14]:


dataset.head()


# In[15]:


y = dataset['target']
X = dataset.drop(['target'], axis = 1)


# In[16]:


from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())


# In[18]:


plt.plot([k for k in range(1,21)], knn_scores, color = 'darkblue')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.rcParams['figure.figsize']=(21,14)


# In[19]:


knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,X,y,cv=10)


# In[20]:


score.mean()


# ### Random Forest Classifier

# In[21]:


from sklearn.ensemble import RandomForestClassifier


# In[22]:


randomforest_classifier= RandomForestClassifier(n_estimators=10)
score=cross_val_score(randomforest_classifier,X,y,cv=10)


# In[23]:


score.mean()


# In[ ]:




