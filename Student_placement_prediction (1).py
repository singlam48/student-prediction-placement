#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[3]:


dataset = pd.read_csv("Placement_Data_Full_Class.csv")


# In[4]:


dataset.head()


# In[5]:


# as salary and sl_no columns are not required for placement status prediction so we drop it
dataset.drop(['salary','sl_no'], axis=1, inplace=True)


# In[6]:


# missing values checking
dataset.isnull().sum()


# In[7]:


# checking column values data type
dataset.info()


# # Label Encoding Data

# In[8]:


# label encoding needs to be done to ensure all values in the dataset is numeric
# hsc_s, degree_t columns needs to be splitted into columns (get_dummies needs to be applied)
features_to_split = ['hsc_s','degree_t']
for feature in features_to_split:
    dummy = pd.get_dummies(dataset[feature])
    dataset = pd.concat([dataset, dummy], axis=1)
    dataset.drop(feature, axis=1, inplace=True)


# In[9]:


dataset


# In[10]:


dataset.rename(columns={"Others": "Other_Degree"},inplace=True)


# In[11]:


dataset


# In[12]:


encoder = LabelEncoder() # to encode string to the values like 0,1,2 etc.


# In[13]:


columns_to_encode = ['gender','ssc_b', 'hsc_b','workex','specialisation','status']
for column in columns_to_encode:
    dataset[column] = encoder.fit_transform(dataset[column])


# In[14]:


dataset


# In[15]:


dataset.describe()


# # Checking for Outliers

# In[16]:


fig, axs = plt.subplots(ncols=6,nrows=3,figsize=(20,10))
index = 0
axs = axs.flatten()
for k,v in dataset.items():
    sns.boxplot(y=v, ax=axs[index])
    index+=1

fig.delaxes(axs[index])
plt.tight_layout(pad=0.3, w_pad=0.5,h_pad = 4.5) # for styling by giving padding


# In[17]:


# deleting some outliers in 2 columns degree_p and hsc_p
dataset = dataset[~(dataset['degree_p']>=90)]
dataset = dataset[~(dataset['hsc_p']>=95)]


# # Checking for Correlation

# In[18]:


dataset.corr()


# In[19]:


# heatmap for checking correlation or linearity

plt.figure(figsize=(20,10))
sns.heatmap(dataset.corr().abs(), annot=True)


# In[20]:


dataset.shape


# In[21]:


# checking distributions of all features
fig, axs = plt.subplots(ncols=6,nrows=3,figsize=(20,10))
index = 0
axs = axs.flatten()
for k,v in dataset.items():
    sns.distplot(v, ax=axs[index])
    index+=1

fig.delaxes(axs[index]) # deleting the 18th figure
plt.tight_layout(pad=0.3, w_pad=0.2,h_pad = 4.5)


# In[22]:


x = dataset.loc[:,dataset.columns!='status'] # all features are used
y = dataset.loc[:, 'status'] # label is status of placement


# In[23]:


x


# In[24]:


y


# In[25]:


sc= StandardScaler()
x_scaled = sc.fit_transform(x) # for standardising the features
x_scaled = pd.DataFrame(x_scaled)


# In[26]:


x_train,x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.18, random_state=0)


# # Using Logistic Regression

# In[27]:


lr = LogisticRegression()


# In[28]:


lr.fit(x_train, y_train)


# In[29]:


y_pred = lr.predict(x_test)


# In[30]:


y_pred


# In[31]:


y_test


# In[32]:


accuracy_score(y_test, y_pred)


# In[33]:


lr.score(x_train,y_train)


# In[34]:


confusion_matrix(y_test, y_pred)


# In[35]:


print(classification_report(y_test,y_pred))


# # Using Naive Bayes Classifier - Gaussian Naive Bayes

# In[36]:


nbclassifier = GaussianNB()


# In[37]:


nbclassifier.fit(x_train, y_train)


# In[38]:


y_pred_nb = nbclassifier.predict(x_test)


# In[39]:


accuracy_score(y_test, y_pred_nb)


# In[40]:


nbclassifier.score(x_train, y_train)


# In[41]:


confusion_matrix(y_test, y_pred_nb)


# # Using SVM Linear Kernel

# In[42]:


clf = svm.SVC(kernel="linear")


# In[43]:


clf.fit(x_train, y_train)


# In[44]:


y_pred_svm = clf.predict(x_test)


# In[45]:


accuracy_score(y_test, y_pred_svm)


# In[46]:


clf.score(x_train, y_train)


# In[47]:


confusion_matrix(y_test, y_pred_svm)


# In[48]:


print(classification_report(y_test, y_pred_svm))


# # So, Naive Bayes was better for not overfitting the data
# # SVM gave better accuracy with least difference in score.
# #So, Our final model would use SVM for Student Placement Prediction.
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




