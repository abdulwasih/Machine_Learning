
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


import numpy as np


# In[4]:


from sklearn.tree import DecisionTreeClassifier


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


from sklearn.linear_model import LogisticRegression


# In[7]:


from sklearn.ensemble import RandomForestClassifier


# In[8]:


from sklearn.svm import SVC


# In[9]:


#importing training data
data = pd.read_csv("C:\\Users\\user\\Desktop\\samples\\train.csv")


# In[10]:


#shape of the data
data.shape


# In[11]:


#attributes/features
X=data.iloc[:,1:]


# In[12]:


#labels
y=data.iloc[:,0]


# In[13]:


X.shape


# In[14]:


y.shape


# In[15]:


#split data into training and testing
train_img, test_img, train_lbl, test_lbl = train_test_split(X, y, test_size=0.2, random_state=0)


# In[16]:


#decision tree model
decTree = DecisionTreeClassifier()


# In[17]:


#model fitting
decTree.fit(train_img,train_lbl)


# In[18]:


#logistic regression model
logisticRegr = LogisticRegression(C=1000,solver = 'lbfgs')


# In[19]:


#model fitting
logisticRegr.fit(train_img, train_lbl)


# In[282]:


#random forest model
randomForest = RandomForestClassifier(n_estimators=1000,max_depth=3,random_state=0)


# In[283]:


#model fitting
randomForest.fit(train_img,train_lbl)


# In[ ]:


#SVM model
svmclf = SVC(gamma='auto')


# In[ ]:


#model fitting
svmclf.fit(train_img,train_lbl)


# In[26]:


#test image
xc=test_img.iloc[650]


# In[27]:


#lablel corresponding to test image
yc=test_lbl.iloc[650]


# In[28]:


plottable_image = np.reshape(xc.values, (28, 28))


# In[29]:


plt.imshow(plottable_image, cmap='gray_r')
plt.title('Digit Label: {}'.format(yc))
plt.show()


# In[48]:


predictionDecTree = decTree.predict([test_img.iloc[650]])


# In[49]:


print("The predicted number using decision tree is {}".format(predictionDecTree[0]))


# In[50]:


predictionTreeTest = decTree.score(test_img,test_lbl)


# In[51]:


print("The score of Decision tree (Testing) is {}".format(predictionTreeTest))


# In[52]:


predictionTreeTrain = decTree.score(train_img,train_lbl)


# In[53]:


print("The score of Decision tree(Training) is {}".format(predictionTreeTrain))


# In[54]:


predictionLogReg=logisticRegr.predict([test_img.iloc[650]])


# In[55]:


print("The predicted number using logistic regression is {}".format(predictionLogReg[0]))


# In[56]:


predictionLogRegTest= logisticRegr.score(test_img,test_lbl)


# In[60]:


print("The score of Logistic Regression (Testing) is {}".format(predictionLogRegTest))


# In[61]:


predictionLogRegTrain= logisticRegr.score(train_img,train_lbl)


# In[62]:


print("The score of Logistic Regression (Training) is {}".format(predictionLogRegTrain))


# In[ ]:


predictionRandForest = randomForest.predict([test_img.iloc[650]])


# In[ ]:


print("The predicted number using Random Forest is {}".format(predictionRandForest))


# In[ ]:


predictionRandForestTest= randomForest.score(test_img,test_lbl)


# In[ ]:


print("The score of Random Forest (Testing) is {}".format(predictionRandForestTest))


# In[ ]:


predictionRandForestTrain= randomForest.score(train_img,train_lbl)


# In[ ]:


print("The score of Random Forest (Training) is {}".format(predictionRandForestTrain)


# In[ ]:


predictionsvm=svmclf.predict([test_img.iloc[650]])


# In[ ]:


print("The predicted number using SVM is {}".format(predictionsvm))


# In[ ]:


predictionsvmTest= svmclf.score(test_img,test_lbl)


# In[ ]:


print("The score of SVM (Testing) is {}".format(predictionsvmTest))


# In[ ]:


predictionsvmTrain= svmclf.score(train_img,train_lbl)


# In[ ]:


print("The score of SVM (Training) is {}".format(predictionsvmTrain))

