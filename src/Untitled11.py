#!/usr/bin/env python
# coding: utf-8

# In[62]:
 #pip install pandas, numpy,sklearn, xgboost, seaborn

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import log_loss
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.ensemble import (AdaBoostClassifier,GradientBoostingClassifier)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[2]:


data = pd.read_csv('train_values.csv')
target=pd.read_csv('train_labels.csv')
submit=pd.read_csv('test_values.csv')
data.head()


# In[3]:


scaler = MinMaxScaler()

features=['resting_blood_pressure','serum_cholesterol_mg_per_dl','oldpeak_eq_st_depression','age','max_heart_rate_achieved']

def Scaler(features):
    for feature in features:
        t=data[feature].values
        t = t.reshape(len(t), 1)
        t=scaler.fit_transform(t)
        data[feature]=t[:,0]
        
def ScalerSub(features):
    for feature in features:
        t=submit[feature].values
        t = t.reshape(len(t), 1)
        t=scaler.fit_transform(t)
        submit[feature]=t[:,0]        
        
        
Scaler(features)
ScalerSub(features)


# In[4]:


encoder=OneHotEncoder(categories='auto',sparse=False)


x=data['sex'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["Male"] = x[:,0]
data["Female"] = x[:,1]

x=data['resting_ekg_results'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["tipo_0"] = x[:,0]
data["tipo_1"] = x[:,1]
data["tipo_2"] = x[:,2]

x=data['thal'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["thal_normal"] = x[:,0]
data["thal_reversible_defect"] = x[:,1]
data["thal_fixed_defect"] = x[:,2]

x=data['chest_pain_type'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["type_1"] = x[:,0]
data["type_2"] = x[:,1]
data["type_3"] = x[:,2]
data['type_4'] = x[:,3]


x=data['slope_of_peak_exercise_st_segment'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["slope_type_1"] = x[:,0]
data["slope_type_2"] = x[:,1]
data["slope_type_3"] = x[:,2]

data=data.drop(columns=['resting_ekg_results'])
data=data.drop(columns= ['sex'])
data=data.drop(columns= ['thal'])
data=data.drop(columns= ['patient_id'])
data=data.drop(columns= ['chest_pain_type'])
data=data.drop(columns=['slope_of_peak_exercise_st_segment'])


z=submit['sex'].values
z = z.reshape(len(z), 1)
z= encoder.fit_transform(z)
submit["Male"] = z[:,0]
submit["Female"] = z[:,1]


z=submit['thal'].values
z = z.reshape(len(z), 1)
z= encoder.fit_transform(z)
submit["thal_normal"] = z[:,0]
submit["thal_reversible_defect"] = z[:,1]
submit["thal_fixed_defect"] = z[:,2]

z=submit['chest_pain_type'].values
z = z.reshape(len(z), 1)
z = encoder.fit_transform(z)
submit["type_1"] = z[:,0]
submit["type_2"] = z[:,1]
submit["type_3"] = z[:,2]
submit['type_4'] = z[:,3]


z=submit['slope_of_peak_exercise_st_segment'].values
z= z.reshape(len(z), 1)
z = encoder.fit_transform(z)
submit["slope_type_1"] = z[:,0]
submit["slope_type_2"] = z[:,1]
submit["slope_type_3"] = z[:,2]


submit=submit.drop(columns= ['sex'])
submit=submit.drop(columns= ['thal'])
submit=submit.drop(columns= ['patient_id'])
submit=submit.drop(columns= ['chest_pain_type'])
submit=submit.drop(columns=['slope_of_peak_exercise_st_segment'])


# In[5]:


feat = [ "var", "median", "mean", "std", "max", "skew"]
for i in feat:
    data[i] = data.aggregate(i,  axis =1)


# In[6]:


target=target.drop(columns='patient_id')


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.25, random_state = 12)


# In[10]:


from sklearn.ensemble import ExtraTreesClassifier

TOP_FEATURES = 15

forest = ExtraTreesClassifier(n_estimators=250, max_depth=5, random_state=1)
forest.fit(x_train, y_train)

importances = forest.feature_importances_
std = np.std(
    [tree.feature_importances_ for tree in forest.estimators_],
    axis=0
)
indices = np.argsort(importances)[::-1]
indices = indices[:TOP_FEATURES]

print('Top features:')
for f in range(TOP_FEATURES):
    print('%d. feature %d (%f)' % (f + 1, indices[f], importances[indices[f]]))
data.head()


# In[11]:


feat=['type_4','thal_fixed_defect','thal_reversible_defect','exercise_induced_angina','num_major_vessels','slope_type_1' ,'slope_type_2','oldpeak_eq_st_depression','median']
data=data[feat]
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.25, random_state = 12)


# In[12]:


from sklearn.linear_model import LogisticRegression

l = LogisticRegression(random_state=12,
    penalty='l1',
    C=0.1
)
l.fit(x_train, y_train)
y_pred=l.predict_proba(x_test)


# In[13]:


print(log_loss(y_test, np.clip(y_pred,0.025,0.975)))


# In[14]:


m = XGBClassifier(random_state=12,
    max_depth=2,
    gamma=2,
    eta=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5
)
m.fit(x_train,y_train)
y_pred=m.predict_proba(x_test)


# In[15]:


print(log_loss(y_test, np.clip(y_pred,0.025,0.975)))


# In[16]:


from sklearn.feature_selection import RFE
labels=[]
rfe = RFE(XGBClassifier(n_jobs=-1, random_state=12))

rfe.fit(x_train, y_train)

print('Selected features:')
print(rfe.support_)


# In[63]:


models = [
    LogisticRegression(),
    XGBClassifier(max_depth=2),
    RandomForestClassifier(max_depth=2, random_state=12)
]

preds = pd.DataFrame()
for i, m in enumerate(models):
    m.fit(x_train, y_train),
    preds[i] = m.predict_proba(x_test)[:,1]

weights = [1, 0.3,0.4]
preds['weighted_pred'] = (preds * weights).sum(axis=1) / sum(weights)
a=preds['weighted_pred']
preds.head


# In[64]:


print(log_loss(y_test, np.clip(a,0.025,0.975)))


# In[65]:


from mlxtend.classifier import StackingClassifier
preds = pd.DataFrame()

m = StackingClassifier(
    classifiers=[
        LogisticRegression(),
        RandomForestClassifier(max_depth=2,random_state=12),
        XGBClassifier(max_depth=2)
    ],
    use_probas=True,
    meta_classifier=GradientBoostingClassifier(random_state=12)
)

m.fit(x_train, y_train)
preds['stack_pred'] = m.predict_proba(x_test)[:,1]

a=preds['stack_pred']
a.head()


# In[66]:


print(log_loss(y_test, np.clip(a,0.025,0.975)))


# In[67]:


a.head(50)


# In[ ]:




