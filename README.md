<<<<<<< HEAD
# TCC

=======


```python
import pandas as pd
import numpy as np


data = pd.read_csv('train_values.csv')
target=pd.read_csv('train_labels.csv')
submit=pd.read_csv('test_values.csv')
data.head()
```

Essa parte escala as features tanto do training set tanto quanto do test set.( Sim, eu sei que da para fazer uma função.)


```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
encoder=OneHotEncoder(categories='auto',sparse=False)
#x=data['sex'].values
#x = x.reshape(len(x), 1)
#x = encoder.fit_transform(x)
#data["Female"] = x[:,0]
#data["Male"] = x[:,1]
#data=data.drop(columns= ['sex'])
#data.head(15)

scaler = MinMaxScaler()
t=data['resting_blood_pressure'].values
t = t.reshape(len(t), 1)
t=scaler.fit_transform(t)
data['resting_blood_pressure']=t[:,0]

scaler = MinMaxScaler()
t=data['serum_cholesterol_mg_per_dl'].values
t = t.reshape(len(t), 1)
t=scaler.fit_transform(t)
data['serum_cholesterol_mg_per_dl']=t[:,0]

scaler = MinMaxScaler()
t=data['oldpeak_eq_st_depression'].values
t = t.reshape(len(t), 1)
t=scaler.fit_transform(t)
data['oldpeak_eq_st_depression']=t[:,0]

scaler = MinMaxScaler()
t=data['age'].values
t = t.reshape(len(t), 1)
t=scaler.fit_transform(t)
data['age']=t[:,0]

scaler = MinMaxScaler()
t=data['max_heart_rate_achieved'].values
t = t.reshape(len(t), 1)
t=scaler.fit_transform(t)
data['max_heart_rate_achieved']=t[:,0]


#z=submit['sex'].values
#z = z.reshape(len(z), 1)
#z = encoder.fit_transform(z)
#submit["Female"] = z[:,0]
#submit["Male"] = z[:,1]
#submit=submit.drop(columns= ['sex'])
data.head(15)

scaler = MinMaxScaler()
t=submit['resting_blood_pressure'].values
t = t.reshape(len(t), 1)
t=scaler.fit_transform(t)
submit['resting_blood_pressure']=t[:,0]

scaler = MinMaxScaler()
t=submit['serum_cholesterol_mg_per_dl'].values
t = t.reshape(len(t), 1)
t=scaler.fit_transform(t)
submit['serum_cholesterol_mg_per_dl']=t[:,0]

scaler = MinMaxScaler()
t=submit['oldpeak_eq_st_depression'].values
t = t.reshape(len(t), 1)
t=scaler.fit_transform(t)
submit['oldpeak_eq_st_depression']=t[:,0]

scaler = MinMaxScaler()
t=submit['age'].values
t = t.reshape(len(t), 1)
t=scaler.fit_transform(t)
submit['age']=t[:,0]

scaler = MinMaxScaler()
t=submit['max_heart_rate_achieved'].values
t = t.reshape(len(t), 1)
t=scaler.fit_transform(t)
submit['max_heart_rate_achieved']=t[:,0]

#submit
data.head(15)
```

Essa parte transforma a variável categorica em ordinal


```python
x=data['thal'].values
x = x.reshape(len(x), 1)
x = encoder.fit_transform(x)
data["thal_normal"] = x[:,0]
data["thal_reversible_defect"] = x[:,1]
data["thal_fixed_defect"] = x[:,2]

data=data.drop(columns= ['thal'])
data=data.drop(columns= ['patient_id'])

z=submit['thal'].values
z = z.reshape(len(z), 1)
z = encoder.fit_transform(z)
submit["thal_normal"] = z[:,0]
submit["thal_reversible_defect"] = z[:,1]
submit["thal_fixed_defect"] = z[:,2]

submit=submit.drop(columns= ['thal'])
submit=submit.drop(columns= ['patient_id'])
```

Essa transforma as feature em array e separa a base em treino e test


```python
x = data.iloc[:,0:18].values
y = target.iloc[:, 1].values
z = submit.iloc[:,0:18].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

```

Essa seleciona as melhores features do training set


```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1)


feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)


feat_selector.fit(x_train, y_train)


X_filtered = feat_selector.transform(x_train)
```

Seta o campo dos parametros da Ramdon forest para a otimizacao randomica


```python
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

min_samples_split = [2,3,4, 5, 10]

min_samples_leaf = [1, 2, 4,6]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)
```

Utiliza o grid gerada para a busca randomica dos melhores parâmetros.


```python

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 1000, cv = 5, verbose=2, random_state=12, n_jobs = -1)

rf_random.fit(X_filtered, y_train)
```

printa os melhores parametros encontrados.


```python
rf_random.best_params_
```

Por fim, utilizamos os parametros anteriores, aplicamos Ramdomforest sobre o dataset


```python
rf =RandomForestRegressor(
                       n_estimators=1200,
 min_samples_split= 5,
 min_samples_leaf= 1,
 max_features='sqrt',
 max_depth=80,
 bootstrap =True)
rf.fit(X_filtered, y_train)

y_pred=rf.predict(feat_selector.transform(x_test))
```

Utilizando log-loss como medida, estimamos o erro de nosso modelo



```python
from sklearn.metrics import log_loss

log_loss(y_test, y_pred)
```
>>>>>>> fcbaf063af8e048853e98d3b9217aa3bf7ca2cca
