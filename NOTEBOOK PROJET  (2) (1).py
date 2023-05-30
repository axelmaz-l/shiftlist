#!/usr/bin/env python
# coding: utf-8

# In[49]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from random import *
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import anderson
from sklearn.impute import KNNImputer
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[50]:


np.random.seed(42)


# **ANALYSE EXPLORATOIRE DES DONNEES**.
# 
# 
# Cette partie du code sera consacrée à l'analyse descriptive de notre jeu de données, suivi d'une analyse plus poussée de la variable cible,nommée "Sale Price", contenue dans le dataframe df_train.

# In[51]:


# Chargement de nos deux jeux de données
df_train=pd.read_csv(r"C:\Users\axel\Downloads\train.csv")
df_test=pd.read_csv((r"C:\Users\axel\Downloads\test.csv"))
df_train.head()


# In[52]:


# taille des jeux de données test et train
print(df_train.shape,df_test.shape)


# In[85]:


y=df["SalePrice"].values


# In[53]:


#concaténation de nos deux dataframes
df_add=pd.concat([df_train.drop("SalePrice",axis=1),df_test],axis=0,ignore_index=True)


# In[54]:


# Nombre de variables qualitatives et quantitatives
df_add.dtypes.value_counts()


# In[55]:


df_add.info()


# In[56]:


# pourcentage de valeurs manquantes dans nos variables
(df_add.isna().sum()/df_add.shape[0]).sort_values(ascending=False)


# Analyse de la variable "SalePrice"

# In[57]:


# valeurs statistiques de la variable "target"
df_train["SalePrice"].describe()


# In[58]:


#représentation graphique de "SalePrice"
sns.displot(data=df_train,x="SalePrice",kde=True)


# In[59]:


#corrélation des variables explicatives par rapport à la variable cible "SalePrice"
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df_train.corr()[['SalePrice']].sort_values(by='SalePrice', ascending=False), vmin=-1, vmax=1, annot=True)


# In[60]:


# Nuage de points représentant "Sale Price" en fonction des variables dont la corrélation avec "Sale price" est > 0.5 (1)
sns.pairplot(data=df_train,x_vars=["TotalBsmtSF","1stFlrSF","FullBath","TotRmsAbvGrd"],y_vars="SalePrice")


# In[61]:


# Nuage de points représentant "Sale Price" en fonction des variables dont la corrélation avec "Sale price" est > 0.5 (2)
sns.pairplot(data=df_train,x_vars=["OverallQual","GrLivArea","GarageCars","GarageArea"],y_vars="SalePrice")


# # II) Preprocessing

# In[62]:


#II.1) Suppression de la colonne des colonnes au taux de valeurs manquantes > 80% et des valeurs aberrantes
df_add=df_add[df_add.columns[df_add.isna().sum()/df_add.shape[0]<0.8]]
df_add.drop('Id',axis=1)


# Imputation des données manquantes pour les variables numériques

# In[63]:


#Imputation des données manquantes pour les variables numériques
df_add.select_dtypes('float').isnull().sum()


# In[64]:


df_add["GarageYrBlt"].describe()


# In[65]:


sns.displot(data=df_add,x="GarageYrBlt")


# In[66]:



df_add['GarageYrBlt'].fillna(2207,inplace=True)

#Imputation des données manquantes de la variable "GarageYrBlt"
# In[67]:


#valeurs manquantes dans les variables types objet
df_add.select_dtypes('float').isnull().sum()


# In[91]:


#corrélation des variables explicatives par rapport à la variable cible "LotFrontage"
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df_train.corr()[['LotFrontage']].sort_values(by='LotFrontage', ascending=False), vmin=-1, vmax=1, annot=True)


# In[68]:


#imputation spécifique de la variable ('LotFrontage')
imputer = KNNImputer(n_neighbors=5)
imputer.fit(df_add['LotFrontage'].values.reshape(-1, 1))
df_add['LotFrontage']= imputer.transform(df_add['LotFrontage'].values.reshape(-1, 1))
    
    


# In[ ]:





# In[69]:


#imputation des valeurs manquantes pour les variables de type float
for col in df_add.select_dtypes('float'):
    df_add[col].fillna(0.0,inplace=True)


# In[70]:


#imputation variables de type object
for col in df_add.select_dtypes('object'):
    df_add[col].fillna('None',inplace=True)


# In[71]:


df_add.info()


# In[72]:


df_add.shape


# In[ ]:





# #II.2) Encodage des variables qualititatives 

# In[73]:


# Encodage des variables nominales
Ordinal_features=['ExterQual','ExterCond','LotShape','BsmtQual','BsmtCond',
              'BsmtExposure','BsmtFinType1', 'BsmtFinType2','HeatingQC',
              'Functional','FireplaceQu','KitchenQual', 'GarageFinish',
              'GarageQual','GarageCond',]
values_ordered=[
    #ExterQual
    ['Po','Fa','TA','Gd','Ex'],
    #ExterCond
    ['Po','Fa','TA','Gd','Ex'],
    #LotShape
    ['Reg','IR1','IR2','IR3'],
    #BsmtQual
    ['None','Fa','TA','Gd','Ex'],
    #BsmtCond
    ['None','Po','Fa','TA','Gd','Ex'],
    #BsmtExposure
    ['None','No','Mn','Av','Gd'],
    #BsmtFinType1
    ['None','Unf','LwQ', 'Rec','BLQ','ALQ' , 'GLQ' ],
    #BsmtFinType2
   ['None','Unf','LwQ', 'Rec','BLQ','ALQ' , 'GLQ' ],
    #HeatingQC
    ['Po','Fa','TA','Gd','Ex'],
    #Functional
   ['None','Sev','Maj2','Maj1','Mod','Min2','Min1','Typ'],
    #FireplaceQu 
    ['None','Po','Fa','TA','Gd','Ex'],
    #KitchenQual
    ['None','Fa','TA','Gd','Ex'],
    #GarageFinish
    ['None','Unf','RFn','Fin'],
    #GarageQual
    ['None','Po','Fa','TA','Gd','Ex'],
    #GarageCond
    ['None','Po','Fa','TA','Gd','Ex'],
]


# In[74]:


for i in range(len(values_ordered)):
    ord = OrdinalEncoder(categories = {0:values_ordered[i]})
    df_add.loc[:,Ordinal_features[i]] = ord.fit_transform(df_add.loc[:,Ordinal_features[i]].values.reshape(-1,1))


# In[75]:


# encodage des variables qualitatives nominales
df_add=pd.concat([df_add,pd.get_dummies(df_add.select_dtypes('object'))],axis=1)


# In[76]:


df_add.head()


# In[77]:


# suppression des variables objets
df_add=df_add.drop(df_add.select_dtypes('object'),axis=1)


# In[78]:


df_add.shape


# In[79]:


# Normalisation de nos variables
from sklearn.preprocessing import RobustScaler
robust=RobustScaler()
df_scaled=pd.DataFrame(robust.fit_transform(df_add),columns=df_add.columns)


# In[80]:


df_scaled.head()


# In[93]:


df_scaled.drop("Id",axis=1)


# # III) IMPLEMENTATION D'ALGORITHMES DE MACHINE LEARNING

# In[81]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error


# In[82]:


X=df_scaled[:df_train.shape[0]]
values_tested=df_scaled[df_train.shape[0]:]


# In[86]:


#séparation du jeu de données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[87]:


print(X.shape,y.shape)


# In[88]:


#Régression Lasso
model=Lasso()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[89]:


print( "train acc",model.score(X_train, y_train))
print("test acc ", model.score(X_test, y_test))


# In[ ]:


# Le résultat du dessus est la précision du modèle il y a d'autres modèles que l'on peut implémenter . Je me charge de finaliser le code en faisant une soumission sur kaggle. Le code est terminéà 98%%.


# In[ ]:


#Régression Ridge


# In[94]:


model=Lasso()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# In[95]:


from sklearn.ensemble import RandomForestRegressor


# In[97]:


model_rf=RandomForestRegressor(random_state=3).fit(X_train,y_train)
y_pred=model_rf.predict(X_train)


# In[98]:


from sklearn.metrics import r2_score
r2_s=r2_score(y_train,y_pred)
print(r2_s)


# In[ ]:




