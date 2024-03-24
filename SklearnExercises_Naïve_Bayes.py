###DATASET Y MANIPULACION
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
###NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
###METRICAS 
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


# fetch dataset 
diabetes = fetch_ucirepo(id=891) 

X = diabetes.data.features 
y = diabetes.data.targets 
###normalizamos los datos
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_n = scaler.fit_transform(X)
X = pd.DataFrame(X_n, columns=X.columns)
X = X.drop(columns=['AnyHealthcare','NoDocbcCost','Sex','Fruits','Veggies'])
########FILTRADO PARA IGUALAR TARJETS#######


### CORRELACIONES
data = pd.concat([X, y], axis=1)  # Combina 'X' y 'y' en un solo DataFrame

# Calcula las correlaciones entre todas las características y 'y'
correlations = data.corrwith(data['Diabetes_binary'])

# Convierte las correlaciones a valores absolutos y ordénalas de manera descendente
correlations = correlations.abs().sort_values(ascending=False)

# Muestra las características más correlacionadas con 'y'


### PARAMETROS A VALIDAR
param_grid = {
    'priors': [None, [0.25, 0.75], [0.4, 0.6], [0.3, 0.7]],  
    'var_smoothing': [1e-9, 1e-8, 1e-7, 5e-8],  
}

#KFOLD
rd = 10
particiones = 10
skf = StratifiedKFold(n_splits=particiones,shuffle=True,random_state=rd)

#A continuación hacemos cada subdivisión. 
i=1
ACCM=[]
PRM=[]
FALLM=[]
RCM=[]
F1M=[]

AUCM=[]
for train, test in skf.split(X,y):
  
  X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
  ###DIVISION DE TRAINING EN VALIDACION TRAINING
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=rd)
  ###VALIDACION DE HIPERPARAMETROS
  y_val=np.ravel(y_val)
  acc=0
  for prior in param_grid['priors']:
      for varsm in param_grid['var_smoothing']:
          
          gnb1=GaussianNB(priors=prior,var_smoothing=varsm)
          gnb1.fit(X_train, y_train)
          y_val_prob = gnb1.predict_proba(X_val)
          auc1 = metrics.roc_auc_score(y_val, y_val_prob[:,1]) 
          
          if auc1>acc:
              gnb=gnb1
      
  ### NOS QUEDAMOS CON EL GNB CON LA MEJOR AUC
  y_train=np.ravel(y_train)
  gnb.fit(X_train, y_train)
  y_pred = gnb.predict(X_test)
  
  
  ### PERFORMANCE EVALUATION
  conf_matrix = confusion_matrix(y_test, y_pred)
  TN = conf_matrix[0, 0]  # True Negatives
  FP = conf_matrix[0, 1]  # False Positives
  FN = conf_matrix[1, 0]  # False Negatives
  TP = conf_matrix[1, 1]  # True Positives
  #Accuracy
  acc_score = accuracy_score(y_test, y_pred)
  #precision
  precision = TP / (TP + FP)
  #Fallout
  fallout = FP / (FP + TN)
  #Recall
  recall = recall_score(y_test, y_pred)
  #f1
  f1 = f1_score(y_test, y_pred)
  #Matriz de confusion
  metrics.ConfusionMatrixDisplay.from_estimator(gnb, X_test, y_test,cmap=plt.cm.Blues)
  #pintamos la curva
  y_prob = gnb.predict_proba(X_test)
  auc = metrics.roc_auc_score(y_test, y_prob[:,1]) 
  metrics.RocCurveDisplay.from_estimator(gnb, X_test, y_test)
  
  print(f"El valor de AUC para el GNB {i} es {round(auc,4)}")
  i+=1
  
  ACCM.append(acc_score)
  PRM.append(precision)
  FALLM.append(fallout)
  RCM.append(recall)
  F1M.append(f1)
  AUCM.append(auc)
  

####MEDIA DE LAS MEDIDAS  
print('')
print(f"The mean ACCURACY is {round(np.mean(np.array(ACCM)),4)} with a standard deviation of {round(np.std(np.array(ACCM)),4)}")
print(f"The mean PRECISION is {round(np.mean(np.array(PRM)),4)}  with a standard deviation of {round(np.std(np.array(PRM)),4)}")
print(f"The mean FALLOUT is {round(np.mean(np.array(FALLM)),4)}  with a standard deviation of {round(np.std(np.array(FALLM)),4)}")
print(f"The mean RECALL is {round(np.mean(np.array(RCM)),4)}  with a standard deviation of {round(np.std(np.array(RCM)),4)}")
print(f"The mean F1 is {round(np.mean(np.array(F1M)),4)}  with a standard deviation of {round(np.std(np.array(F1M)),4)}")
print(f"The mean AUC is {round(np.mean(np.array(AUCM)),4)}  with a standard deviation of {round(np.std(np.array(AUCM)),4)}")

