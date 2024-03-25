###DATASET Y MANIPULACION
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
### Decision Tree
from sklearn import tree
###METRICAS 
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


# fetch dataset 
diabetes = fetch_ucirepo(id=891) 

X = diabetes.data.features 
y = diabetes.data.targets 
###normalizamos los datos
#scaler = StandardScaler()
scaler = MinMaxScaler()
X_n = scaler.fit_transform(X)
X = pd.DataFrame(X_n, columns=X.columns)
X = X.drop(columns=['AnyHealthcare','NoDocbcCost','Sex','Fruits','Veggies'])
########FILTRADO PARA IGUALAR TARJETS#######

y0 = y[y == 0]
y1 = y[y == 1]
y0_sampled = y0.sample(frac=0.2, random_state=10)
y_combined = pd.concat([y0_sampled, y1]).dropna()
y=pd.DataFrame(y_combined)
X = X.loc[(~y_combined.isna()).index].dropna()

### CORRELACIONES
data = pd.concat([X, y], axis=1)  # Combina 'X' y 'y' en un solo DataFrame

# Calcula las correlaciones entre todas las características y 'y'
correlations = data.corrwith(data['Diabetes_binary'])

# Convierte las correlaciones a valores absolutos y ordénalas de manera descendente
correlations = correlations.abs().sort_values(ascending=False)

# Muestra las características más correlacionadas con 'y'
print(correlations)

### PARAMETROS A VALIDAR
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [7, 10, 15,30,50],  # Añadido 10 y 15
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3],  # Añadido 3
    'max_features': [None, 'sqrt', 'log2', 0.5],  # Añadido 0.5
    'max_leaf_nodes': [None, 5, 10, 15],  # Añadido 15
    'min_impurity_decrease': [0.0, 0.1, 0.2]
}
#SI PONGO MAXLEAFNODES MUCHO MÁS ALTO EN VALIDACION SE SOBREAJUSTA Y DSP SALE MUY MAL

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
  auc=0
  for criterion_i in param_grid['criterion']:
      for max_depth_i in param_grid['max_depth']:
          for min_samples_split_i in param_grid['min_samples_split']:
              for min_samples_leaf_i in param_grid['min_samples_leaf']:
                  for max_features_i in param_grid['max_features']:
                      for max_leaf_nodes_i in param_grid['max_leaf_nodes']:
                          for min_impurity_decrease_i in param_grid['min_impurity_decrease']:
                              
          
                              dt1 = tree.DecisionTreeClassifier(criterion= criterion_i,
                                                              max_depth = max_depth_i,
                                                              max_features = max_features_i,
                                                              max_leaf_nodes = max_leaf_nodes_i,
                                                              min_impurity_decrease = min_impurity_decrease_i,
                                                              min_samples_leaf = min_samples_leaf_i,
                                                              min_samples_split = min_samples_split_i)
                              dt1.fit(X_train, y_train)
                              y_val_prob = dt1.predict_proba(X_val)
                              auc1 = metrics.roc_auc_score(y_val, y_val_prob[:,1]) 
          
                              if auc1>auc:
                                  auc=auc1
                                  dt=dt1
                                  print(auc)
      
  ### NOS QUEDAMOS CON EL GNB CON LA MEJOR AUC
  y_train=np.ravel(y_train)
  dt.fit(X_train, y_train)
  y_pred = dt.predict(X_test)
  
  
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
  metrics.ConfusionMatrixDisplay.from_estimator(dt, X_test, y_test,cmap=plt.cm.Blues)
  #pintamos la curva
  y_prob = dt.predict_proba(X_test)
  auc = metrics.roc_auc_score(y_test, y_prob[:,1]) 
  metrics.RocCurveDisplay.from_estimator(dt, X_test, y_test) 
  print(f"El valor de AUC para el dt {i} es {round(auc,4)}")
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








