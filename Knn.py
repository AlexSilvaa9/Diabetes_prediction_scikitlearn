###DATASET MANIPULATION
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
### KNN
from sklearn.neighbors import KNeighborsClassifier
###METRICS 
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


# fetch dataset 
diabetes = pd.read_csv("C://Users//alexs//OneDrive//Documentos//UNI//TERCERO//Sistemas inteligentes//Scikit-learn//Dataset//diabetes_binary_health_indicators_BRFSS2015.csv")

X = diabetes
y = diabetes["Diabetes_binary"]
X.drop(["Diabetes_binary"], axis = 1, inplace=True)

###normalizamos los datos
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_n = scaler.fit_transform(X)
X = pd.DataFrame(X_n, columns=X.columns)

########FILTRADO PARA IGUALAR TARJETS#######

y0 = y[y == 0]
y1 = y[y == 1]
y0_sampled = y0.sample(frac=0.2, random_state=10)
y_combined = pd.concat([y0_sampled, y1]).dropna()
y=pd.DataFrame(y_combined)
X = X.loc[(~y_combined.isna()).index].dropna()

### PARAMETROS A VALIDAR

param_grid = {
    'n_neighbors': [3,301,1001],
    'weights': ['distance'],
    'p': [1, 2]
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
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=rd)
  ###VALIDACION DE HIPERPARAMETROS
  y_val=np.ravel(y_val)
  auc=0
  for n_neighbors_i in param_grid['n_neighbors']:
        for weights_i in param_grid['weights']:
            for p_i in param_grid['p']:
                knn1 = KNeighborsClassifier(n_neighbors= n_neighbors_i,
                                            weights= weights_i,
                                            p=p_i,
                                            leaf_size= 1,)
                                                  
                knn1.fit(X_train, y_train)
                y_val_prob = knn1.predict_proba(X_val)
                auc1 = metrics.roc_auc_score(y_val, y_val_prob[:,1]) 
                  
                if auc1>auc:
                   auc=auc1
                   knn=knn1
                   print(auc)
      
  ### NOS QUEDAMOS CON EL GNB CON LA MEJOR AUC
  print(knn.get_params())
  y_train=np.ravel(y_train)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  
  
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
  metrics.ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test,cmap=plt.cm.Blues)
  #pintamos la curva
  y_prob = knn.predict_proba(X_test)
  auc = metrics.roc_auc_score(y_test, y_prob[:,1]) 
  metrics.RocCurveDisplay.from_estimator(knn, X_test, y_test) 
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








