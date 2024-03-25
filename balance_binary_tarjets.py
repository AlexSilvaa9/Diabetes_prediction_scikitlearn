# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 00:10:46 2023

@author: alexs
"""

from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd

# fetch dataset 
diabetes = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = diabetes.data.features 
y = diabetes.data.targets 
y0 = y[y == 0]

# Filtrar las etiquetas de la clase 1
y1 = y[y == 1]

# Muestrear un subconjunto de etiquetas de la clase 0
y0_sampled = y0.sample(frac=0.2, random_state=10)

# Combinar las etiquetas de ambas clases
y_combined = pd.concat([y0_sampled, y1]).dropna()


y_combined=pd.DataFrame(y_combined)
X_combined = X.loc[(~y_combined.isna()).index].dropna()






