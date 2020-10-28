#!../venv/bin python3.8
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 10:49:25 2020

@author: johnazedo
"""

# %% 
# Importando pandas
import pandas as pd
import numpy as np
base = pd.read_csv('../#3/credit_data.csv')


# %% 
# Idade menor que 0
# Não aceita valores menor que zero para descobrir a media
mean = base['age'][base.age > 0].mean()
# Localiza os dados e substitui na coluna age pelo novo valor
base.loc[base.age < 0, 'age'] = mean
base.loc[base.age == mean]


# %%
# Idade nula
# Procura os valores nulos do DF
base.loc[pd.isnull(base['age'])]


# %%
# Divisão de previsores e classe
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values


# %%
# Tratamento dos valores faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores = imputer.transform(previsores[:, 0:3])
previsores


# %%
# Escalonamento de atributos: Padronização
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Nesse caso, algorítimos que usam a distancia euclidiana não
# irão considerar valores maiores como mais importantes
previsores = scaler.fit_transform(previsores)


# %%
# Divisão de dados para treinamento e dados para teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)   


# %%
# Naive Bayes
# OBS: Fazer o pre-processamento não mostrou nenhum ganho de
# performace do algorítmo naive bayes
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)


# %%
# Predizendo valores
previsoes = classificador.predict(previsores_teste)


# %%
# Verificação
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)