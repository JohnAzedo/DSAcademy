# %%
# Imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# %%
# Carregando a base
base = pd.read_csv('census.csv')


# %%
# Variaveis
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values
attrsNominal = [1, 3, 5, 6, 7, 8, 9, 13]


# %% Não executar essa célula
# Previsores: Transformando atributos nominais em produtos discretos
labelencoder = LabelEncoder()

for column in attrsNominal:
    previsores[:, column] = labelencoder.fit_transform(previsores[:,column])


# %%
# Previsores: Criando variáveis dummys para atributos nominais
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), attrsNominal)],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()


# %%
# Classe(s): Transformando atributos nominais em produtos discretos
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)


# %%
# Escalonamento de atributos: Padronização
scaler = StandardScaler()

# Nesse caso, algorítimos que usam a distancia euclidiana não
# irão considerar valores maiores como mais importantes
previsores = scaler.fit_transform(previsores)