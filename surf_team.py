import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors as KNN

'''
Esse programa tenta predizer qual deve ser o valor cobrado ao usuário
baseado no valor do serviço e do produto. Foram utilizados dados de um
mesmo CNPJ na região de SP.
'''

# Aqui se faz a leitura dos dados usados.
db_SP = pd.read_csv('db_SP.csv')

# Para tornar o processo mais rápido, vamos analisar o hospital com o maior numero de casos
cnpj = db_SP['base_hackaturing.cnpj'].value_counts(normalize = True).to_dict()
cnpj = list(cnpj.keys())[0]
new_db = db_SP[db_SP['base_hackaturing.cnpj'] == cnpj]

### Parte 1 - Validação do dado ###
# Vamos analisar os dados de maior importancia das duas caracteristicas selecionadas: Serviço e CBOS
db_cbo = pd.value_counts(new_db["base_hackaturing.cbos_solicitante"], normalize= True)
db_serv = pd.value_counts(new_db["base_hackaturing.servico"], normalize= True)


soma_c = 0
soma_s = 0

for i in range(len(db_serv)):
    if soma_s > 0.30:
        break
    soma_s += db_serv.iloc[i]
db_s_dict = db_serv.to_dict()
lista_db_s = []
for j in range(i):
    lista_db_s.append(list(db_s_dict.keys())[j])
    
for i in range(len(db_cbo)):
    if soma_c > 0.30:
        break
    soma_c += db_cbo.iloc[i]
db_c_dict = db_cbo.to_dict()
lista_db_c = []
for j in range(i):
    lista_db_c.append(list(db_c_dict.keys())[j])

# Criando as databases
lista_db_cbos = []
for cbos in lista_db_c:
    lista_db_cbos.append(new_db[new_db['base_hackaturing.cbos_solicitante'] == cbos] )
db_cbos = pd.concat(lista_db_cbos)

lista_db = []
for serv in lista_db_s:
    lista_db.append(db_cbos[db_cbos['base_hackaturing.servico'] == serv])

db_val = pd.concat(lista_db)[['base_hackaturing.cbos_solicitante','base_hackaturing.servico']]

# Criando o one hot encode
one_hot_serv = pd.get_dummies(db_val['base_hackaturing.servico'])
one_hot_cbo = pd.get_dummies(db_val['base_hackaturing.cbos_solicitante'])

db_val_one_hot = pd.concat([one_hot_cbo, one_hot_serv], axis = 1)

# Criando o modelo que verificará se um dado é válido, esse modelo será um KNN, os pontos que possuirem
#uma distancia maior que d_max com todos os centros será considerado um dado inválido.
knn = KNN()
knn.fit(db_val_one_hot)

d_max = np.median(knn.kneighbors(db_val_one_hot.iloc[0].values.reshape(1,db_val_one_hot.iloc[0].values.size))[1])

def validacao(ponto, d_max):
	return np.min(knn.kneighbors(ponto)[1]) < d_max

##################################################################################################################

### PARTE 2 - Regressor para o valor cobrado

# Como existem diversos produtos, vamos usar apenas os mais comuns, que formam 50% da quantidade total
products = new_db['base_hackaturing.descricao_despesa'].value_counts(normalize=True).to_dict()
soma = 0
for i in range( len(products.keys())):
	if soma > 0.5:
		break
	soma += products[list(products.keys())[i]]

list_prod = list(products.keys())[0:i]

# Aqui criamos um Data Frame só com os produtos selecionados na etapa anterior
list_db = []
for prod in list_prod:
	list_db.append(new_db[new_db['base_hackaturing.descricao_despesa'] == prod])
db = pd.concat(list_db)

# Passamos de dados categóricos para númericos usando one hot enconding e juntamos os dados
db_one = pd.get_dummies(db['base_hackaturing.descricao_despesa'])
db_two = db['base_hackaturing.valor_item']
db_total = pd.concat([db_one,db_two], axis=1)

# Aqui preparamos os dados para o treino e teste do classificador (Usamos MLP)
Y = db['base_hackaturing.valor_cobrado'].copy()
X = db_total.copy()
Y = Y.values
X = X.values
#Removemos os NaN
list_nan = []
for i in range(Y.shape[0]):
	if np.isnan(Y[i]):
		list_nan.append(i)
Y = pd.DataFrame(Y)
X = pd.DataFrame(X)

Y = Y.drop(Y.index[list_nan]).values
X = X.drop(X.index[list_nan]).values

Y = Y.reshape((Y.size,))

# Separamos os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X,Y)

# Criamos o classificador
clf = RFR()
clf.fit(X_train, y_train)

def predicacao(dado):
	return clf.predict(dado)


###############################

def main(dado):
	# Programa principal
	if validacao(dado, d_max):
		print("Dado validado! Preparando regressão.")
		print("Predição concluída:", predicacao(dado))
		return 1
	print("Dado inválido. Encaminhando para Auditoria")
	return 0