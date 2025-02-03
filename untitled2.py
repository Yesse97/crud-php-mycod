# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:00:09 2024

@author: Hp
"""
#PORTAFOLIO_GENERAL 
#IMPORTAR LIBRERIAS
import pandas as pd
import numpy as np 
import seaborn as sns
pip install yfinance
import yfinance as yf

#SUBIR BASE DE DATOS 
EBF_df =  yf.download("EBF", start= "2024-01-31", end="2024-06-28")
EBF= pd.DataFrame(EBF_df["Adj Close"])
EBF.columns=["EBF"]

FFBC_df = yf.download("FFBC", start="2024-01-31", end="2024-06-28")
FFBC = pd.DataFrame(FFBC_df["Adj Close"])
FFBC.columns=["FFBC"]

FULT_df = yf.download("FULT", start="2024-01-31", end="2024-06-28")
FULT = pd.DataFrame(FULT_df["Adj Close"])
FULT.columns=["FULT"]

HVT_df = yf.download("HVT", start="2024-01-31", end="2024-06-28")
HVT = pd.DataFrame(HVT_df["Adj Close"])
HVT.columns=["HVT"]

IMKTA_df = yf.download("IMKTA", start="2024-01-31", end="2024-06-28")
IMKTA = pd.DataFrame(IMKTA_df["Adj Close"])
IMKTA.columns=["IMKTA"]

MLKN_df =  yf.download("MLKN", start= "2024-01-31", end="2024-06-28")
MLKN= pd.DataFrame(MLKN_df["Adj Close"])
MLKN.columns=["MLKN"]

MPC_df = yf.download("MPC", start="2024-01-31", end="2024-06-28")
MPC = pd.DataFrame(MPC_df["Adj Close"])
MPC.columns=["MPC"]

NVDA_df =  yf.download("NVDA", start= "2024-01-31", end="2024-06-28")
NVDA= pd.DataFrame(NVDA_df["Adj Close"])
NVDA.columns=["NVDA"]

OSIS_df =  yf.download("OSIS", start= "2024-01-31", end="2024-06-28")
OSIS= pd.DataFrame(OSIS_df["Adj Close"])
OSIS.columns=["OSIS"]

PEBO_df = yf.download("PEBO", start="2024-01-31", end="2024-06-28")
PEBO = pd.DataFrame(PEBO_df["Adj Close"])
PEBO.columns=["PEBO"]

QUAD_df =  yf.download("QUAD", start= "2024-01-31", end="2024-06-28")
QUAD= pd.DataFrame(QUAD_df["Adj Close"])
QUAD.columns=["QUAD"]

VTLE_df = yf.download("VTLE", start="2024-01-31", end="2024-06-28")
VTLE = pd.DataFrame(VTLE_df["Adj Close"])
VTLE.columns=["VTLE"]

VVX_df = yf.download("VVX", start="2024-01-31", end="2024-06-28")
VVX = pd.DataFrame(VVX_df["Adj Close"])
VVX.columns=["VVX"]

# 2.2. Rendimiento y riesgo 
NVDA_var = NVDA.pct_change()
NVDA_logf = np.log(NVDA).diff()
NVDA_var = NVDA_var.dropna()
NVDA_logf = NVDA_logf.dropna() 
NVDA_rend = np.mean(NVDA_logf)
NVDA_std = NVDA_logf.std()
print(NVDA_rend)
print(NVDA_std)

OSIS_var = OSIS.pct_change()
OSIS_logf = np.log(OSIS).diff()
OSIS_var = OSIS_var.dropna()
OSIS_logf = OSIS_logf.dropna() 
OSIS_rend = np.mean(OSIS_logf)
OSIS_std = OSIS_logf.std()
print(OSIS_rend)
print(OSIS_std)

EBF_var = EBF.pct_change()
EBF_logf = np.log(EBF).diff()
EBF_var = EBF_var.dropna()
EBF_logf = EBF_logf.dropna()
EBF_rend = np.mean(EBF_logf)
EBF_std = EBF_logf.std()
print(EBF_rend)
print(EBF_std)

MLKN_var = MLKN.pct_change()
MLKN_logf = np.log(MLKN).diff()
MLKN_var = MLKN_var.dropna()
MLKN_logf = MLKN_logf.dropna()
MLKN_rend = np.mean(MLKN_logf)
MLKN_std = MLKN_logf.std()
print(MLKN_rend)
print(MLKN_std)

QUAD_var = QUAD.pct_change()
QUAD_logf = np.log(QUAD).diff()
QUAD_var = QUAD_var.dropna()
QUAD_logf = QUAD_logf.dropna()
QUAD_rend = np.mean(QUAD_logf)
QUAD_std = QUAD_logf.std()
print(QUAD_rend)
print(QUAD_std)

FULT_var = FULT.pct_change()
FULT_logf = np.log(FULT).diff()
FULT_var = FULT_var.dropna()
FULT_logf = FULT_logf.dropna()
FULT_rend = np.mean(FULT_logf)
FULT_std = FULT_logf.std()
print(FULT_rend)
print(FULT_std)

HVT_var = HVT.pct_change()
HVT_logf = np.log(HVT).diff()
HVT_var = HVT_var.dropna()
HVT_logf = HVT_logf.dropna()
HVT_rend = np.mean(HVT_logf)
HVT_std = HVT_logf.std()
print(HVT_rend)
print(HVT_std)

PEBO_var = PEBO.pct_change()
PEBO_logf = np.log(PEBO).diff()
PEBO_var = PEBO_var.dropna()
PEBO_logf = PEBO_logf.dropna()
PEBO_rend = np.mean(PEBO_logf)
PEBO_std = PEBO_logf.std()
print(PEBO_rend)
print(PEBO_std)

FFBC_var = FFBC.pct_change()
FFBC_logf = np.log(FFBC).diff()
FFBC_var = FFBC_var.dropna()
FFBC_logf = FFBC_logf.dropna()
FFBC_rend = np.mean(FFBC_logf)
FFBC_std = FFBC_logf.std()
print(FFBC_rend)
print(FFBC_std)

MPC_var = MPC.pct_change()
MPC_logf = np.log(MPC).diff()
MPC_var = MPC_var.dropna()
MPC_logf = MPC_logf.dropna()
MPC_rend = np.mean(MPC_logf)
MPC_std = MPC_logf.std()
print(MPC_rend)
print(MPC_std)

VTLE_var = VTLE.pct_change()
VTLE_logf = np.log(VTLE).diff()
VTLE_var = VTLE_var.dropna()
VTLE_logf = VTLE_logf.dropna()
VTLE_rend = np.mean(VTLE_logf)
VTLE_std = VTLE_logf.std()
print(VTLE_rend)
print(VTLE_std)

VVX_var = VVX.pct_change()
VVX_logf = np.log(VVX).diff()
VVX_var = VVX_var.dropna()
VVX_logf = VVX_logf.dropna()
VVX_rend = np.mean(VVX_logf)
VVX_std = VVX_logf.std()
print(VVX_rend)
print(VVX_std)

IMKTA_var = IMKTA.pct_change()
IMKTA_logf = np.log(IMKTA).diff()
IMKTA_var = IMKTA_var.dropna()
IMKTA_logf = IMKTA_logf.dropna()
IMKTA_rend = np.mean(IMKTA_logf)
IMKTA_std = IMKTA_logf.std()
print(IMKTA_rend)
print(IMKTA_std)

#UNIR ACTIVOS
CARTERA0 = pd.concat([NVDA,OSIS,EBF,MLKN,QUAD,FULT,HVT,PEBO,FFBC,MPC,VTLE,VVX,IMKTA],axis=1)
Rendimientos = pd.concat([NVDA_logf,OSIS_logf,EBF_logf,MLKN_logf,QUAD_logf,FULT_logf,HVT_logf,PEBO_logf,FFBC_logf,MPC_logf,VTLE_logf,VVX_logf,IMKTA_logf],axis=1)

#GRAFICO DE COMPORTAMIENTO DE LOS ACTIVOS
import matplotlib.pyplot as plt
plt.figure(figsize=(12.2,4)) 
for i in CARTERA0.columns.values:
    plt.plot(CARTERA0[i], label=i)
plt.title("Price of the Stocks")
plt.xlabel("Date",fontsize=8)
plt.ylabel("Price",fontsize=8)
plt.legend(CARTERA0.columns.values, loc="upper left")
plt.show() 

#################PORTAFOLIO#######################
#Matriz de correlación 
Rendimientos1 = pd.concat([NVDA_logf,OSIS_logf,EBF_logf,MLKN_logf,QUAD_logf,FULT_logf,HVT_logf,PEBO_logf,FFBC_logf,MPC_logf,VTLE_logf,VVX_logf,IMKTA_logf],axis=1)
Rendimientos1.corr()

#   Mapa De Calor 
import seaborn as sns
correlation_mat = Rendimientos1.corr()
plt.figure(figsize=(12.2,4.5))
sns.heatmap(correlation_mat, annot = True)
plt.title("Matriz de Correlación") 
plt.xlabel("Activos",fontsize=18)
plt.ylabel("Activos",fontsize=18)
plt.show()



#DESPUÉS DE ELEGIR A LAS VARIABLES, CREAMOS NUESTRA LISTA
#TSLA-FDX 
Rendimientos2 = pd.concat([NVDA_logf,OSIS_logf,EBF_logf,MLKN_logf,QUAD_logf,FULT_logf,HVT_logf,PEBO_logf,FFBC_logf,MPC_logf,VTLE_logf,VVX_logf,IMKTA_logf],axis=1)
lista_stocks =  ["NVDA_rend","OSIS_rend","EBF_rend","MLKN_rend","QUAD_rend","FULT_rend","HVT_rend","PEBO_rend","FFBC_rend","MPC_rend","VTLE_rend","VVX_rend","IMKTA_rend"]
print(lista_stocks)
num_stocks = len(lista_stocks) #Contará los activos
print(num_stocks) 

# Crear Variables
portafolioR = []
portafolioSD = []
portafolioPESOS = [] 

#CREAR PESOS
for x in range (5000):
    pesos = np.random.random(num_stocks)
    pesos /= np.sum(pesos)
    portafolioPESOS.append(pesos)
    portafolioR.append(np.dot(Rendimientos2.mean(),pesos))
    portafolioSD.append(np.sqrt(np.dot(pesos.T, np.dot(Rendimientos2.cov(), pesos))))
#PESOS
print(portafolioPESOS)
print(portafolioR)
print(portafolioSD)

#DICCIONARIO DE RENDIMIENTOS, RIESGOS Y PESOS (organiza)
diccionario = {"Rendimientos2":portafolioR, "Riesgo":portafolioSD}
for contador, ticker in enumerate(Rendimientos2.columns.tolist()):
    diccionario["Peso" + ticker] = [w[contador] for w in portafolioPESOS]
print(diccionario)
    
matrizportafolios = pd.DataFrame(diccionario) #ordenar en DATA FRAME
print(matrizportafolios) 

#PORTAFOLIO EFICIENTE: MÍNIMA VARIANZA, MENOR RIESGO
EFICIENTE = matrizportafolios.plot(x= "Riesgo", y="Rendimientos2", kind = "scatter") 
#ver rendimiento, riesgo y pesos
varianzaminima = matrizportafolios.iloc[matrizportafolios["Riesgo"].idxmin()]
plt.scatter(x = varianzaminima[1], y = varianzaminima[0], color = "red", marker = "*", s = 100)
print(varianzaminima)

#PORTAFOLIO ÓPTIMO O TANGENTE: MAXIMIZA LA RENTABILIDAD

rf = 2.841
matrizportafolios['Sharpe Ratio'] = (matrizportafolios['Rendimientos2'] - rf) / matrizportafolios['Riesgo']
indice_optimo = matrizportafolios['Sharpe Ratio'].idxmax()
#ver rendimiento, riesgo y pesos
OPTIMO = matrizportafolios.loc[indice_optimo]
matrizportafolios.plot(x = 'Riesgo', y = 'Rendimientos2', kind = 'scatter')
plt.scatter(x=varianzaminima[1], y=varianzaminima[0], color ='red', marker = '*', s=100)
plt.scatter(x=OPTIMO['Riesgo'], y=OPTIMO['Rendimientos2'], color='black', marker='*', s=100)

print(OPTIMO)

#CALCULAR RATIO SHARPE

RATIO_SHARPE_O = (OPTIMO["Rendimientos2"]-rf)/OPTIMO["Riesgo"]
#Cuando un Ratio de Sharpe es negativo indica que su rendimiento es menor al de la rentabilidad sin riesgo.
se toma el eficeinte

"El portafolio eficiente asume normalidad en los rendimientos; sin embargo, el portafolio"
 "optimo muestra que hay un riesgo que engloba la no normalidad de los rendmientos"
" óptimo el cual maximice la rentabilidad sujeta a una restricción de riesgo de pérdida y no solo a una medida de riesgo paramétrica como la desviación estándar."
#si se tiene dos portafolios optimos... se elige el que tiene menor ratio sharpe 


















#TSLA-AAPL
Rendimientos3 = pd.concat([TSLA_logf,AAPL_logf],axis=1)
lista_stocks =  ["TSLA_rend","AAPL_rend"]
print(lista_stocks)
num_stocks = len(lista_stocks) #Contará los activos
print(num_stocks) 

# Crear Variables
portafolioR = []
portafolioSD = []
portafolioPESOS = [] 

#CREAR PESOS
for x in range (5000):
    pesos = np.random.random(num_stocks)
    pesos /= np.sum(pesos)
    portafolioPESOS.append(pesos)
    portafolioR.append(np.dot(Rendimientos3.mean(),pesos))
    portafolioSD.append(np.sqrt(np.dot(pesos.T, np.dot(Rendimientos3.cov(), pesos))))
#PESOS
print(portafolioPESOS)
print(portafolioR)
print(portafolioSD)

#DICCIONARIO DE RENDIMIENTOS, RIESGOS Y PESOS (organiza)
diccionario = {"Rendimientos3":portafolioR, "Riesgo":portafolioSD}
for contador, ticker in enumerate(Rendimientos3.columns.tolist()):
    diccionario["Peso" + ticker] = [w[contador] for w in portafolioPESOS]
print(diccionario)
    
matrizportafolios = pd.DataFrame(diccionario) #ordenar en DATA FRAME
print(matrizportafolios) 

#PORTAFOLIO EFICIENTE: MÍNIMA VARIANZA, MENOR RIESGO
EFICIENTE = matrizportafolios.plot(x= "Riesgo", y="Rendimientos3", kind = "scatter") 
#ver rendimiento, riesgo y pesos
varianzaminima = matrizportafolios.iloc[matrizportafolios["Riesgo"].idxmin()]
plt.scatter(x = varianzaminima[1], y = varianzaminima[0], color = "red", marker = "*", s = 100)
print(varianzaminima)

#PORTAFOLIO ÓPTIMO O TANGENTE: MAXIMIZA LA RENTABILIDAD

rf = 0
matrizportafolios['Sharpe Ratio'] = (matrizportafolios['Rendimientos3'] - rf) / matrizportafolios['Riesgo']
indice_optimo = matrizportafolios['Sharpe Ratio'].idxmax()
#ver rendimiento, riesgo y pesos
OPTIMO = matrizportafolios.loc[indice_optimo]
matrizportafolios.plot(x = 'Riesgo', y = 'Rendimientos3', kind = 'scatter')
plt.scatter(x=varianzaminima[1], y=varianzaminima[0], color ='red', marker = '*', s=100)
plt.scatter(x=OPTIMO['Riesgo'], y=OPTIMO['Rendimientos3'], color='black', marker='*', s=100)

print(OPTIMO)

#FDX-AAPL
Rendimientos4 = pd.concat([FDX_logf,AAPL_logf],axis=1)
lista_stocks =  ["FDX_rend","AAPL_rend"]
print(lista_stocks)
num_stocks = len(lista_stocks) #Contará los activos
print(num_stocks) 

# Crear Variables
portafolioR = []
portafolioSD = []
portafolioPESOS = [] 

#CREAR PESOS
for x in range (5000):
    pesos = np.random.random(num_stocks)
    pesos /= np.sum(pesos)
    portafolioPESOS.append(pesos)
    portafolioR.append(np.dot(Rendimientos3.mean(),pesos))
    portafolioSD.append(np.sqrt(np.dot(pesos.T, np.dot(Rendimientos3.cov(), pesos))))
#PESOS
print(portafolioPESOS)
print(portafolioR)
print(portafolioSD)

#DICCIONARIO DE RENDIMIENTOS, RIESGOS Y PESOS (organiza)
diccionario = {"Rendimientos4":portafolioR, "Riesgo":portafolioSD}
for contador, ticker in enumerate(Rendimientos4.columns.tolist()):
    diccionario["Peso" + ticker] = [w[contador] for w in portafolioPESOS]
print(diccionario)
    
matrizportafolios = pd.DataFrame(diccionario) #ordenar en DATA FRAME
print(matrizportafolios) 

#PORTAFOLIO EFICIENTE: MÍNIMA VARIANZA, MENOR RIESGO
EFICIENTE = matrizportafolios.plot(x= "Riesgo", y="Rendimientos4", kind = "scatter") 
#ver rendimiento, riesgo y pesos
varianzaminima = matrizportafolios.iloc[matrizportafolios["Riesgo"].idxmin()]
plt.scatter(x = varianzaminima[1], y = varianzaminima[0], color = "red", marker = "*", s = 100)
print(varianzaminima)

#PORTAFOLIO ÓPTIMO O TANGENTE: MAXIMIZA LA RENTABILIDAD

rf = 0
matrizportafolios['Sharpe Ratio'] = (matrizportafolios['Rendimientos4'] - rf) / matrizportafolios['Riesgo']
indice_optimo = matrizportafolios['Sharpe Ratio'].idxmax()
#ver rendimiento, riesgo y pesos
OPTIMO = matrizportafolios.loc[indice_optimo]
matrizportafolios.plot(x = 'Riesgo', y = 'Rendimientos4', kind = 'scatter')
plt.scatter(x=varianzaminima[1], y=varianzaminima[0], color ='red', marker = '*', s=100)
plt.scatter(x=OPTIMO['Riesgo'], y=OPTIMO['Rendimientos4'], color='black', marker='*', s=100)

print(OPTIMO)