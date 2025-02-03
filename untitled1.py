# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 00:35:21 2024

@author: Hp
"""

#IMPORTAR LIBRERIAS
import pandas as pd
import numpy as np 
import seaborn as sns
pip install yfinance
import yfinance as yf

#SUBIR BASE DE DATOS 

NVDA_df =  yf.download("NVDA", start= "2024-01-01", end="2024-07-01")
NVDA= pd.DataFrame(NVDA_df["Adj Close"])
NVDA.columns=["NVDA"]

OSIS_df =  yf.download("OSIS", start= "2024-01-01", end="2024-07-01")
OSIS= pd.DataFrame(OSIS_df["Adj Close"])
OSIS.columns=["OSIS"]

EBF_df =  yf.download("EBF", start= "2024-01-01", end="2024-07-01")
EBF= pd.DataFrame(EBF_df["Adj Close"])
EBF.columns=["EBF"]

CZNC_df =  yf.download("CZNC", start= "2024-01-01", end="2024-07-01")
CZNC= pd.DataFrame(CZNC_df["Adj Close"])
CZNC.columns=["CZNC"]

RDN_df = yf.download("RDN", start="2024-01-01", end="2024-07-01")
RDN = pd.DataFrame(RDN_df["Adj Close"])
RDN.columns=["RDN"]

PDM_df = yf.download("PDM", start="2024-01-01", end="2024-07-01")
PDM = pd.DataFrame(PDM_df["Adj Close"])
PDM.columns=["PDM"]

IMKTA_df = yf.download("IMKTA", start="2024-01-01", end="2024-07-01")
IMKTA = pd.DataFrame(IMKTA_df["Adj Close"])
IMKTA.columns=["IMKTA"]

FNLC_df = yf.download("FNLC", start="2024-01-01", end="2024-07-01")
FNLC = pd.DataFrame(FNLC_df["Adj Close"])
FNLC.columns=["FNLC"]

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


CZNC_var = CZNC.pct_change()
CZNC_logf = np.log(CZNC).diff()
CZNC_var = CZNC_var.dropna()
CZNC_logf = CZNC_logf.dropna()
CZNC_rend = np.mean(CZNC_logf)
CZNC_std = CZNC_logf.std()
print(CZNC_rend)
print(CZNC_std)



RDN_var = RDN.pct_change()
RDN_logf = np.log(RDN).diff()
RDN_var = RDN_var.dropna()
RDN_logf = RDN_logf.dropna()
RDN_rend = np.mean(RDN_logf)
RDN_std = RDN_logf.std()
print(RDN_rend)
print(RDN_std)

PDM_var = PDM.pct_change()
PDM_logf = np.log(PDM).diff()
PDM_var = PDM_var.dropna()
PDM_logf = PDM_logf.dropna()
PDM_rend = np.mean(PDM_logf)
PDM_std = PDM_logf.std()
print(PDM_rend)
print(PDM_std)

IMKTA_var = IMKTA.pct_change()
IMKTA_logf = np.log(IMKTA).diff()
IMKTA_var = IMKTA_var.dropna()
IMKTA_logf = IMKTA_logf.dropna()
IMKTA_rend = np.mean(IMKTA_logf)
IMKTA_std = IMKTA_logf.std()
print(IMKTA_rend)
print(IMKTA_std)

FNLC_var = FNLC.pct_change()
FNLC_logf = np.log(FNLC).diff()
FNLC_var = FNLC_var.dropna()
FNLC_logf = FNLC_logf.dropna()
FNLC_rend = np.mean(FNLC_logf)
FNLC_std = FNLC_logf.std()
print(FNLC_rend)
print(FNLC_std)

#UNIR ACTIVOS
CARTERA0 = pd.concat([NVDA,OSIS,EBF,CZNC,RDN,PDM,IMKTA,FNLC],axis=1)
Rendimientos = pd.concat([NVDA_logf,OSIS_logf,EBF_logf,CZNC_logf,RDN_logf,PDM_logf,IMKTA_logf,FNLC_logf],axis=1)

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
Rendimientos1 = pd.concat([NVDA_logf,OSIS_logf,EBF_logf,CZNC_logf,RDN_logf,PDM_logf,IMKTA_logf,FNLC_logf],axis=1)
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
Rendimientos2 = pd.concat([NVDA_logf,OSIS_logf,EBF_logf,CZNC_logf,RDN_logf,PDM_logf,IMKTA_logf,FNLC_logf],axis=1)
lista_stocks =  ["NVDA_rend","OSIS_rend","EBF_rend","CZNC_rend","RDN_rend","PDM_rend","IMKTA_rend","FNLC_rend"]
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

rf = 0
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