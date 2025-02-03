# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 18:00:55 2023

@author:
"""


#IMPORTAR LIBRERIAS
import pandas as pd
import numpy as np 
import seaborn as sns
pip install yfinance
import yfinance as yf


import pandas as pd          # MANIPULACION DE DATOS
import numpy as np          # GRAFICAS
import seaborn as sns       # Libreria de graficas
import matplotlib.pyplot as plt

# SUBIR BASE DE DATOS
path ="C:/Users/Hp/Downloads/Libro_1.xlsx"
Base = pd.read_excel(path)
print(Base)

# Convertir la primera columna en el índice
Base.set_index('Unnamed: 0', inplace=True)

# Renombrar el índice para mayor claridad (opcional)
Base.index.name = 'Period'

# Rellenar valores nulos con interpolación basada en períodos anteriores
Base = Base.interpolate(method='linear', axis=0)

# Verificar los cambios
print("Base con valores proyectados:")
print(Base.head())





#SUBIR BASE DE DATOS 
TSLA_df =  yf.download("TSLA", start= "2022-06-01", end="2023-09-21")
TSLA= pd.DataFrame(TSLA_df["Adj Close"])
TSLA.columns=["TSLA"] 

FDX_df =  yf.download("FDX", start= "2022-06-01", end="2023-09-21")
FDX= pd.DataFrame(FDX_df["Adj Close"])
FDX.columns=["FDX"]

AAPL_df =  yf.download("AAPL", start= "2022-06-01", end="2023-09-21")
AAPL= pd.DataFrame(AAPL_df["Adj Close"])
AAPL.columns=["AAPL"]

HST_df =  yf.download("HST", start= "2022-06-01", end="2023-09-21")
HST= pd.DataFrame(HST_df["Adj Close"])
HST.columns=["HST"]

C_df =  yf.download("C", start= "2022-06-01", end="2023-09-21")
C= pd.DataFrame(C_df["Adj Close"])
C.columns=["C"]

KO_df =  yf.download("KO", start= "2022-06-01", end="2023-09-21")
KO= pd.DataFrame(KO_df["Adj Close"])
KO.columns=["KO"] 

# 2.2. Rendimiento y riesgo 
TSLA_var = TSLA.pct_change()
TSLA_logf = np.log(TSLA).diff()
TSLA_var = TSLA_var.dropna()
TSLA_logf = TSLA_logf.dropna() 
TSLA_rend = np.mean(TSLA_logf)
TSLA_std = TSLA_logf.std()
print(TSLA_rend)
print(TSLA_std)

FDX_var = FDX.pct_change()
FDX_logf = np.log(FDX).diff()
FDX_var = FDX_var.dropna()
FDX_logf = FDX_logf.dropna() 
FDX_rend = np.mean(FDX_logf)
FDX_std = FDX_logf.std()
print(FDX_rend)
print(FDX_std)


AAPL_var = AAPL.pct_change()
AAPL_logf = np.log(AAPL).diff()
AAPL_var = AAPL_var.dropna()
AAPL_logf = AAPL_logf.dropna() 
AAPL_rend = np.mean(AAPL_logf)
AAPL_std = AAPL_logf.std()
print(AAPL_rend)
print(AAPL_std)

HST_var = HST.pct_change()
HST_logf = np.log(HST).diff()
HST_var = HST_var.dropna()
HST_logf = HST_logf.dropna() 
HST_rend = np.mean(HST_logf)
HST_std = HST_logf.std()
print(HST_rend)
print(HST_std)

C_var = C.pct_change()
C_logf = np.log(C).diff()
C_var = C_var.dropna()
C_logf = C_logf.dropna() 
C_rend = np.mean(C_logf)
C_std = C_logf.std()
print(C_rend)
print(C_std)

KO_var = KO.pct_change()
KO_logf = np.log(KO).diff()
KO_var = KO_var.dropna()
KO_logf = KO_logf.dropna() 
KO_rend = np.mean(KO_logf)
KO_std = KO_logf.std()
print(KO_rend)
print(KO_std)

#UNIR ACTIVOS
CARTERA0 = pd.concat([TSLA,FDX,AAPL,HST,C,KO],axis=1)
Rendimientos = pd.concat([TSLA_logf,FDX_logf,AAPL_logf,HST_logf,C_logf,KO_logf],axis=1)

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
Rendimientos1 = pd.concat([TSLA_logf,FDX_logf,AAPL_logf],axis=1)
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
Rendimientos2 = pd.concat([TSLA_logf,FDX_logf],axis=1)
lista_stocks =  ["TSLA_rend","FDX_rend"]
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