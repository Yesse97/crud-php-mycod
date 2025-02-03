# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 23:07:50 2024

@author: Hp
"""

#PORTAFOLIO_GENERAL_COBERTURADO
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

CMC_df = yf.download("CMC", start="2024-01-01", end="2024-07-01")
CMC = pd.DataFrame(CMC_df["Adj Close"])
CMC.columns=["CMC"]

FNLC_df = yf.download("FNLC", start="2024-01-01", end="2024-07-01")
FNLC = pd.DataFrame(FNLC_df["Adj Close"])
FNLC.columns=["FNLC"]

GEF_df = yf.download("GEF", start="2024-01-01", end="2024-07-01")
GEF = pd.DataFrame(GEF_df["Adj Close"])
GEF.columns=["GEF"]

KMT_df =  yf.download("KMT", start= "2024-01-01", end="2024-07-01")
KMT= pd.DataFrame(KMT_df["Adj Close"])
KMT.columns=["KMT"]

NMIH_df = yf.download("NMIH", start="2024-01-01", end="2024-07-01")
NMIH = pd.DataFrame(NMIH_df["Adj Close"])
NMIH.columns=["NMIH"]

PDM_df = yf.download("PDM", start="2024-01-01", end="2024-07-01")
PDM = pd.DataFrame(PDM_df["Adj Close"])
PDM.columns=["PDM"]

SCS_df =  yf.download("SCS", start= "2024-01-01", end="2024-07-01")
SCS= pd.DataFrame(SCS_df["Adj Close"])
SCS.columns=["SCS"]

TPC_df = yf.download("TPC", start="2024-01-01", end="2024-07-01")
TPC = pd.DataFrame(TPC_df["Adj Close"])
TPC.columns=["TPC"]

UAL_df = yf.download("UAL", start="2024-01-01", end="2024-07-01")
UAL = pd.DataFrame(UAL_df["Adj Close"])
UAL.columns=["UAL"]

WAFD_df = yf.download("WAFD", start="2024-01-01", end="2024-07-01")
WAFD = pd.DataFrame(WAFD_df["Adj Close"])
WAFD.columns=["WAFD"]

URTH_df = yf.download("URTH", start="2024-01-01", end="2024-07-01")
URTH = pd.DataFrame(URTH_df["Adj Close"])
URTH.columns=["URTH"]


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

CMC_var = CMC.pct_change()
CMC_logf = np.log(CMC).diff()
CMC_var = CMC_var.dropna()
CMC_logf = CMC_logf.dropna()
CMC_rend = np.mean(CMC_logf)
CMC_std = CMC_logf.std()
print(CMC_rend)
print(CMC_std)

FNLC_var = FNLC.pct_change()
FNLC_logf = np.log(FNLC).diff()
FNLC_var = FNLC_var.dropna()
FNLC_logf = FNLC_logf.dropna()
FNLC_rend = np.mean(FNLC_logf)
FNLC_std = FNLC_logf.std()
print(FNLC_rend)
print(FNLC_std)

GEF_var = GEF.pct_change()
GEF_logf = np.log(GEF).diff()
GEF_var = GEF_var.dropna()
GEF_logf = GEF_logf.dropna()
GEF_rend = np.mean(GEF_logf)
GEF_std = GEF_logf.std()
print(GEF_rend)
print(GEF_std)

KMT_var = KMT.pct_change()
KMT_logf = np.log(KMT).diff()
KMT_var = KMT_var.dropna()
KMT_logf = KMT_logf.dropna() 
KMT_rend = np.mean(KMT_logf)
KMT_std = KMT_logf.std()
print(KMT_rend)
print(KMT_std)

NMIH_var = NMIH.pct_change()
NMIH_logf = np.log(NMIH).diff()
NMIH_var = NMIH_var.dropna()
NMIH_logf = NMIH_logf.dropna()
NMIH_rend = np.mean(NMIH_logf)
NMIH_std = NMIH_logf.std()
print(NMIH_rend)
print(NMIH_std)

PDM_var = PDM.pct_change()
PDM_logf = np.log(PDM).diff()
PDM_var = PDM_var.dropna()
PDM_logf = PDM_logf.dropna()
PDM_rend = np.mean(PDM_logf)
PDM_std = PDM_logf.std()
print(PDM_rend)
print(PDM_std)

SCS_var = SCS.pct_change()
SCS_logf = np.log(SCS).diff()
SCS_var = SCS_var.dropna()
SCS_logf = SCS_logf.dropna()
SCS_rend = np.mean(SCS_logf)
SCS_std = SCS_logf.std()
print(SCS_rend)
print(SCS_std)


TPC_var = TPC.pct_change()
TPC_logf = np.log(TPC).diff()
TPC_var = TPC_var.dropna()
TPC_logf = TPC_logf.dropna()
TPC_rend = np.mean(TPC_logf)
TPC_std = TPC_logf.std()
print(TPC_rend)
print(TPC_std)


UAL_var = UAL.pct_change()
UAL_logf = np.log(UAL).diff()
UAL_var = UAL_var.dropna()
UAL_logf = UAL_logf.dropna()
UAL_rend = np.mean(UAL_logf)
UAL_std = UAL_logf.std()
print(UAL_rend)
print(UAL_std)

WAFD_var = WAFD.pct_change()
WAFD_logf = np.log(WAFD).diff()
WAFD_var = WAFD_var.dropna()
WAFD_logf = WAFD_logf.dropna()
WAFD_rend = np.mean(WAFD_logf)
WAFD_std = WAFD_logf.std()
print(WAFD_rend)
print(WAFD_std)


URTH_var = URTH.pct_change()
URTH_logf = np.log(URTH).diff()
URTH_var = URTH_var.dropna()
URTH_logf = URTH_logf.dropna()
URTH_rend = np.mean(URTH_logf)
URTH_std = URTH_logf.std()
print(URTH_rend)
print(URTH_std)

#UNIR ACTIVOS
CARTERA0 = pd.concat([NVDA,OSIS,EBF,MLKN,QUAD,FULT,PEBO,FFBC,MPC,VTLE,VVX,CMC,FNLC,GEF,KMT,NMIH,PDM,SCS,TPC,UAL,WAFD,URTH],axis=1)
Rendimientos = pd.concat([NVDA_logf,OSIS_logf,EBF_logf,MLKN_logf,QUAD_logf,FULT_logf,PEBO_logf,FFBC_logf,MPC_logf,VTLE_logf,VVX_logf,CMC_logf,FNLC_logf,GEF_logf,KMT_logf,NMIH_logf,PDM_logf,SCS_logf,TPC_logf,UAL_logf,WAFD_logf,URTH_logf],axis=1)

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
Rendimientos1 = pd.concat([NVDA_logf,OSIS_logf,EBF_logf,MLKN_logf,QUAD_logf,FULT_logf,PEBO_logf,FFBC_logf,MPC_logf,VTLE_logf,VVX_logf,CMC_logf,FNLC_logf,GEF_logf,KMT_logf,NMIH_logf,PDM_logf,SCS_logf,TPC_logf,UAL_logf,WAFD_logf,URTH_logf],axis=1)
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
Rendimientos2 = pd.concat([NVDA_logf,OSIS_logf,EBF_logf,MLKN_logf,QUAD_logf,FULT_logf,PEBO_logf,FFBC_logf,MPC_logf,VTLE_logf,VVX_logf,CMC_logf,FNLC_logf,GEF_logf,KMT_logf,NMIH_logf,PDM_logf,SCS_logf,TPC_logf,UAL_logf,WAFD_logf,URTH_logf],axis=1)
lista_stocks =  ["NVDA_rend","OSIS_rend","EBF_rend","MLKN_rend","QUAD_rend","FULT_rend","PEBO_rend","FFBC_rend","MPC_rend","VTLE_rend","VVX_rend","CMC_rend","FNLC_rend","GEF_rend","KMT_rend","NMIH_rend","PDM_rend","SCS_rend","TPC_rend","UAL_rend","WAFD_rend","URTH_rend"]
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