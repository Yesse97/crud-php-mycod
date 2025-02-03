# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 23:27:35 2024

@author: Hp
"""
import pandas as pd      # para  manipulacion de datos
import numpy as np       # para hacer operaciones o graficos
import seaborn as sns
import matplotlib.pyplot as plt

path = r"C:\Users\Hp\Downloads\DATA agro nuevo.xlsx"
DATA = pd.read_excel(path)	
print(DATA)
****************CONVERSION A LOGARITMOS
log_DATA = np.log(DATA)
print(log_DATA)


log_log_DATA = np.log(log_DATA)
print(log_log_DATA)


****************

from statsmodels.formula.api import ols

MODELO1 = ols(formula ='PBI ~  X + GAS', data = log_DATA).fit()
MODELO1.summary()
 


 
##############
##CALCULO DE LA VARIANZA RESIDUAL O VARIANZA ESTIMADA POBLACIONAL
MODELO1.mse_resid
print(np.sqrt(MODELO1.mse_resid))


#ANOVA 
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm

anova_table = sm.stats.anova_lm(MODELO1, typ=2)
sum_sq=anova_table.iloc[:,[0,]]
df=anova_table.iloc[:,[1,]]

#SUMA DE CUADRADOS MEDIOS 
meansumsq = sum_sq.div(df.values)


cor1 = log_DATA.corr()
cor2 = log_DATA.corr()
cor3 = log_DATA.corr()


#CORRELACIONES correlacion 
cor1 = log_DATA['PBI'].corr(log_DATA['X'],method = 'pearson')
print('correlación Pearson  EXPORTACIONES: ', cor1)

cor2 = log_DATA['PBI'].corr(log_DATA['CRE'],method = 'pearson')
print('correlación PearsonCREDITO DIRECTO: ', cor2)

cor3 = log_DATA['PBI'].corr(log_DATA['GAS'],method = 'pearson')
print('correlación Pearson  GASTO PUBLICO: ', cor3)



# GRAFICO: X1 - error2 (HAY TENDENCIA NEGATIVA)
sns.lmplot(x = 'X', y = 'PBI', data = log_DATA)
plt.show()

# GRAFICO: X2 - error2 (HAY TENDENCIA NEGATIVA)
sns.lmplot(x = 'CRE', y = 'PBI', data = log_DATA)
plt.show()

# GRAFICO: X3 - error2 (HAY TENDENCIA NEGATIVA)
sns.lmplot(x = 'GAS', y = 'PBI', data = log_DATA)
plt.show()



####################################################################################
############################ DETECTAR HETEROCEDASTICIDAD ############################
####################################################################################

# PRIMERO EXTRAEMOS EL ERROR DEL MODELO
error = MODELO1.resid
error = error**2

# AGREGAR LA DATA
DATA1 = pd.concat([log_DATA, error], axis = 1)

# RENOMBRAR LAS COLUMNAS AGRGADAS
DATA1.rename({0:'error2'}, axis = 1, inplace = True)

############################### METODO INFORMAL ############################
# GRAFICAS

# GRAFICO: X1 - error2 (HAY TENDENCIA NEGATIVA)
sns.lmplot(x = 'X', y = 'error2', data = DATA1)
plt.show()

# GRAFICO: X2 - error2 (HAY TENDENCIA NEGATIVA)
sns.lmplot(x = 'CRE', y = 'error2', data = DATA1)
plt.show()

# GRAFICO: X3 - error2 (HAY TENDENCIA NEGATIVA)
sns.lmplot(x = 'GAS', y = 'error2', data = DATA1)
plt.show()


############################### METODO FORMAL ############################
'''
PRUEBA DE HIPOTESIS
HO =  HOMOCEDASTICIDAD     P-VALUE > 0.05
H1 =  HETEROCEDASTICIDAD   P-VALUE < 0.05
'''
# TEST DE BRESCH - PAGAN
import statsmodels.stats.api as smd
BP = smd.het_breuschpagan(MODELO1.resid, MODELO1.model.exog)[1]
print('P-value:', BP)

# TEST DE GOLDFELD - QUANT
import statsmodels.stats.api as sms
GQ = sms.het_goldfeldquandt(MODELO1.resid, MODELO1.model.exog)[1]
print('P-value:', GQ)


'''
# TEST DE WHITE - METODO 1 "GENERAL"
import statsmodels.stats.diagnostic as het_white
WHITE = het_white(MODELO1.resid, MODELO1.model.exog)[1]
print('P-value:', WHITE)
'''


# TEST DE WHITE - METODO 1 "GENERAL"
from statsmodels.stats.diagnostic import het_white
WHITE = het_white(MODELO1.resid, MODELO1.model.exog)[1]
print('P-value:',WHITE)


# TEST DE WHITE - METODO 2 "ESPECIFICO"
MODELO = ols(formula ='PBI ~  X  + GAS', data = DATA1).fit()
MODELO.summary()

***********************************************************
#MULTICOLINEALIDAD
from statsmodels.stats.outliers_influence import variance_inflation_factor

#VIF TEST
x = DATA1[["X" , "GAS" ]]


# Calcula los VIF para cada variable predictora
vif = pd.DataFrame()
vif["VARIABLE"] = x.columns

vif["VIF"] = [variance_inflation_factor(x.values, i)
                          for i in range(len(x.columns))]
print(vif)

*****************************************************
import matplotlib.pyplot as plt
sns.lmplot(x="HRS", y="NEIN", data=Base)
sns.lmplot(x="HRS", y="SCHOOL", data=Base)
sns.lmplot(x="HRS", y="RATE", data=Base)
sns.lmplot(x="HRS", y="ASSET", data=Base)
