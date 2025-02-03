# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 22:14:03 2024

@author: Hp
"""

import pandas as pd      # para  manipulacion de datos
import numpy as np       # para hacer operaciones o graficos
import seaborn as sns
import matplotlib.pyplot as plt

path = r"C:\Users\Hp\Downloads\Seminario_Tesis.xlsx"
DATA = pd.read_excel(path)	
print(DATA)

from statsmodels.formula.api import ols

MODELO1 = ols(formula ='Tasa_AcTiv ~  T_Desempleo + Var_PBI + T_Colocaciones ', data = DATA).fit()
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


cor1 = DATA.corr()
cor2 = DATA.corr()
cor3 = DATA.corr()


#CORRELACIONES correlacion 
cor1 = DATA['Tasa_AcTiv'].corr(DATA['T_Desempleo'],method = 'pearson')
print('correlación Pearson: ', cor1)

cor2 = DATA['Tasa_AcTiv'].corr(DATA['Var_PBI'],method = 'pearson')
print('correlación Pearson: ', cor2)

cor3 = DATA['Tasa_AcTiv'].corr(DATA['T_Colocaciones'],method = 'pearson')
print('correlación Pearson: ', cor3)


####################################################################################
############################ DETECTAR HETEROCEDASTICIDAD ############################
####################################################################################

# PRIMERO EXTRAEMOS EL ERROR DEL MODELO
error = MODELO1.resid
error = error**2

# AGREGAR LA DATA
DATA1 = pd.concat([DATA, error], axis = 1)

# RENOMBRAR LAS COLUMNAS AGRGADAS
DATA1.rename({0:'error2'}, axis = 1, inplace = True)

############################### METODO INFORMAL ############################
# GRAFICAS

# GRAFICO: X1 - error2 (HAY TENDENCIA NEGATIVA)
sns.lmplot(x = 'T_Desempleo', y = 'error2', data = DATA1)
plt.show()

# GRAFICO: X2 - error2 (HAY TENDENCIA NEGATIVA)
sns.lmplot(x = 'Var_PBI', y = 'error2', data = DATA1)
plt.show()

# GRAFICO: X3 - error2 (HAY TENDENCIA NEGATIVA)
sns.lmplot(x = 'T_Colocaciones', y = 'error2', data = DATA1)
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

# TEST DE WHITE - METODO 2 "ESPECIFICO"
MODELO = ols(formula ='error2 ~  T_Desempleo + Var_PBI + T_Colocaciones', data = DATA1).fit()
MODELO.summary()

***********************************************************
#MULTICOLINEALIDAD
from statsmodels.stats.outliers_influence import variance_inflation_factor

#VIF TEST
x = DATA[["T_Desempleo" , "Var_PBI"  , "T_Colocaciones" ]]


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
