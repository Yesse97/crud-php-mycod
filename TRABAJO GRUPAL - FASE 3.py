# Librerías
# ======================================================================================
import numpy as np
import pandas as pd
from io import StringIO
import contextlib
import re
import matplotlib.pyplot as plt
from pmdarima import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.stattools import ppca

plt.style.use('seaborn-v0_8-darkgrid')
pip install --upgrade statsmodels
pip install --upgrade pmdarima


#SUBIR EL ARCHIVO
path = r"C:/Users\Hp\Downloads\BALANZA.xlsx"
data = pd.read_excel(path)
data.columns
PERIODO  = data['PERIODO']
EXPORTACIONES = data['EXPORTACIONES']
IMPORTACIONES = data['IMPORTACIONES']


##### nombrarla fecha  #####
data['PERIODO'] = pd.to_datetime(data['PERIODO'])
data.set_index('PERIODO', inplace=True)

#Pruena informal 
# Crea un gráfico de líneas para las exportaciones
plt.figure(figsize=(10, 6))
plt.plot(PERIODO, EXPORTACIONES, label='Exportaciones', color='blue', marker='o')
plt.plot(PERIODO, IMPORTACIONES, label='Importaciones', color='red', marker='o')
plt.title('Series Temporales de Exportaciones e Importaciones')
plt.xlabel('Periodo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()
############ Prueba formal  ############
# Realizar el Test de Dickey-Fuller Aumentado (ADF)
result_adf1= adfuller(data['EXPORTACIONES'])
result_adf2= adfuller(data['IMPORTACIONES'])

# Mostrar resultados del test
#EXPORTACIONES
print('ADF Statistic:', result_adf1[0])
print('p-value:', result_adf1[1])
print('Critical Values:', result_adf1[4])
if result_adf1[1] <= 0.05:
    print("Rechazamos la hipótesis nula. La serie es estacionaria.")
else:
    print("No podemos rechazar la hipótesis nula. La serie no es estacionaria.")

#IMPORTACIONES
print('ADF Statistic:', result_adf2[0])
print('p-value:', result_adf2[1])
print('Critical Values:', result_adf2[4])
if result_adf2[1] <= 0.05:
    print("Rechazamos la hipótesis nula. La serie es estacionaria.")
else:
    print("No podemos rechazar la hipótesis nula. La serie no es estacionaria.")

###### Aplicamos diff

diff_export = data['EXPORTACIONES'].diff().dropna()
diff_import = data['IMPORTACIONES'].diff().dropna()

#Grafico para las exportaciones
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(diff_export, label='Exportaciones (Diferenciada)', color='blue', marker='o')
plt.title('Serie Temporal Diferenciada de Exportaciones')
plt.xlabel('Fecha')
plt.ylabel('Diferencia')
plt.legend()
plt.grid(True)

#Grafico para las importaciones
plt.subplot(2, 1, 2)
plt.plot(diff_import, label='Importaciones (Diferenciada)', color='green', marker='o')
plt.title('Serie Temporal Diferenciada de Importaciones')
plt.xlabel('Fecha')
plt.ylabel('Diferencia')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#######Prueba de fuller
result_adf1 = adfuller(data['EXPORTACIONES'])
result_adf2 = adfuller(data['IMPORTACIONES'])

#Exportaciones
print('ADF Statistic para Exportaciones (Original):', result_adf1[0])
print('p-value:', result_adf1[1])
print('Critical Values:', result_adf1[4])

if result_adf1[1] <= 0.05:
    print("Rechazamos la hipótesis nula. La serie de Exportaciones es estacionaria.")
else:
    print("No podemos rechazar la hipótesis nula. La serie de Exportaciones no es estacionaria.")

#Importaciones
print('ADF Statistic para Importaciones (Original):', result_adf2[0])
print('p-value:', result_adf2[1])
print('Critical Values:', result_adf2[4])

# Evaluar el p-value para tomar una decisión sobre la estacionariedad
if result_adf2[1] <= 0.05:
    print("Rechazamos la hipótesis nula. La serie de Importaciones es estacionaria.")
else:
    print("No podemos rechazar la hipótesis nula. La serie de Importaciones no es estacionaria.")

# Realizar el Test de Dickey-Fuller Aumentado (ADF) para las series diferenciadas
result_adf_diff1 = adfuller(diff_export)
result_adf_diff2 = adfuller(diff_import)

# Mostrar resultados del test para las series diferenciadas
#Exportaciones
print('\nADF Statistic para Exportaciones (Diferenciada):', result_adf_diff1[0])
print('p-value:', result_adf_diff1[1])
print('Critical Values:', result_adf_diff1[4])

if result_adf_diff1[1] <= 0.05:
    print("Rechazamos la hipótesis nula. La serie de Exportaciones (Diferenciada) es estacionaria.")
else:
    print("No podemos rechazar la hipótesis nula. La serie de Exportaciones (Diferenciada) no es estacionaria.")

#Importaciones
print('\nADF Statistic para Importaciones (Diferenciada):', result_adf_diff2[0])
print('p-value:', result_adf_diff2[1])
print('Critical Values:', result_adf_diff2[4])

if result_adf_diff2[1] <= 0.05:
    print("Rechazamos la hipótesis nula. La serie de Importaciones (Diferenciada) es estacionaria.")
else:
    print("No podemos rechazar la hipótesis nula. La serie de Importaciones (Diferenciada) no es estacionaria.")
    
#### Test de Pearson

result_pp_diff1 = adfuller(diff_export, autolag='AIC', regression='ct')
result_pp_diff2 = adfuller(diff_import, autolag='AIC', regression='ct')

# Mostrar resultados del test para las series diferenciadas
print('Phillips-Perron Test para Exportaciones (Diferenciada):')
print('Estadístico de prueba:', result_pp_diff1[0])
print('p-value:', result_pp_diff1[1])
print('Valores críticos:', result_pp_diff1[4])
print("Resultados:", "Estacionaria" if result_pp_diff1[1] <= 0.05 else "No estacionaria")
print("\n")

print('Phillips-Perron Test para Importaciones (Diferenciada):')
print('Estadístico de prueba:', result_pp_diff2[0])
print('p-value:', result_pp_diff2[1])
print('Valores críticos:', result_pp_diff2[4])
print("Resultados:", "Estacionaria" if result_pp_diff2[1] <= 0.05 else "No estacionaria")
print("\n")


###############Análisis de cointegración: 
    
 # Realizar el Test de Johansen
from statsmodels.tsa.vector_ar.vecm import coint_johansen
data_concatenated = pd.concat([diff_export, diff_import], axis=1)
result_johansen = coint_johansen(data_concatenated, det_order=0, k_ar_diff=1)
print('Estadístico de traza:', result_johansen.lr1)
print('Valores críticos (0.05):', result_johansen.cvt)
print('\nEstadístico de máximo autovalor:', result_johansen.lr2)
print('Valores críticos (0.05):', result_johansen.cvm)

# estadístico de traza
for i in range(len(result_johansen.lr1)):
    if result_johansen.lr1[i] > result_johansen.cvt[:, 2][i]:
        print(f"\nVector {i + 1}: El estadístico de traza es {result_johansen.lr1[i]}, lo cual es mayor que el valor crítico. Se rechaza la hipótesis nula de no cointegración.")
    else:
        print(f"\nVector {i + 1}: El estadístico de traza es {result_johansen.lr1[i]}, lo cual no es mayor que el valor crítico. No hay suficiente evidencia para rechazar la hipótesis nula.")
# estadístico de máximo autovalor
for i in range(len(result_johansen.lr2)):
    if result_johansen.lr2[i] > result_johansen.cvm[:, 2][i]:
        print(f"\nVector {i + 1}: El estadístico de máximo autovalor es {result_johansen.lr2[i]}, lo cual es mayor que el valor crítico. Se rechaza la hipótesis nula de no cointegración.")
    else:
        print(f"\nVector {i + 1}: El estadístico de máximo autovalor es {result_johansen.lr2[i]}, lo cual no es mayor que el valor crítico. No hay suficiente evidencia para rechazar la hipótesis nula.")
             
# Concat
data_diff = pd.concat([diff_export, diff_import], axis=1)
data_diff.columns = ['EXPORTACIONES', 'IMPORTACIONES']
print(data_diff.head())

from statsmodels.tsa.vector_ar.vecm import select_order

# Determinación del número de rezagos utilizando el criterio BIC
max_lag = 5  # Puedes ajustar según tus necesidades
result_bic = select_order(data, max_lag, deterministic='co')
best_lag_bic = result_bic.selected_orders['bic']

# Ahora puedes utilizar best_lag_bic en tu código
best_lag_bic = int(best_lag_bic)
print(f"Mejor número de rezagos según BIC: {best_lag_bic}")
    
#### modelos var

## VAR- 
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import select_order
model = VAR(data_diff)
results = model.fit(1)  # Ajustar con 1 rezago
print(results.summary())
    
## VAR DE ORDEN SUPERIOR
num_rezagos = 2
model = VAR(data_diff)
results = model.fit(num_rezagos)
print(results.summary())


# MODELO  VAR y modelos univariantes
pip install pmdarima
from pmdarima import auto_arima
from pmdarima import ARIMA

# AUTOARIMA - EXPORT ##
best_aic = float('inf')
best_order = None

#### AUTO ARIMA EXPORTACIONES ###

for p in range(5):  # Ajusta según tus necesidades
    for q in range(5):  # Ajusta según tus necesidades
        model = ARIMA(diff_export, order=(p, 1, q))
        results = model.fit()
        aic = results.aic
        if aic < best_aic:
            best_aic = aic
            best_order = (p, 1, q)

print(f"Mejor modelo ARIMA: {best_order} con AIC: {best_aic}")

# Mejor modelo ARIMA: (3, 1, 4) con AIC: 2072.234786246378

# AUTOARIMA - IMPORT ##
best_aic = float('inf')
best_order = None
#### AUTO ARIMA  IMPORTACIONES ###
for p in range(5):  # Ajusta según tus necesidades
    for q in range(5):  # Ajusta según tus necesidades
        model = ARIMA(diff_import, order=(p, 1, q))
        results = model.fit()
        aic = results.aic
        if aic < best_aic:
            best_aic = aic
            best_order = (p, 1, q)
print(f"Mejor modelo ARIMA: {best_order} con AIC: {best_aic}")

# Mejor modelo ARIMA: (3, 1, 1) con AIC: 1968.350303664923

# MODELO  VAR 
from statsmodels.tsa.arima.model import ARIMA
# Ajustar un modelo VAR
num_rezagos_var = 1
model_var = VAR(data_diff)
results_var = model_var.fit(num_rezagos_var)

# Imprimir un resumen del modelo VAR
print("Resumen del modelo VAR:")
print(results_var.summary())

# Ajustar 
order_arima_export = (3, 1, 4) 
model_arima_export = ARIMA(data_diff['EXPORTACIONES'], order=order_arima_export)
results_arima_export = model_arima_export.fit()
order_arima_import = (3, 1, 1)  # Ajusta los órdenes según tus necesidades
model_arima_import = ARIMA(data_diff['IMPORTACIONES'], order=order_arima_import)
results_arima_import = model_arima_import.fit()

print("\nResumen del modelo ARIMA para EXPORTACIONES:")
print(results_arima_export.summary())

print("\nResumen del modelo ARIMA para IMPORTACIONES:")
print(results_arima_import.summary())

## MODELO  Estimacion y contrastacion de hipotesis
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import select_order

# Determinación del número de rezagos utilizando el criterio BIC
max_lag = 5  # Ajusta según tus necesidades
result_bic = select_order(data_diff, max_lag, deterministic='co')
best_lag_bic = int(result_bic.bic)

# Ajustar un modelo VAR con el número óptimo de rezagos
model_var = VAR(data_diff)
results_var = model_var.fit(best_lag_bic)
print(results_var.summary())

# Contrastación de restricciones 
#Utilizamos el Metodo de prueba de causalidad de Granger,
causality_test = results_var.test_causality(['EXPORTACIONES'], ['IMPORTACIONES'], kind='f')
print("\nContraste de causalidad:")
print(causality_test)
#La hipótesis nula establece que la variable 
#"IMPORTACIONES" no tiene un efecto causal significativo en la variable "EXPORTACIONES"

######## MODELO  identificacion de un modelo var
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import select_order, coint_johansen

max_lag = 5  
result_bic = select_order(data_diff, max_lag, deterministic='co')
best_lag_bic = result_bic.selected_orders['bic']
print(f"Número óptimo de rezagos según BIC: {best_lag_bic}")

model_var = VAR(data_diff)
results_var = model_var.fit(best_lag_bic)

#Prueba de Johansen para cointegración:
cointegration_test = coint_johansen(results_var.resid, det_order=0, k_ar_diff=best_lag_bic - 1)
print("\nPrueba de Johansen para cointegración:")
print(cointegration_test.cvt)
print("\nResultados del modelo VAR:")
print(results_var.summary())

#Rchaza la Hiposis Nula por ser mayor la primera fila 
#El modelo para estimar la seie es o 1 0 segun el modelo que estamo siguendo 

## MODELO Representacion MA de un modelo VAR

max_lag = 5  # Puedes ajustar según tus necesidades
result_bic = select_order(data_diff, max_lag, deterministic='co')
best_lag_bic = result_bic.selected_orders['bic']

model_var = VAR(data_diff)
results_var = model_var.fit(best_lag_bic)

# Obtener la representación MA
ma_representation = results_var.ma_rep(maxn=best_lag_bic)
print("\nRepresentación MA del modelo VAR:")
print(ma_representation)
#No están directamente influenciadas por los errores de observaciones pasadas
#Los demás elementos de la matriz son 0, indicando que no hay relaciones 
#directas entre las variables actuales y los errores pasados


# MODELO respuesta al impulso
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.vector_ar.irf import IRAnalysis
max_lag = 5  # Puedes ajustar según tus necesidades
result_bic = select_order(data_diff, max_lag, deterministic='co')
best_lag_bic = result_bic.selected_orders['bic']

model_var = VAR(data_diff)
results_var = model_var.fit(best_lag_bic)
irf = results_var.irf(5)  
print(irf)
irf.plot()

#CUANTO TIEMPO DURARA EL EFECTO DE ...... 
#SI EXPORTAMOS MAS LA BALANZA COMERCIAL SUBE 

#####VALIDACION DEL MODELO
#export
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Supongamos que 'data' es tu DataFrame con las series temporales
# Asegúrate de que 'data' contenga ambas series 'EXPORTACIONES' y 'IMPORTACIONES'

# Dividir los datos en entrenamiento y prueba
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:]

model = VAR(train)
order = 2  # Ajusta el orden del modelo según tus necesidades
results_var = model.fit(order)

forecast_steps = len(test)
forecast = results_var.forecast(train.values[-order:], steps=forecast_steps)

mse_exportaciones = mean_squared_error(test['EXPORTACIONES'], forecast[:, 0])
mse_importaciones = mean_squared_error(test['IMPORTACIONES'], forecast[:, 1])

print(f'MSE para EXPORTACIONES: {mse_exportaciones}')
print(f'MSE para IMPORTACIONES: {mse_importaciones}')

residuals = test - forecast
for column in residuals.columns:
    result_df = adfuller(residuals[column])
    print(f"Prueba de Dickey-Fuller para los residuos de {column}:")
    print(f"Estadístico de prueba: {result_df[0]}")
    print(f'P-valor: {result_df[1]}')
    print(f'Valores críticos: {result_df[4]}')
    print("Resultados:", "Estacionaria" if result_df[1] <= 0.05 else "No estacionaria")
    print("\n")

##### FUNCION IMPULSO

# Calcular las funciones de respuesta al impulso (IRF)
pip install --upgrade statsmodels

import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import select_order
max_lag = 5  # Puedes ajustar según tus necesidades
result_bic = select_order(data, max_lag, deterministic='co')
best_lag_bic = result_bic.selected_orders['bic']
model_var = VAR(data)
results_var = model_var.fit(best_lag_bic)
irf = results_var.irf(10)  # Puedes ajustar el número de pasos según tus necesidades
model_var = VAR(data)
results_var = model_var.fit(best_lag_bic)
irf = results_var.irf(10)  
results_var.plot_forecast(10)  
plt.show()

#############DESCOMPOSICION DE VARIANZA
variance_decomposition = results_var.fevd()
plt.figure(figsize=(12, 6))
variance_decomposition.plot()
plt.title('Descomposición de la Varianza')
plt.show()

############# DESCOMPOSICION HISTORICA
max_lag = 5  # Puedes ajustar según tus necesidades
result_bic = select_order(data, max_lag, deterministic='co')
best_lag_bic = result_bic.selected_orders['bic']
model_var = VAR(data)
results_var = model_var.fit(best_lag_bic)
decomposition = results_var.plot_forecast(steps=10)  # 10 es el número de pasos hacia adelante
# Visualizar los resultados
plt.figure(figsize=(12, 8))
decomposition.plot()
plt.suptitle('Descomposición Histórica')
plt.show()

############ PRONOSTICO
max_lag = 5 
result_bic = select_order(data, max_lag, deterministic='co')
best_lag_bic = result_bic.selected_orders['bic']
model_var = VAR(data)
results_var = model_var.fit(best_lag_bic)
# Realizar pronósticos para 12 periodos hacia adelante
forecast_steps = 12
forecast = results_var.forecast(data.values[-best_lag_bic:], steps=forecast_steps)
forecast_index = pd.date_range(data.index[-1], periods=forecast_steps + 1, freq=data.index.freq)
forecast_df = pd.DataFrame(forecast, index=forecast_index[1:], columns=data.columns)
plt.figure(figsize=(20, 12))
plt.plot(data.index, data['EXPORTACIONES'], label='Exportaciones (Histórico)', color='blue', marker='o')
plt.plot(data.index, data['IMPORTACIONES'], label='Importaciones (Histórico)', color='red', marker='o')
plt.plot(forecast_df.index, forecast_df['EXPORTACIONES'], label='Exportaciones (Pronóstico)', linestyle='dashed', color='blue')
plt.plot(forecast_df.index, forecast_df['IMPORTACIONES'], label='Importaciones (Pronóstico)', linestyle='dashed', color='red')
plt.title('Pronósticos de Exportaciones e Importaciones')
plt.xlabel('Periodo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()

## VALIDACION DE MODELOS

train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:]
model = VAR(train)
order = 2 
forecast_steps = len(test)
forecast = results.forecast(train.values[-order:], steps=forecast_steps)
test_exportaciones = test['EXPORTACIONES'].values
test_importaciones = test['IMPORTACIONES'].values
mse_exportaciones = mean_squared_error(test_exportaciones, forecast[:, 0])
mse_importaciones = mean_squared_error(test_importaciones, forecast[:, 1])
print(f'MSE para EXPORTACIONES: {mse_exportaciones}')
print(f'MSE para IMPORTACIONES: {mse_importaciones}')

