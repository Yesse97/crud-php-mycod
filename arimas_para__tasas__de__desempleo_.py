
#Caso 1: Modelo ARIMA - Box Jenkins - Parte 1



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#SUBIR EL ARCHIVO
path = r"C:/Users\Hp\Downloads\datos_tarea1 (3).xlsx"
train = pd.read_excel(path)
print(train)
train.columns

"""**Setear el dataframe**"""

train = train.set_index('tiempo')
train

# Crear una copia de la columna 'INFLA_CHILE' y establecer 'tiempo' como índice
INFLA_CHILE = train[['INFLA_CHILE']].copy()
INFLA_CHILE.index = pd.to_datetime(train.index, format='%YM%m')  # Convertir el índice a formato de fecha

# Imprimir el nuevo DataFrame con formato de fechas
print(INFLA_CHILE)

INFLA_CHILE

"""Análisis de la Estacionariedad: Prueba **informal**   CHILE"""

# Gráfico
# ======================================================================================
fig, ax=plt.subplots(figsize=(10, 6))
#datos_train.plot(ax=ax, label='train')
INFLA_CHILE.plot(ax=ax, label='inflación de Chile')
ax.set_title('Serie Temporal de la inflación de Chile')
ax.legend();

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


# Calcular ACF y PACF
#AR ES ACF
#MA ES PACF
lags = 36  # Puedes ajustar el número de lags según tus necesidades entonces aqui es donde puedo ver un mejor grafico 
acf_values = acf(INFLA_CHILE, nlags=lags)
pacf_values = pacf(INFLA_CHILE, nlags=lags)

# Plot ACF
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_acf(INFLA_CHILE, lags=lags, ax=plt.gca())
plt.title('Autocorrelagrama (ACF)')

# Plot PACF
plt.subplot(1, 2, 2)
plot_pacf(INFLA_CHILE, lags=lags, ax=plt.gca())
plt.title('Función de Autocorrelación Parcial (PACF)')

plt.show()
# el componente AR (autorregresivo comportamiento pasado)
# el compnente MA (riesgo y volatilidad)

#AR ES ACF (se ve el degradado que va poco poco disminuyendo , quiere decir que tiene mayor componente MA osea rezagos pasado )(autorregresivo son los periodos pasados , que vendria a ser rezagos)
#MA ES PACF (como las vanditas sobresalen entonces se puede ver que es o no estacional , en este caso es estacional)(componente de riesgo volatilidad , datos fuera de lo comun)

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Calcular ACF y PACF
lags = 36  # Ajustar el número de lags según tus necesidades
acf_values = acf(INFLA_CHILE, nlags=lags)
pacf_values = pacf(INFLA_CHILE, nlags=lags)

# Plot ACF para lags 12, 24 y 36
plt.figure(figsize=(18, 4))
plt.subplot(1, 3, 1)
plot_acf(INFLA_CHILE, lags=12, ax=plt.gca())
plt.title('Autocorrelagrama (ACF) - Lag 12')

plt.subplot(1, 3, 2)
plot_acf(INFLA_CHILE, lags=24, ax=plt.gca())
plt.title('Autocorrelagrama (ACF) - Lag 24')

plt.subplot(1, 3, 3)
plot_acf(INFLA_CHILE, lags=36, ax=plt.gca())
plt.title('Autocorrelagrama (ACF) - Lag 36')

plt.show()

# Plot PACF para lags 12, 24 y 36
plt.figure(figsize=(18, 4))
plt.subplot(1, 3, 1)
plot_pacf(INFLA_CHILE, lags=12, ax=plt.gca())
plt.title('Función de Autocorrelación Parcial (PACF) - Lag 12')

plt.subplot(1, 3, 2)
plot_pacf(INFLA_CHILE, lags=24, ax=plt.gca())
plt.title('Función de Autocorrelación Parcial (PACF) - Lag 24')

plt.subplot(1, 3, 3)
plot_pacf(INFLA_CHILE, lags=36, ax=plt.gca())
plt.title('Función de Autocorrelación Parcial (PACF) - Lag 36')

plt.show()

import matplotlib.pyplot as plt

# Supongamos que tienes tu serie temporal llamada 'serie_temporal'
# serie_temporal = ...

# Graficar la serie temporal
plt.figure(figsize=(10, 6))
plt.plot(INFLA_CHILE.index, INFLA_CHILE['INFLA_CHILE'], label='INFLA_CHILE', color='blue')
plt.title('Gráfico de Serie Temporal')
plt.xlabel('Fecha')
plt.ylabel('INFLA_CHILE')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Supongamos que tienes tu DataFrame llamado 'serie_temporal' con 'tiempo' como índice
# serie_temporal = ...

# Extraer la columna 'INFLA_CHILE'
data = INFLA_CHILE['INFLA_CHILE']

# Calcular ángulos para cada punto en la serie temporal
angles = np.linspace(0, 2 * np.pi, len(data), endpoint=False)

# Crear el gráfico polar
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Graficar los datos en coordenadas polares
ax.plot(angles, data, marker='o', linestyle='-', linewidth=2)

# Ajustar la dirección del cero al inicio
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Etiquetar los ángulos
ax.set_xticks(angles)
ax.set_xticklabels(INFLA_CHILE.index)

# Mostrar el gráfico
plt.title('Gráfico Polar - Patrones Estacionales')
plt.show()

"""**análisis de la Estacionariedad: Pruebas Formales**

**Test de Dickey-Fuller Aumentado (ADF)**
"""

from statsmodels.tsa.stattools import adfuller

# Realizar el Test de Dickey-Fuller Aumentado (ADF)
result_adf = adfuller(INFLA_CHILE['INFLA_CHILE'])

# Mostrar resultados del test
print('ADF Statistic:', result_adf[0])
print('p-value:', result_adf[1])
print('Critical Values:', result_adf[4])

# Evaluar el p-value para tomar una decisión sobre la estacionariedad
if result_adf[1] <= 0.05:
    print("Rechazamos la hipótesis nula. La serie es estacionaria.")
else:
    print("No podemos rechazar la hipótesis nula. La serie no es estacionaria.")

# Evaluar el p-value para tomar una decisión sobre la estacionariedad
# H0 = LA SERIE NO ES ESTACIONARIA : TIENE RAIZ UNITARIA (esta serie que no es etacionaria)
# H1 = LA SERIE ES ESTACIONARIA :NO TIENE RAIZ UNITARIA

"""Test Phillips-**Perron**"""

!pip install arch

import pandas as pd
from arch.unitroot import PhillipsPerron

# Supongamos que ya tienes tu DataFrame llamado 'INFLA_CHILE' con la columna 'DEMANDA'
# INFLA_CHILE = ...

# Realizar el Test de Phillips-Perron
pp = PhillipsPerron(INFLA_CHILE['INFLA_CHILE'])

# Mostrar resultados del test
print('Phillips-Perron Statistic:', pp.stat)
print('p-value:', pp.pvalue)
print('Lags:', pp.lags)
print('Trend:', pp.trend)

# Evaluar las hipótesis y mostrar la conclusión
if pp.pvalue <= 0.05:
    print("Conclusión: Rechazamos la hipótesis nula. La serie es estacionaria.")
else:
    print("Conclusión: No podemos rechazar la hipótesis nula. La serie no es estacionaria.")

# Evaluar el p-value para tomar una decisión sobre la estacionariedad
# H0 = LA SERIE NO ES ESTACIONARIA : TIENE RAIZ UNITARIA
# H1 = LA SERIE ES ESTACIONARIA :NO TIENE RAIZ UNITARIA
#por lo tanto aceptamos la hipotesis nula porque la serie no e

"""**Prueba informal para la Primera Diferencia:**"""

#---------- Prueba informal para la Primera Diferencia:
# Generar la primera diferencia:
INFLA_CHILE_diff_1 = INFLA_CHILE.diff().dropna()
# Gráfico:
fig, ax=plt.subplots(figsize=(7, 3))
INFLA_CHILE_diff_1.plot(ax=ax, label='Primera Diferencia ')
ax.set_title('.............. en su Primera Diferencia')
ax.legend()

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


# Calcular ACF y PACF
lags = 12  # Puedes ajustar el número de lags según tus necesidades
acf_values = acf(INFLA_CHILE_diff_1 , nlags=lags)
pacf_values = pacf(INFLA_CHILE_diff_1 , nlags=lags)

# Plot ACF
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_acf(INFLA_CHILE_diff_1, lags=lags, ax=plt.gca())
plt.title('Autocorrelagrama (ACF)')

# Plot PACF
plt.subplot(1, 2, 2)
plot_pacf(INFLA_CHILE_diff_1, lags=lags, ax=plt.gca())
plt.title('Función de Autocorrelación Parcial (PACF)')

plt.show()
# el componente AR (autorregresivo comportamiento pasado)
# el compnente MA (riesgo y volatilidad)

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Calcular ACF y PACF
lags = 36  # Ajustar el número de lags según tus necesidades
acf_values = acf(INFLA_CHILE_diff_1, nlags=lags)
pacf_values = pacf(INFLA_CHILE_diff_1, nlags=lags)

# Plot ACF para lags 12, 24 y 36
plt.figure(figsize=(18, 4))
plt.subplot(1, 3, 1)
plot_acf(INFLA_CHILE_diff_1, lags=12, ax=plt.gca())
plt.title('Autocorrelagrama (ACF) - Lag 12')

plt.subplot(1, 3, 2)
plot_acf(INFLA_CHILE_diff_1, lags=24, ax=plt.gca())
plt.title('Autocorrelagrama (ACF) - Lag 24')

plt.subplot(1, 3, 3)
plot_acf(INFLA_CHILE_diff_1, lags=36, ax=plt.gca())
plt.title('Autocorrelagrama (ACF) - Lag 36')

plt.show()

# Plot PACF para lags 12, 24 y 36
plt.figure(figsize=(18, 4))
plt.subplot(1, 3, 1)
plot_pacf(INFLA_CHILE_diff_1, lags=12, ax=plt.gca())
plt.title('Función de Autocorrelación Parcial (PACF) - Lag 12')

plt.subplot(1, 3, 2)
plot_pacf(INFLA_CHILE_diff_1, lags=24, ax=plt.gca())
plt.title('Función de Autocorrelación Parcial (PACF) - Lag 24')

plt.subplot(1, 3, 3)
plot_pacf(INFLA_CHILE_diff_1, lags=36, ax=plt.gca())
plt.title('Función de Autocorrelación Parcial (PACF) - Lag 36')

plt.show()

from statsmodels.tsa.stattools import adfuller

# Aplicar el Test de Dickey-Fuller Aumentado a la serie con primera diferencia
result_diff = adfuller(INFLA_CHILE_diff_1, autolag='AIC')

# Interpretar los resultados
print(f'ADF Statistic: {result_diff[0]}')
print(f'p-value: {result_diff[1]}')
print(f'Critical Values: {result_diff[4]}')

# Evaluar el p-value para tomar una decisión sobre la estacionariedad
if result_diff[1] <= 0.05:
    print("Conclusión: Se rechaza la hipótesis nula. La serie con primera diferencia es estacionaria.")
else:
    print("Conclusión: No se puede rechazar la hipótesis nula. La serie con primera diferencia puede tener una raíz unitaria y ser no estacionaria.")

from arch.unitroot import PhillipsPerron

# Aplicar el Test de Phillips-Perron a la serie con primera diferencia
pp_test = PhillipsPerron(INFLA_CHILE_diff_1)
pp_result = pp_test

# Interpretar los resultados
print(pp_result.summary().as_text())

"""#se aplican hasta 3 diferencias
MODELAMIENTO DEL ARIMA

#NOTA DE LOS COMPONENETES DEL MODELO :

*   modelo de corto plazo
#AR MA
#AR = AUTORREGRESIVO
#MA = MEDIA MOVIL
#SAR = AUTORREGRESIVO ESTACIONAL
#SMA = MEDIA MOVIL ESTACIONAL
#LA I REPRESENTA A LAS DIFERENCIAS

**CON esto hallo el mejor modelo ARIMA pero no considera la normalidad, pero los p valor son significativos**
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import numpy as np

#existen 4 criterios de información
#AIC (ESTE SE ELIGE)
#HANAQUIEN
#SHUAS
#se le puede agregar estacionalidad , ya sea al AR como al MA
#puede ver o suceder que se puede encontrar el ARMA pero si es que la data ya fuera estacionaria
#la estacionalidad se puede dara tanto en el AR , MA O ARMA O ARIMA


# Supongamos que ya tienes tu DataFrame 'INFLA_CHILE' con 'tiempo' como índice
# INFLA_CHILE = ...

# Aplicar la prueba ADF para verificar estacionariedad
result_adf = adfuller(INFLA_CHILE['INFLA_CHILE'])
if result_adf[1] > 0.05:
    print("La serie no es estacionaria. Aplicar diferenciación.")
    INFLA_CHILE_diff = INFLA_CHILE.diff().dropna()
else:
    print("La serie es estacionaria.")

best_aic = np.inf
best_order = None

for p in range(5):
    for d in range(2):
        for q in range(5):
            try:
                model = ARIMA(INFLA_CHILE_diff, order=(p, d, q))
                result = model.fit()

                # Verificar significancia de los coeficientes
                if all(result.pvalues < 0.05):
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
            except:
                continue

print(f"Mejor modelo ARIMA: {best_order} con AIC: {best_aic}")

"""**CON esto hallo el mejor modelo ARIMA pero si considera la normalidad y los p valor son significativos**"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
import numpy as np

# Supongamos que ya tienes tu DataFrame 'INFLA_CHILE' con 'tiempo' como índice
# INFLA_CHILE = ...

# Aplicar la prueba ADF para verificar estacionariedad
result_adf = adfuller(INFLA_CHILE['INFLA_CHILE'])
if result_adf[1] > 0.05:
    print("La serie no es estacionaria. Aplicar diferenciación.")
    INFLA_CHILE = INFLA_CHILE
else:
    print("La serie es estacionaria.")

best_aic = np.inf
best_order = None

for p in range(5):
    for d in range(2):
        for q in range(5):
            try:
                model = ARIMA(INFLA_CHILE_diff, order=(p, d, q))
                result = model.fit()

                # Verificar significancia de los coeficientes
                if all(result.pvalues < 0.05):

                    # Verificar heterocedasticidad
                    residuals_squared = result.resid**2
                    lb_test_stat, lb_pvalue, _, _ = het_arch(residuals_squared)

                    if lb_pvalue > 0.05:  # No hay evidencia de heterocedasticidad
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_order = (p, d, q)

            except:
                continue

print(f"Mejor modelo ARIMA: {best_order} con AIC: {best_aic}")

"""**Step 2: Determine ARIMA models parameters p, q,d**"""

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(INFLA_CHILE_diff_1, order=(2,1,3))
model_fit = model.fit()
print(model_fit.summary())

!pip install pmdarima

import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

# Supongamos que ya tienes tu DataFrame 'CANA_1' con 'tiempo' como índice
# CANA_1 = ...

# Aplicar la primera diferencia a la serie para hacerla estacionaria
INFLA_CHILE_diff = INFLA_CHILE.diff().dropna()

# Aplicar la prueba ADF para verificar estacionariedad
result_adf = adfuller(INFLA_CHILE_diff['INFLA_CHILE'])
if result_adf[1] > 0.05:
    print("La serie no es estacionaria después de la primera diferencia. Aplicar diferenciación adicional si es necesario.")
else:
    print("La serie es estacionaria después de la primera diferencia.")

# Aplicar la búsqueda automática de ARIMA en la serie diferenciada
model = auto_arima(INFLA_CHILE_diff, suppress_warnings=True, trace=True)

# Mostrar el mejor modelo ARIMA y sus coeficientes
print(model.summary())

import pmdarima as pm
auto_arima = pm.auto_arima(INFLA_CHILE, stepwise=False, seasonal=False)
auto_arima

"""**INFLA_CHILE(t) =-1.4241*INFLA_CHILE(t-1) -0.8059*INFLA_CHILE(t-2) +0.8871*INFLA_CHILE.s(t-1).....+0.2181*e.s(t-1)**

**el test de normalidad jarque bera , criterio de decisión: si p > 0.05 distribucion normal**

1.   h0 :la muestra proviene de una distribución normal
2.   h1 :la muestra no proviene de una distribución normal


1.   Jarque-Bera 1.58
2.   Prob(JB):0.45 por lo que si pasa ya que es mayor al 0.05
considerdad el modelo autoarima como guia , porque no se da cuenta de la

1.   considerdad el modelo autoarima como guia , porque no se da cuenta de la normalidad osea la distribucion normal de los datos , si o si tiene que pasar el test de normalidad
"""

# Suponiendo que ya tienes ajustado tu modelo ARIMA y lo guardaste en 'model_fit'
# model_fit = ARIMA(INFLA_CHILE_diff_1, order=(4, 1, 3)).fit()

# Obtener los residuos
residuals = model_fit.resid
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

# Plot autocorrelograma de los residuos
plt.figure(figsize=(12, 4))
plot_acf(residuals, lags=20, ax=plt.gca())
plt.title('Autocorrelograma de Residuos')
plt.show()

#!pip install
!pip install arch

"""**prueba de estacionariedad PhillipsPerron**"""

from arch.unitroot import PhillipsPerron
import arch.data.frenchdata
from statsmodels.tsa.arima.model import ARIMA

# Supongamos que ya tienes ajustado tu modelo ARIMA y lo guardaste en 'model_fit'
# model_fit = ARIMA(INFLA_CHILE_diff_1, order=(4, 1, 3)).fit()

# Obtener los residuos del modelo ARIMA
residuals = model_fit.resid

# Realizar el Test de Phillips-Perron en los residuos
pp_test = PhillipsPerron(residuals)
pp_result = pp_test
print('Phillips-Perron Statistic (residuos):', pp_result)
print('p-value (residuos):', pp_test.pvalue)

# Evaluar el p-value para tomar una decisión sobre la estacionariedad de los residuos
if pp_test.pvalue <= 0.05:
    print("Rechazamos la hipótesis nula. Los residuos son estacionarios (Phillips-Perron).")
else:
    print("No podemos rechazar la hipótesis nula. Los residuos no son estacionarios (Phillips-Perron).")

"""**prueba de estacionariedad adfuller**"""

from statsmodels.tsa.stattools import adfuller

# Obtener los residuos del modelo ARIMA
residuals = model_fit.resid

# Realizar el Test de Dickey-Fuller Aumentado (ADF) en los residuos
result_adf_residuals = adfuller(residuals)
print('ADF Statistic (residuos):', result_adf_residuals[0])
print('p-value (residuos):', result_adf_residuals[1])
print('Critical Values (residuos):', result_adf_residuals[4])

# Evaluar el p-value para tomar una decisión sobre la estacionariedad de los residuos
if result_adf_residuals[1] <= 0.05:
    print("Rechazamos la hipótesis nula. Los residuos son estacionarios.")
else:
    print("No podemos rechazar la hipótesis nula. Los residuos no son estacionarios.")

"""**prueba de normalidad > 0.05 mayor**




"""

from scipy.stats import anderson

# Obtener los residuos del modelo ARIMA
residuals = model_fit.resid

# Realizar el test de normalidad de Anderson-Darling
result_anderson = anderson(residuals)

# Imprimir los resultados del test
print('Estadístico de Anderson-Darling:', result_anderson.statistic)
print('Valores críticos:', result_anderson.critical_values)
print('Niveles de significancia:', result_anderson.significance_level)

# Evaluar la hipótesis nula (H0: los residuos siguen una distribución normal)
if result_anderson.statistic < result_anderson.critical_values[2]:
    print("No podemos rechazar la hipótesis nula. Los residuos siguen una distribución normal.")
else:
    print("Rechazamos la hipótesis nula. Los residuos no siguen una distribución normal.")

"""**prueba de normalidad > 0.05 mayor**"""

from scipy.stats import shapiro
from statsmodels.tsa.arima.model import ARIMA

# Supongamos que ya tienes ajustado tu modelo ARIMA y lo guardaste en 'model_fit'
# model_fit = ARIMA(INFLA_CHILE_diff_1, order=(4, 1, 3)).fit()

# Obtener los residuos del modelo ARIMA
residuals = model_fit.resid

# Realizar el Test de Shapiro-Wilk en los residuos
shapiro_stat, shapiro_pvalue = shapiro(residuals)
print('Shapiro-Wilk Statistic (residuos):', shapiro_stat)
print('p-value (residuos):', shapiro_pvalue)

# Evaluar el p-value para tomar una decisión sobre la normalidad de los residuos
if shapiro_pvalue <= 0.05:
    print("Rechazamos la hipótesis nula. Los residuos no siguen una distribución normal.")
else:
    print("No podemos rechazar la hipótesis nula. Los residuos siguen una distribución normal.")

"""**prueba de heterocedasticidad > 0.05 mayor**"""

from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.arima.model import ARIMA

# Supongamos que ya tienes ajustado tu modelo ARIMA y lo guardaste en 'model_fit'
# model_fit = ARIMA(INFLA_CHILE_diff_1, order=(4, 1, 3)).fit()

# Obtener los residuos al cuadrado del modelo ARIMA
residuals_squared = model_fit.resid**2

# Realizar el test de Ljung-Box para evaluar la heterocedasticidad
lb_test_stat, lb_pvalue, _, _ = het_arch(residuals_squared)
print('Ljung-Box Test Statistic (residuos al cuadrado):', lb_test_stat)
print('p-value (residuos al cuadrado):', lb_pvalue)

# Evaluar el p-value para tomar una decisión sobre la heterocedasticidad
if lb_pvalue <= 0.05:
    print("Rechazamos la hipótesis nula. Hay evidencia de heterocedasticidad.")
else:
    print("No podemos rechazar la hipótesis nula. No hay evidencia suficiente de heterocedasticidad.")

import pandas as pd
import matplotlib.pyplot as plt

# Hacer el pronóstico
forecast_steps = 8  # ajusta según tus necesidades
forecast = model_fit.get_forecast(steps=forecast_steps)

# Obtener el intervalo de confianza con un nivel del 95%
forecast_conf_int = forecast.conf_int(alpha=0.05)

# Visualizar el pronóstico y el intervalo de confianza en un gráfico
plt.figure(figsize=(10, 6))

# Plot de la serie temporal original
plt.plot(INFLA_CHILE_diff_1.index, INFLA_CHILE_diff_1, label='Serie Diferenciada', color='blue')

# Plot del pronóstico
plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Pronóstico', color='red')

# Rellenar el área entre los límites del intervalo de confianza
plt.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='red', alpha=0.2, label='Intervalo de Confianza (95%)')

plt.title('Pronóstico con Intervalo de Confianza (95%)')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend()

# Ajustes adicionales para mejorar la visibilidad
plt.ylim([-1.5, 1.5])  # Ajustar los límites del eje y según sea necesario
plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para una mejor visualización

plt.tight_layout()  # Ajustar el diseño para evitar superposiciones
plt.show()

# Hacer el pronóstico
forecast_steps = 8  # ajusta según tus necesidades
forecast = model_fit.get_forecast(steps=forecast_steps)

# Obtener el intervalo de confianza con un nivel del 95%
forecast_conf_int = forecast.conf_int(alpha=0.05)

# Visualizar el pronóstico y el intervalo de confianza
print("Pronóstico:")
print(forecast.predicted_mean)

# Puedes convertir las fechas a un formato más legible si es necesario
forecast_dates = pd.date_range(start=INFLA_CHILE.index[-1], periods=forecast_steps + 1, freq='M')[1:]
forecast_index = forecast_dates.strftime('%Y-%m')
forecast.predicted_mean.index = forecast_index

print("\nIntervalo de confianza (95%):")
print(forecast_conf_int)

forecast_test = model_fit.forecast(len(df_test))

df['forecast_manual'] = [None]*len(df_train) + list(forecast_test)

df.plot()

# Hacer el pronóstico
forecast_steps = 12  # ajusta según tus necesidades
forecast = model_fit.get_forecast(steps=forecast_steps)

# Obtener el intervalo de confianza
forecast_conf_int = forecast.conf_int()

# Visualizar el pronóstico y el intervalo de confianza
print("Pronóstico:")
print(forecast.predicted_mean)
print("\nIntervalo de confianza:")
print(forecast_conf_int)



# Supongamos que ya tienes tu DataFrame 'INFLA_CHILE' con 'tiempo' como índice
# INFLA_CHILE = ...

# Asumiendo que ya obtuviste el mejor orden (p, d, q)
best_order = (p, d, q)


# Entrenar el modelo ARIMA con el mejor orden
model = ARIMA(INFLA_CHILE_diff_1, order=best_order)
result = model.fit()

# Hacer el pronóstico
forecast_steps = 12  # puedes ajustar el número de pasos para el pronóstico
forecast = result.get_forecast(steps=forecast_steps)

# Obtener el intervalo de confianza
forecast_conf_int = forecast.conf_int()

# Visualizar el pronóstico y el intervalo de confianza
print(forecast.predicted_mean)
print(forecast_conf_int)

pip install pmdarima

pip install statsmodels

pip install statsmodels

#!pip install
!pip install arch
import arch.data.frenchdata
from arch.unitroot import PhillipsPerron

!pip install pmdarima
!pip install statsmodels
!pip install skforecast
#!pip install

# Bibliotecas
# ======================================================================================

# básicas
import numpy as np
import pandas as pd
from io import StringIO
import contextlib
import re
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

# pmdarima
from pmdarima import ARIMA
from pmdarima import auto_arima

# statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# skforecast
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import backtesting_sarimax
from skforecast.model_selection_sarimax import grid_search_sarimax
from sklearn.metrics import mean_absolute_error

import warnings

"""#Autocorrelación y Autocorrelación Parcial

Función de autocorrelación (ACF)

La ACF calcula la correlación entre una serie temporal y sus valores retardados (lags). En el contexto de la modelización ARIMA, una caída brusca de la ACF después de unos pocos retardos indica que los datos tienen un orden autorregresivo finito. El retardo en el que cae la ACF proporciona una estimación del valor de q

. Si el ACF muestra un patrón sinusoidal o sinusoidal amortiguado, sugiere la presencia de estacionalidad y requiere la consideración de órdenes estacionales además de órdenes no estacionales.

Función de autocorrelación parcial (PACF)

La PACF mide la correlación entre un valor retardado (lag) y el valor actual de la serie temporal, teniendo en cuenta el efecto de los retardos intermedios. En el contexto de la modelización ARIMA, si la PACF se corta bruscamente después de un determinado retardo, mientras que los valores restantes están dentro del intervalo de confianza, sugiere un modelo AR de ese orden. El desfase en el que se corta el PACF da una idea del valor de p
.

#Generando ARIMAS
"""

warnings.filterwarnings("ignore")
from statsmodels.tsa.arima_model import ARIMA

# ARIMA(1;1;2)
#model_1 = ARIMA(datos.value, order=(1,2,1))
#model_fit_1 = model_1.fit(disp=0)
#print(model_fit_1.summary())

warnings.filterwarnings("ignore", category=UserWarning, message='Non-invertible|Non-stationary')
modelo = SARIMAX(endog = datos, order = (1, 2, 1), seasonal_order = (0, 0, 0, 0))
modelo_res = modelo.fit(disp=0)
warnings.filterwarnings("default")
modelo_res.summary()

warnings.filterwarnings("ignore")
from statsmodels.tsa.arima_model import ARIMA
# ARIMA(2;2;2)
warnings.filterwarnings("ignore", category=UserWarning, message='Non-invertible|Non-stationary')
modelo = SARIMAX(endog = datos, order = (2, 2, 2), seasonal_order = (0, 0, 0, 0))
modelo_res = modelo.fit(disp=0)
warnings.filterwarnings("default")
modelo_res.summary()

warnings.filterwarnings("ignore")
from statsmodels.tsa.arima_model import ARIMA
# ARIMA(3;2;2)
warnings.filterwarnings("ignore", category=UserWarning, message='Non-invertible|Non-stationary')
modelo = SARIMAX(endog = datos, order = (3, 2, 2), seasonal_order = (0, 0, 0, 0))
modelo_res = modelo.fit(disp=0)
warnings.filterwarnings("default")
modelo_res.summary()