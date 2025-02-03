##########################################################################################
####################################### MODELO VAR #######################################
##########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic


# 1. IMPORTAR BASE DE DATOS
filepath = '/Users/khrisvaldera/Desktop/DATOS MOVELO VAR.xlsx'
df = pd.read_excel(filepath, parse_dates=['date'], index_col='date')
print(df.shape) # (144, 4)
df.tail()
        # ipc = Índice de Precios al Consumidor (IPC)
        # m1 = Liquidez del sistema bancario 
        # g = Gastos del gobierno central 
        # c = Dinero Circulante      


# 2. GRÁFICOS
for col in df.columns:
    plt.figure(figsize=(6, 4), dpi=120)
    plt.plot(df[col], color='red', linewidth=1)
    plt.title(col)
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.tick_params(axis='both', which='both', labelsize=6)
    plt.show()

# 3. DIVISIÓN DE DATOS DE PRUEBA Y ENTRENAMIENTO
nobs = 4
df_train, df_test = df[0:-nobs], df[-nobs:]

print(df_train.shape)  # (140, 4)
print(df_test.shape)  # (4, 4)


# 4. VERIFICACIÓN DE ESTACIONARIEDAD
        # Prueba Dickey-Fuller aumentada (prueba ADF)
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

        # Imprimir Resumen
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.") 
        
        #ADF por columna
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('n')

        # H0 = La serie de tiempo < 0.05 es estacionaria
        # H1 = La serie de tiempo > 0.05 no es estacionaria


        #Prueba KPSS
from statsmodels.tsa.stattools import kpss
for col in df.columns[0:]:
    result = kpss(df[col])
    print(f'Variable: {col}')
    print(f'KPSS Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[3].items():
        print(f'   {key}: {value}')
    if result[1] <= 0.05:
        print("Result: The series is non-stationary.")
    else:
        print("Result: The series is stationary.")
    print("\n")

        # H0 = La serie de tiempo > 0.05 es estacionaria
        # H1 = La serie de tiempo < 0.05 no es estacionaria
        

# 5. REALIZACIÓN DE DIFERENCIAS
    # PRIMERA diferencia
df_differenced = df_train.diff().dropna()
        # Dickey Fuller por columna - 1° diferencia
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('n')        
        # KPSS por columna - 1° diferencia
for col in df_differenced.columns[0:]:
    result = kpss(df_differenced[col])
    print(f'Variable: {col}')
    print(f'KPSS Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[3].items():
        print(f'   {key}: {value}')
    if result[1] <= 0.05:
        print("Result: The series is non-stationary.")
    else:
        print("Result: The series is stationary.")
    print("\n")  

    # SEGUNDA diferencia
df_differenced = df_differenced.diff().dropna()
        # Dickey Fuller por columna - 2° diferencia
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('n')
        # KPSS por columna - 2° diferencia
for col in df_differenced.columns[0:]:
    result = kpss(df_differenced[col])
    print(f'Variable: {col}')
    print(f'KPSS Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[3].items():
        print(f'   {key}: {value}')
    if result[1] <= 0.05:
        print("Result: The series is non-stationary.")
    else:
        print("Result: The series is stationary.")
    print("\n")  


# 6. GRÁFICOS ESTACIONARIOS
import matplotlib.pyplot as plt
%matplotlib inline
variables = ["ipc", "m1", "g", "c"]

for var in variables:
    plt.figure(figsize=(8, 5))
    plt.plot(df_differenced[var])
    plt.title(f'Estacionariedad de la serie {var}')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.grid(True)
    plt.show()

# 7. ANÁLISIS DE REZAGOS
model = VAR(df_differenced)

for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)  #Criterio de Información de Akaike
    print('BIC : ', result.bic)  #Criterio de Información de Bayesiana
    print('FPE : ', result.fpe)  #Error de Predicción Final
    print('HQIC: ', result.hqic, 'n')  # Criterio de Información de Hannan-Quinn

        # Más rápido
x = model.select_order(maxlags=15)
x.summary()


# 8. PRUEBA DE COINTEGRACIÓN
        # Si las variables están cointegradas, el VAR ya no sirve, 
        # porque la cointegración es a largo plazo y el VAR es a corto plazo
        
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johansen's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df)

        # H0 = No existen vectores de cointegración. < 0.05
        # H1 = Existen vectores de cointegración.  > 0.05


# 9. PRUEBA DE CAUSALIDAD DE GRANGER
from statsmodels.tsa.stattools import grangercausalitytests
maxlag = 15
test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    """
    Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables (differenced)
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r.replace('_y', ''), c.replace('_x', '')]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    return df

grangers_causation_matrix(df_differenced, variables=df_differenced.columns)

        # Si un valor p dado es < nivel de significancia (0.05), entonces,
        # la serie X correspondiente (columna) causa la Y (fila).

    # Granger es la influencia que tiene una variable sobre otra, además si entre
    # ambas variables se causan entre sí, entonces es una causalidad bidireccional    


# 10. ENTRENAR EL MODELO VAR
model_fitted = model.fit(14)
model_fitted.summary()


# 11. FUNCIÓN IMPULSO RESPUESTA
        # Evalúa como una variable macroeconómica reacciona ante un shock
model = VAR(df_differenced)  
model_fitted = model.fit(14)

irf = model_fitted.irf(periods=10)
for impulse in ['ipc', 'm1', 'g', 'c']:
    for response in ['ipc', 'm1', 'g', 'c']:
        if impulse != response:
            irf.plot(impulse=impulse, response=response)
            plt.title(f"Impulse: {impulse} | Response: {response}")
            plt.show()
            # cómo afecta en el tiempo *** ante un shock en ***


# 12. PRONOSTICAR EL MODELO VAR      
    # Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #> 14

    # Input data for forecasting
forecast_input = df_differenced.values[-lag_order:]
forecast_input

fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')
df_forecast

# 13. OBTENCIÓN DEL PRONÓSTICO - PRUEBA
def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(df_train, df_forecast, second_diff=True)        
df_results.loc[:, ['ipc_forecast', 'm1_forecast', 'g_forecast', 'c_forecast']]

        #GRÁFICOS
for col in df.columns:
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(df_results[col+'_forecast'], label='Forecast', color='blue')
    plt.plot(df_test[col][-nobs:], label='Actual', color='red')
    plt.title(col + ": Forecast vs Actuals")
    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.legend()
    plt.tick_params(axis='both', which='both', labelsize=6)
    plt.show()


# 14. PRONÓSTICO FINAL A 10 PERIODOS MÁS
        # Obtener el orden de rezagos
lag_order = model_fitted.k_ar 
print(lag_order) #> 14
        # Input data para pronosticar 
forecast_input = df_differenced.values[-lag_order:]
        # Pronosticar 10 pasos hacia adelante 
fc = model_fitted.forecast(y=forecast_input, steps=10)
        # Convertir pronósticos a dataframe
df_forecast = pd.DataFrame(fc, columns=df.columns+'_2d', 
                          index=pd.date_range(start='2022-01-01', periods=10, freq='MS'))
        # Volver pronósticos a escala original
df_results = invert_transformation(df_train, df_forecast, second_diff=True)
        # Mostrar pronósticos
df_results = df_results[['ipc_forecast', 'm1_forecast', 'g_forecast','c_forecast']].tail(10)
print(df_results)


# 15. GRAFICAR LOS PRONÓSTICOS A 10 PERIODOS
for col in df.columns: 
  plt.figure(figsize=(6, 4), dpi=150)
  plt.plot(df_results[col+'_forecast'].tail(10), label='Forecast', color='red')   
  plt.title(f"Forecast for {col}")
  plt.xlabel('Date')
  plt.ylabel(f'{col} Value')
  plt.legend()
  plt.xticks(rotation=45) 
  plt.show()


# 16. UNIR LOS PRONÓSTICOS CON LA DATA HISTÓRICA
df_results.rename(columns={
    'ipc_forecast': 'ipc',
    'm1_forecast': 'm1',
    'g_forecast': 'g',
    'c_forecast': 'c'
}, inplace=True)

print(df_results)

new_df = pd.concat([df, df_results], axis=0)
print(new_df)


# 17. GRÁFICOS EN CONJUNTO DE DATA HISTÓRICA Y EL PRONÓSTICO A 10 PERIODOS
for column in new_df.columns:
    plt.figure(figsize=(8, 5))
    
    # Obtener datos antes y después de enero de 2022
    data_before_2022 = new_df[column][new_df.index < '2022-01-01']
    data_after_2022 = new_df[column][new_df.index >= '2022-01-01']
    
    # Graficar datos antes de enero de 2022 en azul y después de enero de 2022 en rojo
    plt.plot(new_df.index, new_df[column], label='Data', color='blue')
    plt.plot(data_after_2022.index, data_after_2022, label = 'Pronóstico',color='red')
    
    plt.title(f"{column} Data histórica y forecast")
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()




    


  
  