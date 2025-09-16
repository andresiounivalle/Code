# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:31:02 2024

@author: Pelu-PC
"""

# Exportar a archivos CSV
#for i, df in enumerate(Lista_Palabras, start=1):
    #df.to_csv(f'texto{i}.csv', index=False)


# =============================================================================
# %%
# =============================================================================
## Librerias

import nltk
from scipy.optimize import curve_fit
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text
import statsmodels.api as sm
from sklearn.metrics import r2_score
from scipy.interpolate import splrep, splev

# =============================================================================
# %%
# =============================================================================

## FUNCIONES ####


## Lectura de texto
def get_pdf(file):
    pdf = extract_text(file)
    text = re.sub('http\S+', ' ', pdf)
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~\\“\\º]'
    text = re.sub(regex , ' ', text)
    # Eliminación de números
    text = re.sub("\d+", ' ', text)
    # Eliminación de espacios en blanco múltiples
    text = re.sub("\\s+", ' ', text)
    return text

## Tokenizado
def clean_tokenize(texto):
    # Se convierte todo el texto a minúsculas
    text = texto.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    text = re.sub('http\S+', ' ', text)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    text = re.sub(regex , ' ', text)
    # Eliminación de números
    text = re.sub("\d+", ' ', text)
    # Eliminación de espacios en blanco múltiples
    text = re.sub("\\s+", ' ', text)
    # Tokenización por palabras individuales
    #text = text.split(sep = ' ')
    text = word_tokenize(text)
    return(text)
    
## Calculo de la entropia de shannon    
def shannon_entropy(dataframe):
    prob = dataframe.iloc[:,1]/ np.sum(dataframe.iloc[:,1])
    return -np.sum(prob * np.log2(prob))

def shannon_entropy2(serie):
    prob = serie/ np.sum(serie)
    return -np.sum(prob * np.log2(prob))

## Función para conteos y organización de base de datos
def crear_dataframe(lista):
    df = pd.DataFrame(lista, columns=['Elemento'])
    conteo = df['Elemento'].value_counts().reset_index()
    conteo.columns = ['Elemento', 'Conteo']
    conteo['Rango'] = conteo['Conteo'].rank(ascending=False)
    return conteo

## Ajuste de modelos lineal
def ajustar_modelo(dataframe):
    X = np.log(dataframe.iloc[:, 2])  # Covariable con logaritmo aplicado
    y = np.log(dataframe.iloc[:, 1])  # Variable respuesta con logaritmo aplicado
    X = sm.add_constant(X)  # Agregar una columna de unos para el término constante
    modelo = sm.OLS(y, X).fit()  # Ajustar el modelo lineal
    return  modelo

## Expresión para ajustar modelo no lineal zipf

def modelo_no_lineal(x,a,c):
    return a/(x)**c


## Expresión para ajustar modelo exponencial

def modelo_no_lineal2(x,a,c):
    return a*np.exp(-c*x)

## Expresión para ajustar modelo zipf mandelbrot

def modelo_no_lineal3(x,a,c,d):
    return a/(x+d)**c

# =============================================================================
# %%
# =============================================================================
## TEXTOS A AJUSTAR Y LISTA DE CONTEOS #

Texto_CienAnios = get_pdf("Textos a ajustar/Gabriel_CienAnios.pdf")
Texto_Rayuela = get_pdf("Textos a ajustar/Cortazar, Julio - Rayuela.pdf")
Texto_Yerba = get_pdf("Textos a ajustar/Creció espesa la yerba - Carmen Conde.pdf")
Texto_HarryP = get_pdf("Textos a ajustar/HARRY POTTER Y LA CAMARA SECRETA.pdf")
Texto = [clean_tokenize(Texto_Yerba), clean_tokenize(Texto_CienAnios), clean_tokenize(Texto_Rayuela), clean_tokenize(Texto_HarryP)]
Lista_Palabras = [crear_dataframe(lista).sort_values(by='Rango') for lista in Texto]

# =============================================================================
# %%
# =============================================================================

## MODELO LINEAL LEY DE ZIPF AJUSTADA

modelos_ajustados = []

for df in Lista_Palabras:
    modelo = ajustar_modelo(df)
    modelos_ajustados.append(modelo)

for i, modelo in enumerate(modelos_ajustados, start=1):
    print(f"Modelo {i}:")
    print(modelo.summary())
    print()
    
np.log(Lista_Palabras[1].iloc[:,0].count())-np.log(10)

# residuos

valores_ajustados_modelos = []

for modelo in modelos_ajustados:
    valores_ajustados = modelo.fittedvalues
    valores_ajustados_modelos.append(valores_ajustados)
    
valores_ajustados_euler = []

# Iterar sobre cada modelo ajustado y aplicar la función exponencial a los valores ajustados
for modelo in modelos_ajustados:
    valores_ajustados = np.exp(modelo.fittedvalues)  # Aplicar la función exponencial
    valores_ajustados_euler.append(valores_ajustados)
    
    
Beta_0 = []

for modelo in modelos_ajustados:
    beta_0 = modelo.params[0]
    Beta_0.append(beta_0)
    
plt.scatter(np.log(Lista_Palabras[0].iloc[:,2]), np.log(Lista_Palabras[0].iloc[:,1]), label='Datos')
plt.plot(np.log(Lista_Palabras[0].iloc[:,2]), valores_ajustados_modelos[0], color='red', label='Regresión lineal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple')
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(np.log(Lista_Palabras[1].iloc[:,2]), np.log(Lista_Palabras[1].iloc[:,1]), label='Datos')
plt.plot(np.log(Lista_Palabras[1].iloc[:,2]), valores_ajustados_modelos[1], color='red', label='Regresión lineal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple')
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(np.log(Lista_Palabras[2].iloc[:,2]), np.log(Lista_Palabras[2].iloc[:,1]), label='Datos')
plt.plot(np.log(Lista_Palabras[2].iloc[:,2]), valores_ajustados_modelos[2], color='red', label='Regresión lineal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple')
plt.legend()
plt.grid(True)
plt.show()


plt.scatter(np.log(Lista_Palabras[3].iloc[:,2]), np.log(Lista_Palabras[3].iloc[:,1]), label='Datos')
plt.plot(np.log(Lista_Palabras[3].iloc[:,2]), valores_ajustados_modelos[3], color='red', label='Regresión lineal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple')
plt.legend()
plt.grid(True)
plt.show()

A = []
for df in Lista_Palabras:
    a = np.sum(df.iloc[:,1])
    A.append(a)

A = np.array(A)


A = np.log(A/10)

A-np.array(Beta_0)
# =============================================================================
# %%
# =============================================================================

##Calculo de Entropía


Entropias_muestra = []

for df in Lista_Palabras:
    shanon = shannon_entropy(df)
    Entropias_muestra.append(shanon)
    
Entropias_ajustadas = []
for series in valores_ajustados_euler:
    shanon = shannon_entropy2(series)
    Entropias_ajustadas.append(shanon)
    
# =============================================================================
# %%
# =============================================================================



## Ajuste de modelo no lineal ley de zipf 

coeficientes_lista = []
y_pred_lista = []
residuos_lista = []
R2_ajustado_lista = []
AIC_lista = []
BIC_lista = []

    # Calcular el total de observaciones en el dataframe
for df in Lista_Palabras:
    # Calcular el total de observaciones en el dataframe
    total_observaciones = np.log2(len(df))/10
    
    # Extraer las columnas y
    y = df.iloc[:, 1].values
    
    # Extraer la columna x
    x = df.iloc[:, 2].values
    
    # Ajustar el modelo no lineal
    popt, pcov = curve_fit(modelo_no_lineal, x, y, p0=[total_observaciones,1])
    
    # Calcular los valores ajustados
    y_pred = modelo_no_lineal(x, *popt)
    
    # Calcular los residuos
    residuos = y - y_pred
    
    # Calcular R^2 ajustado
    n = len(y)  # número de observaciones
    p = len(popt)  # número de parámetros
    R2_ajustado = 1 - (1 - r2_score(y, y_pred)) * (n - 1) / (n - p - 1)
    
    # Calcular AIC y BIC
    SSE = np.sum(residuos ** 2)  # Suma de cuadrados de los residuos
    AIC = n * np.log(SSE / n) + 2 * p
    BIC = n * np.log(SSE / n) + p * np.log(n)
    
    coeficientes_lista.append(popt)
    y_pred_lista.append(y_pred)
    residuos_lista.append(residuos)
    R2_ajustado_lista.append(R2_ajustado)
    AIC_lista.append(AIC)
    BIC_lista.append(BIC)
    
# =============================================================================
# %%
# =============================================================================

## Ajuste modelo no lineal exponencial


coeficientes_lista2 = []
y_pred_lista2 = []
residuos_lista2 = []
R2_ajustado_lista2 = []
AIC_lista2 = []
BIC_lista2 = []

    # Calcular el total de observaciones en el dataframe
for df in Lista_Palabras:
    # Calcular el total de observaciones en el dataframe
    total_observaciones = np.log2(len(df))/10
    
    # Extraer las columnas y
    y = df.iloc[:, 1].values
    
    # Extraer la columna x
    x = df.iloc[:, 2].values
    
    # Ajustar el modelo no lineal
    popt, pcov = curve_fit(modelo_no_lineal2, x, y, p0=[500,0.1])
    
    # Calcular los valores ajustados
    y_pred = modelo_no_lineal2(x, *popt)
    
    # Calcular los residuos
    residuos = y - y_pred
    
    # Calcular R^2 ajustado
    n = len(y)  # número de observaciones
    p = len(popt)  # número de parámetros
    R2_ajustado = 1 - (1 - r2_score(y, y_pred)) * (n - 1) / (n - p - 1)
    
    # Calcular AIC y BIC
    SSE = np.sum(residuos ** 2)  # Suma de cuadrados de los residuos
    AIC = n * np.log(SSE / n) + 2 * p
    BIC = n * np.log(SSE / n) + p * np.log(n)
    
    coeficientes_lista2.append(popt)
    y_pred_lista2.append(y_pred)
    residuos_lista2.append(residuos)
    R2_ajustado_lista2.append(R2_ajustado)
    AIC_lista2.append(AIC)
    BIC_lista2.append(BIC)
    
# =============================================================================
# %%
# =============================================================================

## Modelo Zipf Mandelbrot

coeficientes_lista3 = []
y_pred_lista3 = []
residuos_lista3 = []
R2_ajustado_lista3 = []
AIC_lista3 = []
BIC_lista3 = []

    # Calcular el total de observaciones en el dataframe
for df in Lista_Palabras:
    # Calcular el total de observaciones en el dataframe
    total_observaciones = np.log2(len(df))/10
    
    # Extraer las columnas y
    y = df.iloc[:, 1].values
    
    # Extraer la columna x
    x = df.iloc[:, 2].values
    
    # Ajustar el modelo no lineal
    popt, pcov = curve_fit(modelo_no_lineal3, x, y, p0=[1747, -0.8,0.005])
    
    # Calcular los valores ajustados
    y_pred = modelo_no_lineal3(x, *popt)
    
    # Calcular los residuos
    residuos = y - y_pred
    
    # Calcular R^2 ajustado
    n = len(y)  # número de observaciones
    p = len(popt)  # número de parámetros
    R2_ajustado = 1 - (1 - r2_score(y, y_pred)) * (n - 1) / (n - p - 1)
    
    # Calcular AIC y BIC
    SSE = np.sum(residuos ** 2)  # Suma de cuadrados de los residuos
    AIC = n * np.log(SSE / n) + 2 * p
    BIC = n * np.log(SSE / n) + p * np.log(n)
    
    coeficientes_lista3.append(popt)
    y_pred_lista3.append(y_pred)
    residuos_lista3.append(residuos)
    R2_ajustado_lista3.append(R2_ajustado)
    AIC_lista3.append(AIC)
    BIC_lista3.append(BIC)

# =============================================================================
# %%
# =============================================================================

plt.scatter(Lista_Palabras[3].iloc[:,2], Lista_Palabras[3].iloc[:,1], label='Datos')
plt.plot(Lista_Palabras[3].iloc[:,2], y_pred_lista[3], color='red', label='Regresión zipf')
plt.plot(Lista_Palabras[3].iloc[:,2], y_pred_lista2[3], color='blue', label='Regresión expo')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple')
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(Lista_Palabras[2].iloc[:,2], Lista_Palabras[2].iloc[:,1], label='Datos')
plt.plot(Lista_Palabras[2].iloc[:,2], y_pred_lista[2], color='red', label='Regresión zipf')
plt.plot(Lista_Palabras[2].iloc[:,2], y_pred_lista2[2], color='blue', label='Regresión expo')

plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0,100)
plt.title('Modelo de regresión lineal simple')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# %%
# =============================================================================
##Bspline

def ajustar_b_spline(df, x_fine):
    x = np.array(range(1, len(df)+1))
    y = df.iloc[:,1]
    x_fine = np.linspace(0, len(df)+1, len(df)*100)
    tck = splrep(x, y, s=3)  # Ajuste exacto; ajusta el parámetro `s` según sea necesario
    y_spline = splev(x_fine , tck)
    return y_spline

splines_list = [ajustar_b_spline(df, x_fine) for df in Lista_Palabras]



# =============================================================================
# %%
# =============================================================================


## Boopstrap

import random
import time


# Número de remuestreos
num_remuestreos = 6000

# Lista para almacenar los dataframes resultantes
lista_dataframes = []

# Función para contar el número de veces que aparece cada palabra en el remuestreo
def contar_palabras(remuestreo):
    conteo = {}
    for palabra in remuestreo:
        conteo[palabra] = conteo.get(palabra, 0) + 1
    return conteo

# Iterar sobre cada lista
for lista in Texto:
    # Obtener las palabras de la tercera columna
    
    # Lista para almacenar los dataframes de esta lista
    dataframes_lista = []
    
    # Realizar 1000 remuestreos de la lista con reemplazo
    for i in range(num_remuestreos):
        remuestreo = random.choices(lista, k=len(lista))
        
        # Contar el número de veces que aparece cada palabra
        conteo_palabras = contar_palabras(remuestreo)
        
        # Convertir el conteo en un DataFrame y ordenarlo de mayor a menor
        df = pd.DataFrame(list(conteo_palabras.items()), columns=["Palabra", "Frecuencia"])
        df = df.sort_values(by="Frecuencia", ascending=False).reset_index(drop=True)
        
        # Agregar el DataFrame a la lista de dataframes
        dataframes_lista.append(df)
    
    # Agregar la lista de dataframes de esta lista a la lista de dataframes resultante
    lista_dataframes.append(dataframes_lista)

def calcular_entropia(dataframes):
    entropias = []
    for lista_dataframes in dataframes:
        entropias_lista = []
        for df in lista_dataframes:
            entropia = shannon_entropy(df)
            entropias_lista.append(entropia)
        entropias.append(entropias_lista)
    return entropias

# Calcular entropías
entropias = calcular_entropia(lista_dataframes)

### Dsitribución entropia
plt.hist(entropias[2], label='Datos', density = True, )
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple')
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# %%
# =============================================================================

### Preparación para boopstrap de los modelos


# Lista de vectores de números decimales
lista_vectores_original = [
    [3.5, 2.9, 1.2],
    [4.8, 1.1, 5.3, 2.7],
    [2.4, 3.6, 4.1]
]

# Función para construir un nuevo vector basado en un vector original
def construir_nuevo_vector(vector_original):
    nuevo_vector = []
    for i, valor in enumerate(vector_original, start=1):
        valor_entero = int(valor)
        nuevo_vector.extend([i] * valor_entero)
    return nuevo_vector

# Crear la lista de vectores basada en la lista original
lista_vectores_nueva = [construir_nuevo_vector(vector) for vector in y_pred_lista]

# Imprimir la lista de vectores nueva
for i, vector in enumerate(lista_vectores_nueva, start=1):
    print(f"Vector {i}: {vector}")
    
    
shannon_entropy2(y_pred_lista3[1])
shannon_entropy2(y_pred_lista2[1][y_pred_lista2[1]>0.1])

np.sum(y_pred_lista2[1])
y_pred_lista2[1][y_pred_lista2[1]>0.1]
