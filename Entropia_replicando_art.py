# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

# =============================================================================
# %%
# =============================================================================

import nltk
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pdfminer.high_level import extract_text
from collections import Counter
from scipy.interpolate import interp1d


# =============================================================================
# %%
# =============================================================================


Texto_CienAnios = get_pdf("Textos a ajustar/Gabriel_CienAnios.pdf")
Texto_Rayuela = get_pdf("Textos a ajustar/Cortazar, Julio - Rayuela.pdf")
Texto_Yerba = get_pdf("Textos a ajustar/Creció espesa la yerba - Carmen Conde.pdf")
Texto_HarryP = get_pdf("Textos a ajustar/HARRY POTTER Y LA CAMARA SECRETA.pdf")
Texto = clean_tokenize(Texto_Yerba)
# =============================================================================
# %%
# =============================================================================

def get_pdf(file):
    pdf = extract_text(file)
    text = re.sub('http\S+', ' ', pdf)
    # Eliminación de números
    text = re.sub("\d+", ' ', text)
    # Eliminación de espacios en blanco múltiples
    text = re.sub("\\s+", ' ', text)
    return text

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


def shannon_entropy(freq):
    prob = freq / np.sum(freq)
    return -np.sum(prob * np.log2(prob))

# =============================================================================
# %%
# =============================================================================


def CalEntropy(Texto):
    Texto1 = clean_tokenize(Texto)
    diccionario_pares = {}
    for step in range(1, 500):
        if step <= len(Texto1):
            palabras_agrupadas = [' '.join(Texto1[i:i+step]) for i in range(0, len(Texto1), step)]
            diccionario_pares[step] = palabras_agrupadas
    
    diccionario_bases_df = {}

    for key, palabras_agrupadas in diccionario_pares.items():
        # Contar la frecuencia de cada conjunto de palabras
        conteo = Counter(palabras_agrupadas)
        # Crear un DataFrame a partir de los datos de frecuencia
        df = pd.DataFrame(list(conteo.items()), columns=['Palabras', 'Frecuencia'])
        # Ordenar el DataFrame de mayor a menor por frecuencia
        df = df.sort_values(by='Frecuencia', ascending=False)
        # Reiniciar el índice del DataFrame y agregar 1 para el índice
        df.index = range(1, len(df) + 1)
        diccionario_bases_df[key] = df
        


    # Vector para almacenar las entropías
    entropias1 = []

    # Para cada lista de palabras agrupadas en el diccionario anterior
    for key, df in diccionario_bases_df.items():
        # Calcular la entropía de Shannon para las frecuencias
        freq = df['Frecuencia'].values
        entropia = shannon_entropy(freq)
        entropias1.append(entropia)
        
    diccionario_cortes = {}

    # Iterar sobre diferentes longitudes de caracteres
    for n in range(1, 501):
        cortes = [Texto[i:i+n] for i in range(0, len(Texto), n)]
        diccionario_cortes[n] = cortes

    ## Ordenar y contar

    # Diccionario para almacenar los dataframes
    diccionario_dataframes = {}

    # Iterar sobre diferentes longitudes de caracteres
    for n, cortes in diccionario_cortes.items():
        # Contar la frecuencia de cada conjunto de caracteres
        conteo = Counter(cortes)
        # Crear un DataFrame a partir de los datos de frecuencia
        df = pd.DataFrame(list(conteo.items()), columns=['Caracteres', 'Frecuencia'])
        # Ordenar el DataFrame de mayor a menor por frecuencia
        df = df.sort_values(by='Frecuencia', ascending=False)
        # Reiniciar el índice del DataFrame y agregar 1 para el índice
        df.index = range(1, len(df) + 1)
        # Agregar el DataFrame al diccionario
        diccionario_dataframes[n] = df

    ## Calcular Entropia

    # Vector para almacenar las entropías
    entropias2 = []

    # Para cada DataFrame de conjunto de caracteres
    for key, df in diccionario_dataframes.items():
        # Calcular la entropía de Shannon para las frecuencias
        freq = df['Frecuencia'].values
        entropia = shannon_entropy(freq)
        entropias2.append(entropia)
        
    Number_of_character = len(diccionario_cortes[1])
    Alphabet_Size= len(set(diccionario_cortes[1]))
    Number_of_word = len(diccionario_pares[1])
    Diferent_words= len(set(diccionario_pares[1]))
    Alpha = ((diccionario_bases_df[1]['Palabras'].str.len()*diccionario_bases_df[1]['Frecuencia']).sum())/diccionario_bases_df[1]['Frecuencia'].sum()
    WDR = (Diferent_words/Number_of_word)*100
        
    return entropias1, entropias2, Number_of_character,Alphabet_Size, Number_of_word, Diferent_words, WDR, Alpha, list(diccionario_bases_df.values())[:3], list(diccionario_dataframes.values())[:3] 

# =============================================================================
# %%
# =============================================================================


entropias_p1, entropias_c1, Number_of_character1, Alphabet_Size1, Number_of_word1, Diferent_words1, WDR1, Alpha1, diccionario = CalEntropy(Texto_CienAnios)[:9]
entropias_p2, entropias_c2, Number_of_character2, Alphabet_Size2, Number_of_word2, Diferent_words2, WDR2, Alpha2,a, diccionario2 = CalEntropy(Texto_Rayuela)
entropias_p3, entropias_c3, Number_of_character3, Alphabet_Size3, Number_of_word3, Diferent_words3, WDR3, Alpha3 = CalEntropy(Texto_Yerba)[:-2]
entropias_p4, entropias_c4, Number_of_character4, Alphabet_Size4, Number_of_word4, Diferent_words4, WDR4, Alpha4 = CalEntropy(Texto_HarryP)[:-2]
# Creación de los DataFrames para entropias_c y entropias_p
entropias_c = pd.DataFrame({
    'Texto_CienAnios_Limpio': entropias_c1,
    'Texto_Rayuela_Limpio': entropias_c2,
    'Texto_Yerba': entropias_c3,
    'Texto_HarryP': entropias_c4
})

entropias_p = pd.DataFrame({
    'Texto_CienAnios_Limpio': entropias_p1,
    'Texto_Rayuela_Limpio': entropias_p2,
    'Texto_Yerba': entropias_p3,
    'Texto_HarryP': entropias_p4
})

# Creación de las listas para Number_of_character, Alphabet_Size, Number_of_word, Diferent_words, WDR y Alpha
Number_of_character = [Number_of_character1, Number_of_character2, Number_of_character3,Number_of_character4]
Alphabet_Size = [Alphabet_Size1, Alphabet_Size2, Alphabet_Size3,Alphabet_Size4]
Number_of_word = [Number_of_word1, Number_of_word2,  Number_of_word3,Number_of_word4]
Diferent_words = [Diferent_words1, Diferent_words2, Diferent_words3, Diferent_words4]
WDR = [WDR1, WDR2,  WDR3, WDR4]
Alpha = [Alpha1, Alpha2, Alpha3, Alpha4]

# =============================================================================
# %%
# =============================================================================


# Generar gráfico de entropías frente a su índice con puntos y dos líneas
# Generar gráfico de entropías frente a su índice con puntos y dos líneas
fig, ax = plt.subplots(figsize=(10,10))


ax.plot(range(1, 21), entropias_c.iloc[:,0].head(20), marker='o', linestyle='-', color='b', markersize=8, linewidth=2)
# Aquí puedes agregar otra línea si deseas
ax.plot(range(1, 21), entropias_c.iloc[:,1].head(20), marker='s', linestyle='--', color='r', markersize=6, linewidth=2)
ax.plot(range(1, 21), entropias_c.iloc[:,2].head(20), marker='s', linestyle='--', color='y', markersize=6, linewidth=2)
ax.plot(range(1, 21), entropias_c.iloc[:,3].head(20), marker='s', linestyle='--', color='g', markersize=6, linewidth=2)

ax.set_xlabel('N Caracteres', fontsize=16)
ax.set_ylabel('Entropía de Shannon', fontsize=16)
fig.suptitle('Entropías de las 3 bases de datos', fontsize=16)
ax.legend(['CienAnios', 'Rayuela', 'Yerba', "Harry Potter"], loc='lower right', fontsize=16)
ax.grid(True)
for i in ax.spines:
    ax.spines[i].set_visible(False)    
fig.tight_layout()
fig.show()

# Gráfico de n word

# Generar gráfico de entropías frente a su índice con puntos y dos líneas
fig, ax = plt.subplots(figsize=(10,10))


ax.plot(range(1, 21), entropias_p.iloc[:,0].head(20), marker='o', linestyle='-', color='b', markersize=8, linewidth=2)
# Aquí puedes agregar otra línea si deseas
ax.plot(range(1, 21), entropias_p.iloc[:,1].head(20), marker='s', linestyle='--', color='r', markersize=6, linewidth=2)
ax.plot(range(1, 21), entropias_p.iloc[:,2].head(20), marker='s', linestyle='--', color='y', markersize=6, linewidth=2)
ax.plot(range(1, 21), entropias_p.iloc[:,3].head(20), marker='s', linestyle='--', color='g', markersize=6, linewidth=2)

ax.set_xlabel('N Caracteres', fontsize=16)
ax.set_ylabel('Entropía de Shannon', fontsize=16)
fig.suptitle('Entropías de las 3 bases de datos', fontsize=16)
ax.legend(['CienAnios', 'Rayuela', 'Yerba', "Harry Potter"], loc='lower right', fontsize=16)
ax.grid(True)
for i in ax.spines:
    ax.spines[i].set_visible(False)    
fig.tight_layout()
fig.show()

# =============================================================================
# %%
# =============================================================================


##Log-Log Plot

## Para palabras

DFP1 = diccionario[0]
DFP2 = diccionario[1]
DFP3 = diccionario[2]


DFP1['Frecuencia Relativa'] = DFP1['Frecuencia'] / DFP1['Frecuencia'].sum()
DFP2['Frecuencia Relativa'] = DFP2['Frecuencia'] / DFP2['Frecuencia'].sum()
DFP3['Frecuencia Relativa'] = DFP3['Frecuencia'] / DFP3['Frecuencia'].sum()

fig, ax = plt.subplots(figsize = (7,7))

ax.loglog(range(1,15737), DFP1['Frecuencia Relativa'], marker='o', linestyle='-')
ax.loglog(range(1,40165), DFP2['Frecuencia Relativa'], marker='s', linestyle='-')
ax.loglog(range(1,41895), DFP3['Frecuencia Relativa'], marker='s', linestyle='-')
# Etiquetas y título
ax.set_xlabel('Valores de x (log)')
ax.set_ylabel('Valores de y (log)')
fig.suptitle('Gráfico log-log')
ax.legend(['1-Word', '2-Words', '3-Words'], loc='upper right')
ax.grid(True)
for i in ax.spines:
    ax.spines[i].set_visible(False)    
fig.tight_layout()
fig.show()

## Para caracteres


DFP12 = diccionario2[0]
DFP22 = diccionario2[1]
DFP32 = diccionario2[2]


DFP12['Frecuencia Relativa'] = DFP12['Frecuencia'] / DFP12['Frecuencia'].sum()
DFP22['Frecuencia Relativa'] = DFP22['Frecuencia'] / DFP22['Frecuencia'].sum()
DFP32['Frecuencia Relativa'] = DFP32['Frecuencia'] / DFP32['Frecuencia'].sum()


fig, ax = plt.subplots(figsize = (7,7))

ax.loglog(range(1,96), DFP12['Frecuencia Relativa'], marker='o', linestyle='-')
ax.loglog(range(1,1585), DFP22['Frecuencia Relativa'], marker='s', linestyle='-')
ax.loglog(range(1,8623), DFP32['Frecuencia Relativa'], marker='s', linestyle='-')
# Etiquetas y título
ax.set_xlabel('Valores de x (log)')
ax.set_ylabel('Valores de y (log)')
fig.suptitle('Gráfico log-log')
ax.legend(['1-Word', '2-Words', '3-Words'], loc='upper right')
ax.grid(True)
for i in ax.spines:
    ax.spines[i].set_visible(False)    
fig.tight_layout()
fig.show()



# =============================================================================
# %%
# =============================================================================

## Entropia condicionada


# Crear un nuevo DataFrame para almacenar los resultados
entropias_condicionadas = pd.DataFrame(columns=entropias_c.columns)

# Agregar el vector de valores como la primera fila del nuevo DataFrame
entropias_condicionadas.loc[0] = np.log2(Alphabet_Size)

# Agregar la primera fila del DataFrame original como la segunda fila del nuevo DataFrame
entropias_condicionadas.loc[1] = entropias_c.iloc[0]

# Iterar sobre cada fila del DataFrame original (excepto las dos primeras)
for i in range(1, len(entropias_c)):
    # Calcular las diferencias entre las filas consecutivas del DataFrame original
    diferencia = entropias_c.iloc[i] - entropias_c.iloc[i-1]
    
    # Agregar las diferencias al nuevo DataFrame
    entropias_condicionadas = pd.concat([entropias_condicionadas, pd.DataFrame([diferencia], columns=entropias_c.columns)], ignore_index=True)


# =============================================================================
# %%
# =============================================================================

##Entropias condicionales para todos los textos  (interpolación polinomial )
## PARA OBTENER N_Z Y H_zn
k = entropias_condicionadas.drop(entropias_condicionadas.index[0])
x = range(1,16)
y_values = np.transpose(k.head(15).to_numpy())

interpolations = []
y_interpolated_values = []  # Lista para almacenar los valores interpolados de y
x_new = np.linspace(min(x), max(x), 100)  # Vector de x interpolado
for y in y_values:
    f = interp1d(x, y, kind='cubic')
    interpolations.append(f)
    y_interp = f(x_new)  # Calculamos los valores interpolados de y para x interpolados
    y_interpolated_values.append(y_interp)

# Graficar los datos originales y las interpolaciones
plt.figure(figsize=(10, 6))
for i, f in enumerate(interpolations):
    y_interp = f(x_new)
    plt.plot(x_new, y_interp, label=f'Interpolación {i+1}')

# Graficar los puntos originales
for i, y in enumerate(y_values):
    plt.scatter(x, y, s=15, label=f'Conjunto de datos {i+1}')

plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.title('Interpolaciones Polinomiales', fontsize=15)
plt.legend()
plt.grid(True)
plt.show()

N_z = []
indice_valor_minimo = np.argmin(np.abs(y_interpolated_values[0]))
N_z.append( x_new[indice_valor_minimo])
indice_valor_minimo = np.argmin(np.abs(y_interpolated_values[1]))
N_z.append( x_new[indice_valor_minimo])
indice_valor_minimo = np.argmin(np.abs(y_interpolated_values[2]))
N_z.append( x_new[indice_valor_minimo])
indice_valor_minimo = np.argmin(np.abs(y_interpolated_values[3]))
N_z.append( x_new[indice_valor_minimo])
# =============================================================================
# %%
# =============================================================================

##Entropias puras (interpolación polinomial) y calculo de H_zn

x = range(1,16)
y_values = np.transpose(entropias_c.head(15).to_numpy())


# Interpolación polinomial de tercer grado para cada conjunto de y
interpolations = []
y_interpolated_values2 = []  # Lista para almacenar los valores interpolados de y
x_new = np.linspace(min(x), max(x), 100)  # Vector de x interpolado
for y in y_values:
    f = interp1d(x, y, kind='cubic')
    interpolations.append(f)
    y_interp = f(x_new)  # Calculamos los valores interpolados de y para x interpolados
    y_interpolated_values2.append(y_interp)

# Graficar los datos originales y las interpolaciones
plt.figure(figsize=(10, 6))
for i, f in enumerate(interpolations):
    y_interp = f(x_new)
    plt.plot(x_new, y_interp, label=f'Interpolación {i+1}')

# Graficar los puntos originales
for i, y in enumerate(y_values):
    plt.scatter(x, y, s=15, label=f'Conjunto de datos {i+1}')


##H_z

dataframes_list = []

# Iterar sobre los arreglos y crear un DataFrame para cada uno
for y_values in y_interpolated_values2:
    # Crear un DataFrame con las columnas 'x' y 'y'
    df = pd.DataFrame({'x': x_new, 'y': y_values})
    # Agregar el DataFrame a la lista
    dataframes_list.append(df)

H_zn = []

for i, df in enumerate(dataframes_list):
    x_actual = N_z[i]
    idx = np.abs(df['x'] - x_actual).idxmin()  # Encontrar el índice del valor de 'x' más cercano
    y_correspondiente = df.loc[idx, 'y']  # Obtener el valor de 'y' correspondiente
    H_zn.append(y_correspondiente)
    
## TABLA DE RATIO Y REDUNDANCIA  
# =============================================================================
# %%
# =============================================================================
 
N_z  
H_zn    
H_l = [x / y for x, y in zip(H_zn, N_z)]
H_l
k = [x / y for x, y in zip(H_l, np.log2(Alphabet_Size))]
R = [1-x for x in k]
R
