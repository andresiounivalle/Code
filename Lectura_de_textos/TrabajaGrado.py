# =============================================================================

# DESCARGA DE LOS LIBROS REQUERIDOS ===========================================
import requests as r
import pdfkit
import os
import PyPDF2

html_to_pdf = r"C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe"
config = pdfkit.configuration(wkhtmltopdf=html_to_pdf)
folder = r"C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Libros_Lectura/"


# %% DIARIO DE LECUMBERRI

for i, j in zip(range(6,1,-1), range(1,6)):
    url = "https://telegra.ph/Crónica-de-una-muerte-anunciada-03-07-%s"%(i)
    path = "Cronicas_Muerte_%s.pdf"%(j)
    pdfkit.from_url(url,folder+path,configuration= config)

# VOLVERLO UN SOLO PDF
PDF_Lecumberri = [f"Cronicas_Muerte_{i}.pdf" for i in range(1, 6)]
Combi = PyPDF2.PdfMerger()

# Añadir cada PDF al archivo final
for pdf in PDF_Lecumberri:
    Combi.append(os.path.join(folder, pdf))

# Guardar el PDF combinado
output_pdf = os.path.join(folder, "CronicasDeUna_Muerte_Anun_GM.pdf")
Combi.write(output_pdf)
Combi.close()

# %% MANSION DE ARAUCAI

for i,j in zip(range(18,5,-1), range(1,14)):
    url = "https://telegra.ph/Diario-de-Lecumberri-05-25-%s"%(i)
    path = "Mansion_Arau_%s.pdf"%(j)
    pdfkit.from_url(url,folder+path,configuration= config)
    

# VOLVERLO UN SOLO PDF
PDF_Mansion = [f"Mansion_Arau_{i}.pdf" for i in range(1, 14)]
Combinado = PyPDF2.PdfMerger()

# Añadir cada PDF al archivo final
for pdf in PDF_Mansion:
    Combinado.append(os.path.join(folder, pdf))

# Guardar el PDF combinado
output_pdf = os.path.join(folder, "Mansion_De_Araucaima.pdf")
Combinado.write(output_pdf)
Combinado.close()

# %% MUERTE DEL ESTRATEGA
url = "https://telegra.ph/Diario-de-Lecumberri-05-25-4"
path = "Muerte_D_Estratega.pdf"
pdfkit.from_url(url,folder+path,configuration= config)

# %%
# =============================================================================
## Librerias

import requests as r
import pdfkit
import os
import PyPDF2
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

def shannon_entropy_centering(dataframe):
    prob = dataframe.iloc[:,1] / np.sum(dataframe.iloc[:,1])  
    return (-np.sum(prob * np.log2(prob))) / np.log2(len(dataframe))

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

Texto_DiarioLecu = get_pdf("C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Libros_Lectura/Diario_De_Lecumberri_AM.pdf")
Texto_MansionArauca = get_pdf("C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Libros_Lectura/Mansion_De_Araucaima_AM.pdf")
Texto_MuertEstra = get_pdf("C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Libros_Lectura/Muerte_D_Estratega_AM.pdf")
Texto = [clean_tokenize(Texto_DiarioLecu), clean_tokenize(Texto_MansionArauca), clean_tokenize(Texto_MuertEstra)]
Lista_Palabras = [crear_dataframe(lista).sort_values(by='Rango') for lista in Texto]

# =============================================================================

Models_Ajustados_Mutis = []

for df in List_Words_Mutis:
    modelo = ajustar_modelo(df)
    Models_Ajustados_Mutis.append(modelo)

for i, modelo in enumerate(Models_Ajustados_Mutis, start=1):
    print(f"Modelo {i}:")
    print(modelo.summary())
    print()
    
np.log(List_Words_Mutis[1].iloc[:,0].count())-np.log(10)

Ftd_Values_Mdl_Mutis = []

for modelo in Models_Ajustados_Mutis:
    Ftd_Values = modelo.fittedvalues
    Ftd_Values_Mdl_Mutis.append(Ftd_Values)
    
Ftd_Values_Euler_Mutis = []

# Iterar sobre cada modelo ajustado y aplicar la función exponencial a los valores ajustados
for modelo in Models_Ajustados_Mutis:
    Ftd_Values = np.exp(modelo.fittedvalues)  # Aplicar la función exponencial
    Ftd_Values_Euler_Mutis.append(Ftd_Values)
    
Beta_0_Mutis = []

for modelo in Models_Ajustados_Mutis:
    Beta_0_M = modelo.params[0]
    Beta_0_Mutis.append(Beta_0_M)

plt.scatter(np.log(List_Words_Mutis[0].iloc[:,2]), np.log(List_Words_Mutis[0].iloc[:,1]), label='Datos')
plt.plot(np.log(List_Words_Mutis[0].iloc[:,2]), Ftd_Values_Mdl_Mutis[0], color='red', label='Regresión lineal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple (Empresa y Tripulacion)')
plt.legend()
plt.grid(True)
plt.show()

##Calculo de Entropía

Entropias_Sample_Mutis = []

for df in List_Words_Mutis:
    shanon = shannon_entropy(df)
    Entropias_Sample_Mutis.append(shanon)
    
Entropias_Fited_Mutis = []
for series in Ftd_Values_Euler_Mutis:
    shanon = shannon_entropy2(series)
    Entropias_Fited_Mutis.append(shanon)
    
# =============================================================================


# %%
# A PARTIR DE ACA INICIA LA MANERA DE GENERAR CODIGO EXCASO VOS SABESS ____ ### 
# POEMAS DE ALVARO MUTIS ____ ## ____ ##

Url_Poemas_AM = "C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Libros_Lectura/Poemas_AlvaroMutis.pdf"
Poemas_AM = get_pdf(Url_Poemas_AM)
Text_Poemas_AM = clean_tokenize(Poemas_AM)

List_Words_Poemas_AM = crear_dataframe(Text_Poemas_AM).sort_values(by = "Rango")

Modelos = {
    "modelo_no_lineal": modelo_no_lineal,
    "modelo_no_lineal2": modelo_no_lineal2,
    "modelo_no_lineal3": modelo_no_lineal3,
}
Resultados_Poemas_AM = {}

Total_Observaciones = np.log2(len(List_Words_Poemas_AM)) / 10
y = List_Words_Poemas_AM.iloc[:, 1].values
x = List_Words_Poemas_AM.iloc[:, 2].values

for nombre, modelo in Modelos.items():
    try:
        # Estimar el número inicial de parámetros
        if nombre == "modelo_no_lineal3":  # Este modelo tiene un parámetro adicional
            p0 = [Total_Observaciones, 1, 0.1]
        else:
            p0 = [Total_Observaciones, 1]

        # Ajustar el modelo
        popt, pcov = curve_fit(modelo, x, y, p0=p0)

        # Calcular los valores ajustados
        y_pred = modelo(x, *popt)

        # Calcular los residuos
        residuos = y - y_pred

        # Calcular R² ajustado
        n = len(y)  # Número de observaciones
        p = len(popt)  # Número de parámetros
        R2_ajustado = 1 - (1 - r2_score(y, y_pred)) * (n - 1) / (n - p - 1)

        # Calcular AIC y BIC
        SSE = np.sum(residuos ** 2)  # Suma de cuadrados de los residuos
        AIC = n * np.log(SSE / n) + 2 * p
        BIC = n * np.log(SSE / n) + p * np.log(n)

        # Guardar los resultados del modelo
        Resultados_Poemas_AM[nombre] = {
            "Coeficientes": popt,
            "Valores ajustados": y_pred,
            "Residuos": residuos,
            "R2 ajustado": R2_ajustado,
            "AIC": AIC,
            "BIC": BIC,
        }

    except Exception as e:
        print(f"Error al ajustar {nombre}: {e}")

# VISUALIZACION DE LOS RESULTADOS MÁS COMODO ______ ##
for nombre, datos in Resultados_Poemas_AM.items():
    print(f"Resultados para {nombre}:")
    print(f"  Coeficientes: {datos['Coeficientes']}")
    print(f"  R2 ajustado: {datos['R2 ajustado']}")
    print(f"  AIC: {datos['AIC']}")
    print(f"  BIC: {datos['BIC']}\n")

Valores_Ajustados_Poemas_AM = [
    Resultados_Poemas_AM["modelo_no_lineal"]["Valores ajustados"],
    Resultados_Poemas_AM["modelo_no_lineal2"]["Valores ajustados"],
    Resultados_Poemas_AM["modelo_no_lineal3"]["Valores ajustados"]
]

List_Words_Poemas_AM
Valores_Ajustados_Poemas_AM 

Txt_Poemas_Mutis_Predit = []

for valores_ajustados in Valores_Ajustados_Poemas_AM:
    df_nuevo = List_Words_Poemas_AM.copy()
    
    df_nuevo["Conteo"] = np.round(valores_ajustados).astype(int)
    
    Txt_Poemas_Mutis_Predit.append(df_nuevo)

# LOS DOS DATA FRAME IMPORTANTES SON:
List_Words_Poemas_AM
Txt_Poemas_Mutis_Predit
# _______ ### FIN DE LAS LISTAS IMPORTANTES ___ ##

Txt_Poemas_Mutis_Predit.insert(0, List_Words_Poemas_AM)

Entropias_Poemas_Mutis = []
for df in Txt_Poemas_Mutis_Predit:
    shanon = shannon_entropy(df)
    Entropias_Poemas_Mutis.append(shanon)
    
Entropias_Poemas_Mutis_Centering = []
for df in Txt_Poemas_Mutis_Predit:
    shanon = shannon_entropy_centering(df)
    Entropias_Poemas_Mutis_Centering.append(shanon)
    
# ENTROPIAAS_ ______ | | _______ #|# -----##

Entropias_Poemas_Mutis
Entropias_Poemas_Mutis_Centering

# _____________________________________________________________ ####

# ________________________

# %%
# =============================================================================
## TEXTOS A AJUSTAR Y LISTA DE CONTEOS #

Url_Folder_Mutis = "C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Libros_Lectura/Libros_AlvaroMutis"
Url_Folder_Marquez = "C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Libros_Lectura/Libros_GabrielMarquez"

Lista_Mutis = os.listdir(Url_Folder_Mutis)
Lista_Marquez = os.listdir(Url_Folder_Marquez)

# del Url_Folder_Marquez,Url_Folder_Mutis,Lista_Mutis,Lista_Marquez

Url_Books_Mutis = [os.path.join(Url_Folder_Mutis, Lista_Mutis) for Lista_Mutis in Lista_Mutis]
Url_Books_Marquez = [os.path.join(Url_Folder_Marquez, Lista_Marquez) for Lista_Marquez in Lista_Marquez]

Gt_Mutis = []
for i in Url_Books_Mutis:
    Book_M = get_pdf(i)
    Gt_Mutis.append(Book_M)

Gt_Marquez = []
for i in Url_Books_Marquez:
    Book_Ma = get_pdf(i)
    Gt_Marquez.append(Book_Ma)

Txt_Mutis = []
for i in Gt_Mutis:
    Text_M = clean_tokenize(i)
    Txt_Mutis.append(Text_M)
    
Txt_Marquez = []
for i in Gt_Marquez:
    Text_Ma = clean_tokenize(i)
    Txt_Marquez.append(Text_Ma)

List_Words_Mutis = [crear_dataframe(lista).sort_values(by = "Rango") for lista in Txt_Mutis]
List_Words_Marquez = [crear_dataframe(lista).sort_values(by = "Rango") for lista in Txt_Marquez]

# UNIFICACION DE LOS DATA FRAMES PARA OBSERVAR EL OBJETIVO DEL ESTIMADOR ___ | #

# UNIFICACION DE MUTIS ___ | #
Data_Frame_AllBooks_Mutis = pd.concat(List_Words_Mutis, ignore_index=True)
Data_Frame_AllBooks_Mutis = Data_Frame_AllBooks_Mutis.sort_values(by="Rango", ascending=True).reset_index(drop=True)

# UNIFICACION DE MARQUEZ ___ | ___ ##
Data_Frame_AllBooks_Marquez = pd.concat(List_Words_Marquez, ignore_index=True)
Data_Frame_AllBooks_Marquez = Data_Frame_AllBooks_Marquez.sort_values(by = "Rango", ascending = True).reset_index(drop = True)

# UNIFICACION DE LOS DOS AUTORES EN UN SOLO DATA FRAME ___ | ___ | ____ ###%
Data_Frame_AllBooks = pd.concat(List_Words_Mutis+List_Words_Marquez,ignore_index=True)
Data_Frame_AllBooks = Data_Frame_AllBooks.sort_values(by = "Rango", ascending = True).reset_index(drop = True)

# FIN DEL PROCESO DE UNIFICACION DE LOS DATA FRAMES DEL OBJETIVO DE SIMP ___ #$#

del Book_M,Book_Ma,Gt_Marquez,Gt_Mutis,i,Lista_Marquez,Lista_Mutis,Text_M,Text_Ma
del Url_Books_Marquez,Url_Books_Mutis,Url_Folder_Marquez,Url_Folder_Mutis
# Texto = [clean_tokenize(Texto_DiarioLecu), clean_tokenize(Texto_MansionArauca), clean_tokenize(Texto_MuertEstra)]
# Lista_Palabras = [crear_dataframe(lista).sort_values(by='Rango') for lista in Texto]

## AJUSTE DEL MODELO NO LINEAL DE ZIPF  _____________________________ $##
# CON LA UNIFICACION DE LOS 3 DATA FRAMES, Y POEMAS 

Lista_DataFrames_All_Zipf_Models = []  # Lista vacía
Lista_DataFrames_All_Zipf_Models.extend([Data_Frame_AllBooks, Data_Frame_AllBooks_Marquez, Data_Frame_AllBooks_Mutis, List_Words_Poemas_AM])

# _ AHORA PROCESO PARA AJUSTAR ZIPF CON TODOS LOS DATA FRAMES _____ ##

Coeficiente_List_All = []
Y_Predit_List_All = []
Residuos_List_All = []
R2_Ajustado_List_All = []
AIC_List_All = []
BIC_List_All = []

# Lista_DataFrames_All_Zipf_Models
# ORDEN: TODOS LOS LIBROS, TODOS MARQUEZ, TODOS MUTIS, POEMAS DE MUTIS.
for df in Lista_DataFrames_All_Zipf_Models:
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
    
    Coeficiente_List_All.append(popt)
    Y_Predit_List_All.append(y_pred)
    Residuos_List_All.append(residuos)
    R2_Ajustado_List_All.append(R2_ajustado)
    AIC_List_All.append(AIC)
    BIC_List_All.append(BIC)
    
 
## AJUSTE DEL MODELO LINEAL DE ZIPF  _____________________________ $##

Modelos_Ajustados_Lineales_All = []

for df in Lista_DataFrames_All_Zipf_Models:
    modelo = ajustar_modelo(df)  
    Modelos_Ajustados_Lineales_All.append(modelo)

Parametros_Lineales_All_Zipf = []    
for modelo in Modelos_Ajustados_Lineales_All:
    Beta_0_M = modelo.params[1]
    Parametros_Lineales_All_Zipf.append(Beta_0_M)
    
## ___ | PROCESO DE VALIDACION DE SUPUESTOS DE LOS MODELOS ______ ###

Valores_Ajustados_All = []
Residuos_Studentizados_All = []

for modelo in Modelos_Ajustados_Lineales_All:
    valores_ajustados = modelo.fittedvalues
    residuos_studentizados = modelo.get_influence().resid_studentized_internal
    
    Valores_Ajustados_All.append(valores_ajustados)
    Residuos_Studentizados_All.append(residuos_studentizados)

from scipy.stats import shapiro
shapiro(Residuos_Studentizados_All[1])

len(Valores_Ajustados_All[3])
plt.scatter((Valores_Ajustados_All[1]), (Residuos_Studentizados_All[1]), alpha=0.7, label='Residuos estudentizados')
plt.axhline(0, color='red', linestyle='--', label='Línea de referencia (y=0)')
plt.xlabel('Valores ajustados')
plt.ylabel('Residuos estudentizados')
plt.title('Gráfico de residuos vs valores ajustados')
plt.legend()
plt.grid(True)
plt.show()

# GRAFICA DE AJUSTE DE LOS MODELOS _____ ###

plt.scatter(np.log(Lista_Data_Frames_Agrupados[1].iloc[:,2]), np.log(Lista_Data_Frames_Agrupados[1].iloc[:,1]), label='Datos')
plt.plot(np.log(Lista_Data_Frames_Agrupados[1].iloc[:,2]), Valores_Ajustados_All[1], color='red', label='Regresión lineal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple (Empresa y Tripulacion)')
plt.legend()
plt.grid(True)
plt.show()

## AJUSTE DEL MODELO LINEAL DE ZIPF POR AUTOR  __________________________ $##

# GABRIEL _____ ##
Modelos_Ajustados_Lineales_Marquez = []

for df in List_Words_Marquez:
    modelo = ajustar_modelo(df)  
    Modelos_Ajustados_Lineales_Marquez.append(modelo)
    
Parametros_Lineales_Marquez_Zipf = []    
for modelo in Modelos_Ajustados_Lineales_Marquez:
    Beta_0_M = modelo.params[1]
    Parametros_Lineales_Marquez_Zipf.append(Beta_0_M)
    
# MUTIS _____ $##

Modelos_Ajustados_Lineales_Mutis = []

for df in List_Words_Mutis:
    modelo = ajustar_modelo(df)  
    Modelos_Ajustados_Lineales_Mutis.append(modelo)
    
Parametros_Lineales_Mutis_Zipf = []    
for modelo in Modelos_Ajustados_Lineales_Marquez:
    Beta_0_M = modelo.params[0]
    Parametros_Lineales_Mutis_Zipf.append(Beta_0_M)
np.log(Parametros_Lineales_Mutis_Zipf)

# | MODELOS CON LOS ELEMENTOS AGRUPADOS DENTRO DEL CONSOLIDADO DE LIBROS __|

Lista_Data_Frames_Agrupados = [
    df.groupby("Elemento", as_index=False).agg({"Conteo": "sum", "Rango": "first"})
    for df in Lista_DataFrames_All_Zipf_Models
]

for i, df in enumerate(Lista_Data_Frames_Agrupados):
    df.sort_values(by="Conteo", ascending=False, inplace=True)
    df["Rango"] = range(1, len(df) + 1)
    Lista_Data_Frames_Agrupados[i] = df

Modelos_Ajustados_Lineales_All_Unificacion = []

for df in Lista_Data_Frames_Agrupados:
    modelo = ajustar_modelo(df)  
    Modelos_Ajustados_Lineales_All_Unificacion.append(modelo)
    
Parametros_Lineales_All_Zipf = []    
for modelo in Modelos_Ajustados_Lineales_All_Unificacion:
    Beta_0_M = modelo.params[1]
    Parametros_Lineales_All_Zipf.append(Beta_0_M)
###################################################################################
###################################################################################
###################################################################################
## AJUSTE DEL MODELO NO LINEAL DE ZIPF ________________________________ $##

# MODELO FOR MUTIS
Coeficiente_List_Mutis = []
Y_Predit_List_Mutis = []
Residuos_List_Mutis = []
R2_Ajustado_List_Mutis = []
AIC_List_Mutis = []
BIC_List_Mutis = []

    # Calcular el total de observaciones en el dataframe
for df in List_Words_Mutis:
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
    
    Coeficiente_List_Mutis.append(popt)
    Y_Predit_List_Mutis.append(y_pred)
    Residuos_List_Mutis.append(residuos)
    R2_Ajustado_List_Mutis.append(R2_ajustado)
    AIC_List_Mutis.append(AIC)
    BIC_List_Mutis.append(BIC)
    
# MODELO FOR MARQUEZ ___________________________________ $##$
Coeficiente_List_Marquez = []
Y_Predit_List_Marquez = []
Residuos_List_Marquez = []
R2_Ajustado_List_Marquez = []
AIC_List_Marquez = []
BIC_List_Marquez = []

    # Calcular el total de observaciones en el dataframe
for df in List_Words_Marquez:
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
    
    Coeficiente_List_Marquez.append(popt)
    Y_Predit_List_Marquez.append(y_pred)
    Residuos_List_Marquez.append(residuos)
    R2_Ajustado_List_Marquez.append(R2_ajustado)
    AIC_List_Marquez.append(AIC)
    BIC_List_Marquez.append(BIC)

## FIN DEL AJUSTE DEL MODELO NO LINEAL DE ZIPF ________________________________ $##

## AJUSTE DEL MODELO NO LINEAL EXPONENCIAL __________________________ $##$|#
# MODELO FOR MUTIS

Coeficiente_List_Mutis2 = []
Y_Predit_List_Mutis2 = []
Residuos_List_Mutis2 = []
R2_Ajustado_List_Mutis2 = []
AIC_List_Mutis2 = []
BIC_List_Mutis2 = []

    # Calcular el total de observaciones en el dataframe
for df in List_Words_Mutis:
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
    
    Coeficiente_List_Mutis2.append(popt)
    Y_Predit_List_Mutis2.append(y_pred)
    Residuos_List_Mutis2.append(residuos)
    R2_Ajustado_List_Mutis2.append(R2_ajustado)
    AIC_List_Mutis2.append(AIC)
    BIC_List_Mutis2.append(BIC)

# MODELO FOR MARQUEZ
Coeficiente_List_Marquez2 = []
Y_Predit_List_Marquez2 = []
Residuos_List_Marquez2 = []
R2_Ajustado_List_Marquez2 = []
AIC_List_Marquez2 = []
BIC_List_Marquez2 = []

    # Calcular el total de observaciones en el dataframe
for df in List_Words_Marquez:
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
    
    Coeficiente_List_Marquez2.append(popt)
    Y_Predit_List_Marquez2.append(y_pred)
    Residuos_List_Marquez2.append(residuos)
    R2_Ajustado_List_Marquez2.append(R2_ajustado)
    AIC_List_Marquez2.append(AIC)
    BIC_List_Marquez2.append(BIC)
## FIN DEL AJUSTE DEL MODELO NO LINEAL EXPONENCIAL ___________________________ $##

## AJUSTE DEL MODELO ZIPF MANDELBROT _______________________ $#%!###

# MODELO FOR MUTIS
Coeficiente_List_Mutis3 = []
Y_Predit_List_Mutis3 = []
Residuos_List_Mutis3 = []
R2_Ajustado_List_Mutis3 = []
AIC_List_Mutis3 = []
BIC_List_Mutis3 = []

    # Calcular el total de observaciones en el dataframe
for df in List_Words_Mutis:
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
    
    Coeficiente_List_Mutis3.append(popt)
    Y_Predit_List_Mutis3.append(y_pred)
    Residuos_List_Mutis3.append(residuos)
    R2_Ajustado_List_Mutis3.append(R2_ajustado)
    AIC_List_Mutis3.append(AIC)
    BIC_List_Mutis3.append(BIC)

# MODELO FOR MARQUEZ ____________________________________ #$##
Coeficiente_List_Marquez3 = []
Y_Predit_List_Marquez3 = []
Residuos_List_Marquez3 = []
R2_Ajustado_List_Marquez3 = []
AIC_List_Marquez3 = []
BIC_List_Marquez3 = []

    # Calcular el total de observaciones en el dataframe
for df in List_Words_Marquez:
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
    
    Coeficiente_List_Marquez3.append(popt)
    Y_Predit_List_Marquez3.append(y_pred)
    Residuos_List_Marquez3.append(residuos)
    R2_Ajustado_List_Marquez3.append(R2_ajustado)
    AIC_List_Marquez3.append(AIC)
    BIC_List_Marquez3.append(BIC)

## FIN DEL AJUSTE DEL MODELO ZIPF MANDELBROT __________________________ $##


#plt.scatter(List_Words_Mutis[2].iloc[:,2], List_Words_Mutis[2].iloc[:,1], label='Datos')
#plt.plot(List_Words_Mutis[2].iloc[:,2], Y_Predit_List_Mutis[2], color='red', label='Regresión zipf')
#plt.plot(List_Words_Mutis[2].iloc[:,2], Y_Predit_List_Mutis2[2], color='blue', label='Regresión expo')
#plt.plot(List_Words_Mutis[2].iloc[:,2], Y_Predit_List_Mutis3[2], color='yellow', label='Regresión mandelbrot')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.xlim(0,100)
#plt.title('Comparación Modelos (Abdul Soñador de Navios)')
#plt.legend()
#plt.grid(True)
#plt.show()

#BIC_List_Mutis
#BIC_List_Mutis2
#BIC_List_Mutis3

#AIC_List_Mutis
#AIC_List_Mutis2
#AIC_List_Mutis3


## PROCESO PARA LA ELABORACION DEL BOOPSTRAP EN LOS LIBROS DE MUTIS _____ $$!%"%##

import random
import time

Num_Remuestreos = 2000

# Lista para almacenar los dataframes resultantes
List_DataFramesrames = []
List_DataFramesrames_Ma = [] 

# Función para contar el número de veces que aparece cada palabra en el remuestreo
def contar_palabras(remuestreo):
    conteo = {}
    for palabra in remuestreo:
        conteo[palabra] = conteo.get(palabra, 0) + 1
    return conteo

# Iterar sobre cada lista de Mutis
for lista in Txt_Mutis:
    # Obtener las palabras de la tercera columna
    
    # Lista para almacenar los dataframes de esta lista
    dataframes_lista = []
    
    # Realizar 1000 remuestreos de la lista con reemplazo
    for i in range(Num_Remuestreos):
        remuestreo = random.choices(lista, k=len(lista))
        
        # Contar el número de veces que aparece cada palabra
        conteo_palabras = contar_palabras(remuestreo)
        
        # Convertir el conteo en un DataFrame y ordenarlo de mayor a menor
        df = pd.DataFrame(list(conteo_palabras.items()), columns=["Palabra", "Frecuencia"])
        df = df.sort_values(by="Frecuencia", ascending=False).reset_index(drop=True)
        
        # Agregar el DataFrame a la lista de dataframes
        dataframes_lista.append(df)
    
    # Agregar la lista de dataframes de esta lista a la lista de dataframes resultante
    List_DataFramesrames.append(dataframes_lista)

# Iterar sobre cada lista de Marquez    
for lista in Txt_Marquez:
    # Obtener las palabras de la tercera columna
    
    # Lista para almacenar los dataframes de esta lista
    dataframes_lista = []
    
    # Realizar 1000 remuestreos de la lista con reemplazo
    for i in range(Num_Remuestreos):
        remuestreo = random.choices(lista, k=len(lista))
        
        # Contar el número de veces que aparece cada palabra
        conteo_palabras = contar_palabras(remuestreo)
        
        # Convertir el conteo en un DataFrame y ordenarlo de mayor a menor
        df = pd.DataFrame(list(conteo_palabras.items()), columns=["Palabra", "Frecuencia"])
        df = df.sort_values(by="Frecuencia", ascending=False).reset_index(drop=True)
        
        # Agregar el DataFrame a la lista de dataframes
        dataframes_lista.append(df)
    
    # Agregar la lista de dataframes de esta lista a la lista de dataframes resultante
    List_DataFramesrames_Ma.append(dataframes_lista)


def calcular_entropia(dataframes):
    entropias = []
    for lista_dataframes in dataframes:
        entropias_lista = []
        for df in lista_dataframes:
            entropia = shannon_entropy(df)
            entropias_lista.append(entropia)
        entropias.append(entropias_lista)
    return entropias

List_DataFramesrames[0][:5]
# Calcular entropías
# ENTROPIAS ORIGINALES DE LOS LIBROS PARA USAR EN LAS PRUEBAS DE COMPARACION
# _____________________________________________________________ ####

Entropias_Mutis = calcular_entropia(List_DataFramesrames)
Entropias_Marquez = calcular_entropia(List_DataFramesrames_Ma)

# _____________________________________________________________ ####

### Dsitribución entropia
#plt.hist(Entropias_Mutis[2], label='Datos', density = True, )
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.title('Modelo de regresión lineal simple')
#plt.legend()
#plt.grid(True)
#plt.show()

# FALTA ACOMODAR LOS BOSTRAP DE LOS MODELOS
### Preparación para boopstrap de los modelos_______________________________ ##%#2#

# MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS |  ____ ###

# SE CREAN LOS DATA FRAME DE LOS 7 LIBROS CON LOS VALORES __________________ $
# PREDICTORES DE LOS MODELOS AJUSTADOS __________________ $

List_Words_Mutis_Predit = []
List_Words_Mutis2_Predit = []
List_Words_Mutis3_Predit = []
# del total_observaciones,row,SSE,shanon

for i in range(len(List_Words_Mutis)):
    df = List_Words_Mutis[i].copy()
    y_pred = np.round(Y_Predit_List_Mutis[i],0)
    
    df['Conteo'] = y_pred
    
    List_Words_Mutis_Predit.append(df)
    
for i in range(len(List_Words_Mutis)):
    df = List_Words_Mutis[i].copy()
    y_pred = np.round(Y_Predit_List_Mutis2[i],0)
    
    df['Conteo'] = y_pred
    
    List_Words_Mutis2_Predit.append(df)
    
for i in range(len(List_Words_Mutis)):
    df = List_Words_Mutis[i].copy()
    y_pred = np.round(Y_Predit_List_Mutis3[i],0)
    
    df['Conteo'] = y_pred
    
    List_Words_Mutis3_Predit.append(df)

# MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS |  ____ ###


Txt_Mutis_Predit = []
Txt_Mutis_Predit2 = []
Txt_Mutis_Predit3 = []

for df in List_Words_Mutis_Predit:
    repeated_words = []
    
    for index, row in df.iterrows():
        repeated_words.extend([row['Elemento']] * int(row['Conteo']))
    
    Txt_Mutis_Predit.append(repeated_words)
    
for df in List_Words_Mutis2_Predit:
    repeated_words = []
    
    for index, row in df.iterrows():
        repeated_words.extend([row['Elemento']] * int(row['Conteo']))
    
    Txt_Mutis_Predit2.append(repeated_words)

for df in List_Words_Mutis3_Predit:
    repeated_words = []
    
    for index, row in df.iterrows():
        repeated_words.extend([row['Elemento']] * int(row['Conteo']))
    
    Txt_Mutis_Predit3.append(repeated_words)

# MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS | MUTIS |  ____ ###

num_remuestreos = 2000
lista_dataframes_predit_M = []
lista_dataframes_predit2_M = []
lista_dataframes_predit3_M = []

for lista in Txt_Mutis_Predit:
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
    lista_dataframes_predit_M.append(dataframes_lista)


for lista in Txt_Mutis_Predit2:
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
    lista_dataframes_predit2_M.append(dataframes_lista)
    
for lista in Txt_Mutis_Predit3:
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
    lista_dataframes_predit3_M.append(dataframes_lista)

# Calcular entropías

# ENTROPIAS DE LOS MODELOS AJUSTADOS PARA MUTIS
# 3 MODELOS, 7 LIBROS, POR ENDE 21 BOOSTRAPS.
# SON LAS LISTAS CON LOS QUE SE USARAN PARA LAS PRUEBAS DE HIPOTESIS

# ___________________________________________________________________ ####
Entropias_predit_M = calcular_entropia(lista_dataframes_predit_M)
Entropias_predit2_M = calcular_entropia(lista_dataframes_predit2_M)
Entropias_predit3_M = calcular_entropia(lista_dataframes_predit3_M)
# ___________________________________________________________________ ####

#FIN DEL CALCULO DE ENTROPIAS PARA LOS MODELOS CON RESPECTO A ALVARO MUTIS __ ####

# MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ |  ____ ###

# SE CREAN LOS DATA FRAME DE LOS 7 LIBROS CON LOS VALORES __________________ $
# PREDICTORES DE LOS MODELOS AJUSTADOS __________________ $

List_Words_Marquez_Predit = []
List_Words_Marquez2_Predit = []
List_Words_Marquez3_Predit = []

# del total_observaciones,row,SSE,shanon

for i in range(len(List_Words_Marquez)):
    df = List_Words_Marquez[i].copy()
    y_pred = np.round(Y_Predit_List_Marquez[i],0)
    
    df['Conteo'] = y_pred
    
    List_Words_Marquez_Predit.append(df)
    
for i in range(len(List_Words_Marquez)):
    df = List_Words_Marquez[i].copy()
    y_pred = np.round(Y_Predit_List_Marquez2[i],0)
    
    df['Conteo'] = y_pred
    
    List_Words_Marquez2_Predit.append(df)
    
for i in range(len(List_Words_Marquez)):
    df = List_Words_Marquez[i].copy()
    y_pred = np.round(Y_Predit_List_Marquez3[i],0)
    
    df['Conteo'] = y_pred
    
    List_Words_Marquez3_Predit.append(df)

# MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ |  ____ ###


Txt_Marquez_Predit = []
Txt_Marquez_Predit2 = []
Txt_Marquez_Predit3 = []

for df in List_Words_Marquez_Predit:
    repeated_words = []
    
    for index, row in df.iterrows():
        repeated_words.extend([row['Elemento']] * int(row['Conteo']))
    
    Txt_Marquez_Predit.append(repeated_words)
    
for df in List_Words_Marquez2_Predit:
    repeated_words = []
    
    for index, row in df.iterrows():
        repeated_words.extend([row['Elemento']] * int(row['Conteo']))
    
    Txt_Marquez_Predit2.append(repeated_words)

for df in List_Words_Marquez3_Predit:
    repeated_words = []
    
    for index, row in df.iterrows():
        repeated_words.extend([row['Elemento']] * int(row['Conteo']))
    
    Txt_Marquez_Predit3.append(repeated_words)


# PARTE PARA EL REMUESTREO DE LOS MODELOS DE GABRIEL GARCIA MARQUEZ _____ ###

lista_dataframes_predit_Mar = []
lista_dataframes_predit2_Mar = []
lista_dataframes_predit3_Mar = []

for lista in Txt_Marquez_Predit:
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
    lista_dataframes_predit_Mar.append(dataframes_lista)


for lista in Txt_Marquez_Predit2:
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
    lista_dataframes_predit2_Mar.append(dataframes_lista)
    
for lista in Txt_Marquez_Predit3:
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
    lista_dataframes_predit3_Mar.append(dataframes_lista)

# MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ |  ____ ###
# ENTROPIAS DE INTERES ____ ###

Entropias_predit_Mar = calcular_entropia(lista_dataframes_predit_Mar)
Entropias_predit2_Mar = calcular_entropia(lista_dataframes_predit2_Mar)
Entropias_predit3_Mar = calcular_entropia(lista_dataframes_predit3_Mar)

# ENTROPIAS DE INTERES ____ ###
# MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ | MARQUEZ |  ____ ###


Entropias_Sample_Mutis = []
Entropias_Fited_Mutis = []

List_DataFramesrames[0]
List_Words_Mutis[0]
List_Words_Mutis_Predit[0]

for df in List_Words_Mutis:
    shanon = shannon_entropy(df)
    Entropias_Sample_Mutis.append(shanon)
    
for df in List_Words_Mutis3_Predit:
    shanon = shannon_entropy(df)
    Entropias_Fited_Mutis.append(shanon)

Entropias_Sample_Mutis

Entropias_Sample_Marquez= []

def shannon_entropy_centering(dataframe):
    prob = dataframe.iloc[:,1]/ np.sum(dataframe.iloc[:,1])
    return (-np.sum(prob * np.log2(prob)))/np.log2(np.sum(dataframe.iloc[:,1]))

def shannon_entropy_centering(dataframe):
    prob = dataframe.iloc[:,1] / np.sum(dataframe.iloc[:,1])
    return (-np.sum(prob * np.log2(prob))) / np.log2(len(dataframe))


# _ | CALCULO NUEVO DE LAS ENTROPIAS CON LA CENTRALIZACION DE LA DIVISION DE CANTIDAD
# DE PALABRAS UNICAS EN EL TEXTO _____ ###
 
Entropias_Sample_Mutis = []
Entropias_Fited_Mutis = []
               
List_Words_Mutis_Predit
for df in List_Words_Mutis3_Predit:
    shanon = shannon_entropy_centering(df)
    Entropias_Sample_Mutis.append(shanon)

Entropias_Fited_Marquez = []

for df in List_Words_Marquez3_Predit:
    shanon = shannon_entropy_centering(df)
    Entropias_Fited_Marquez.append(shanon)
    
Entropias_Fited_Marquez


List_Words_Marquez3_Predit
Entropias_Fited_Mutis
Entropias_Sample_Marquez

Entropias_Fited_Marquez = []

for df in List_Words_Marquez3_Predit:
    shanon = shannon_entropy(df)
    Entropias_Fited_Marquez.append(shanon)

Entropias_Fited_Marquez 

List_Words_Mutis[0]["Conteo"].sum()

List_DataFramesrames[0][1]["Frecuencia"].sum()

# || ___________________________________________________________________________ || #
# || RESULTADOS PUNTUALES PARA PONER EN EL TG ______________________ ||| -------- ####
# || ___________________________________________________________________________ || #

# PRUEBAS DE KOLMOGOROV SMIRNOF ____ ##$#$$

from scipy.stats import ks_2samp
from decimal import Decimal

Kolmogorov_Mutis_VS_Marquez = ks_2samp(Entropias_Marquez[1], Entropias_Mutis[1])

Kolmogorov_Marquez_Modelo1 = ks_2samp(Entropias_Marquez[0], Entropias_predit_Mar[0])
Kolmogorov_Marquez_Modelo2 = ks_2samp(Entropias_Marquez[0], Entropias_predit2_Mar[0])
Kolmogorov_Marquez_Modelo3 = ks_2samp(Entropias_Marquez[0], Entropias_predit3_Mar[0])

Kolmogorov_Mutis_Modelo1 = ks_2samp(Entropias_Mutis[0], Entropias_predit_M[0])
Kolmogorov_Mutis_Modelo2 = ks_2samp(Entropias_Mutis[0], Entropias_predit2_M[0])
Kolmogorov_Mutis_Modelo3 = ks_2samp(Entropias_Mutis[0], Entropias_predit3_M[0])

ks_2samp(Entropias_predit3_M[0], Entropias_predit3_M[1])

print("Estadístico KS:", Kolmogorov_Marquez_Modelo1.statistic)
print("P-valor:", Kolmogorov_Marquez_Modelo1.pvalue)

del Kolmogorov_Marquez_Modelo1,Kolmogorov_Marquez_Modelo2,Kolmogorov_Marquez_Modelo3
del Kolmogorov_Mutis_Modelo1,Kolmogorov_Mutis_Modelo2,Kolmogorov_Mutis_Modelo3
del Kolmogorov_Mutis_VS_Marquez

Entropias_Mutis
Entropias_Marquez

# PROCESO DE EXTRACCION DE LAS ENTROPIAS _______ || ###
df = pd.DataFrame({f"Columna_{i+1}": columna for i, columna in enumerate(Entropias_predit3_Mar)})
df.to_excel("C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Entropias_Modelo3_Marquez.xlsx", 
            index=False, 
            engine="openpyxl")

# DESCRIPTIVOS DE LOS LIBROS ______________________ ##

Url_Folder_Mutis = "C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Libros_Lectura/Libros_AlvaroMutis"
Url_Folder_Marquez = "C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Libros_Lectura/Libros_GabrielMarquez"

Lista_Mutis = os.listdir(Url_Folder_Mutis)
Lista_Marquez = os.listdir(Url_Folder_Marquez)

del Url_Folder_Mutis,Url_Folder_Marquez

Lista_Mutis # ORDEN:
    # 1. Abdul Soñador de Navios
    # 2. Diario de Lecumberri
    # 3. Ilona Llega con la Lluvia
    # 4. Mansion de Araucaima
    # 5. La Muerte del Estratega
    # 6. La Nieve del Almirante
    # 7. Tipico de Mar y Tierra

Lista_Marquez # ORDEN:
    # 1. Amor en los tiempos del colera
    # 2. El Coronel no tiene quien le escriba
    # 3. Cronicas de una muerte anunciada
    # 4. Cien Años de Soledad
    # 5. Noticias de un secuestro
    # 6. Relatos de un naufrago
    # 7. Relatos de mis putas tristes

List_Words_Mutis
List_Words_Marquez 

AIC_List_Marquez
AIC_List_Marquez2
AIC_List_Marquez3

Resultados_Descriptivos = []

for df in List_Words_Mutis:
    N_Words = len(df)
    Cant_Words = df['Conteo'].sum()
    Resultados_Descriptivos.append({'Num_Filas': N_Words, 'Suma_Conteo': Cant_Words})

Descriptivos = pd.DataFrame(Resultados_Descriptivos)
print(Descriptivos)

Resultados_Descriptivos = []

for df in List_Words_Marquez:
    N_Words = len(df)
    Cant_Words = df['Conteo'].sum()
    Resultados_Descriptivos.append({'Num_Filas': N_Words, 'Suma_Conteo': Cant_Words})

Descriptivos = pd.DataFrame(Resultados_Descriptivos)
print(Descriptivos)

del df,N_Words,Cant_Words

Data_Frame_Temporal = pd.DataFrame(List_Words_Marquez[5])
Data_Frame_Temporal.to_excel("C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Libro_Temporal.xlsx", 
            index=False, 
            engine="openpyxl")

Data_Frame_Temporal.iloc[:,1]
del Data_Frame_Temporal
# ________________________________ ##### _______________________ ####
# FIN DEL DESARROLLO DEL CODIGO PARA LO QUE VA HASTA AHORA

# %%
# =============================================================================
# FUNCION QUE YA CREAMOS CON ANTERIODIDAD
def calcular_entropia(dataframes):
    entropias = []
    for lista_dataframes_predit in dataframes:
        entropias_lista = []
        for df in lista_dataframes_predit:
            entropia = shannon_entropy(df)
            entropias_lista.append(entropia)
        entropias.append(entropias_lista)
    return entropias


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
plt.title('Modelo de regresión lineal simple (D_Lecumberri)')
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(np.log(Lista_Palabras[1].iloc[:,2]), np.log(Lista_Palabras[1].iloc[:,1]), label='Datos')
plt.plot(np.log(Lista_Palabras[1].iloc[:,2]), valores_ajustados_modelos[1], color='red', label='Regresión lineal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple (Mansion_Arauca)')
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(np.log(Lista_Palabras[2].iloc[:,2]), np.log(Lista_Palabras[2].iloc[:,1]), label='Datos')
plt.plot(np.log(Lista_Palabras[2].iloc[:,2]), valores_ajustados_modelos[2], color='red', label='Regresión lineal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple (Dead Strategist)')
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

plt.scatter(Lista_Palabras[2].iloc[:,2], Lista_Palabras[2].iloc[:,1], label='Datos')
plt.plot(Lista_Palabras[2].iloc[:,2], y_pred_lista[2], color='red', label='Regresión zipf')
plt.plot(Lista_Palabras[2].iloc[:,2], y_pred_lista2[2], color='blue', label='Regresión expo')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Modelo de regresión lineal simple')
plt.legend()
plt.grid(True)
plt.show()

plt.scatter(Lista_Palabras[2].iloc[:,2], Lista_Palabras[2].iloc[:,1], label='Datos')
plt.plot(Lista_Palabras[2].iloc[:,2], y_pred_lista[2], color='red', label='Regresión zipf')
plt.plot(Lista_Palabras[2].iloc[:,2], y_pred_lista2[2], color='blue', label='Regresión expo')
plt.plot(Lista_Palabras[2].iloc[:,2], y_pred_lista3[2], color='yellow', label='Regresión mandelbrot')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0,100)
plt.title('Comparación Modelos (Mansion de Araucaima)')
plt.legend()
plt.grid(True)
plt.show()

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
len(List_Words_Mutis[0])
len(Y_Predit_List_Mutis[0])

List_Words_Mutis_Predit = []

for i in range(len(List_Words_Mutis)):
    df = List_Words_Mutis[i].copy()
    y_pred = np.round(Y_Predit_List_Mutis[i],0)
    
    df['Conteo'] = y_pred
    
    List_Words_Mutis_Predit.append(df)

Txt_Mutis_Predit = []

for df in List_Words_Mutis_Predit:
    repeated_words = []
    
    for index, row in df.iterrows():
        repeated_words.extend([row['Elemento']] * int(row['Conteo']))
    
    Txt_Mutis_Predit.append(repeated_words)

num_remuestreos = 6000
lista_dataframes_predit = []

def contar_palabras(remuestreo):
    conteo = {}
    for palabra in remuestreo:
        conteo[palabra] = conteo.get(palabra, 0) + 1
    return conteo

for lista in Txt_Mutis_Predit:
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
    lista_dataframes_predit.append(dataframes_lista)

def calcular_entropia(dataframes):
    entropias = []
    for lista_dataframes_predit in dataframes:
        entropias_lista = []
        for df in lista_dataframes_predit:
            entropia = shannon_entropy(df)
            entropias_lista.append(entropia)
        entropias.append(entropias_lista)
    return entropias

# Calcular entropías
entropias_predit = calcular_entropia(lista_dataframes_predit)


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
lista_vectores_nueva = [construir_nuevo_vector(vector) for vector in Y_Predit_List_Mutis]

# Imprimir la lista de vectores nueva
for i, vector in enumerate(lista_vectores_nueva, start=1):
    print(f"Vector {i}: {vector}")
    
# %%
List_Words_Mutis[0]

# Tu DataFrame almacenado en List_Words_Mutis[0]
df = List_Words_Mutis[0]

# Especificar la ruta donde quieres guardar el archivo Excel
ruta_excel = "C:/Users/Brand/OneDrive - correounivalle.edu.co/Escritorio/Trabajos U/Trabajo de Grado/Base_Pa_R.xlsx"

# Guardar el DataFrame en un archivo Excel utilizando xlsxwriter como motor
df.to_excel(ruta_excel, engine='xlsxwriter', index=False)

