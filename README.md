### INFORME DE LABORATORIO #3.
Análisis espectral de la voz
---------------
### OBJETIVOS

1. Capturar y procesar señales de voz masculinas y femeninas.
2. Aplicar la Transformada de Fourier como herramienta de análisis espectral de la
voz.
3. Extraer parámetros característicos de la señal de voz: frecuencia fundamental,
frecuencia media, brillo, intensidad, jitter y shimmer.
4. Comparar las diferencias principales entre señales de voz de hombres y mujeres
a partir de su análisis en frecuencia.
5. Desarrollar conclusiones sobre el comportamiento espectral de la voz humana
en función del género.

### PARTE A
En esta etapa se grabó una misma frase corta (≈5 s) pronunciada por seis personas (tres hombres y tres mujeres), manteniendo las mismas condiciones de muestreo (44.1 kHz, 16 bits) para asegurar la comparabilidad.
Cada archivo de voz fue guardado en formato .wav e importado en Python para su análisis en el dominio temporal y frecuencial mediante la Transformada Rápida de Fourier (FFT).
De cada señal se obtuvieron los siguientes parámetros:

1. Frecuencia fundamental (F₀)
2. Frecuencia media
3. Brillo espectral
4. Intensidad (energía)

Estos valores permiten comparar las características acústicas entre voces masculinas y femeninas y servirán como base para los análisis posteriores.

## Diagrama de flujo
![Diagrama Parte A](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/Beige%20Minimal%20Flowchart%20Infographic%20Graph.png?raw=true)
### PROCEDIMIENTO 
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# === 1. Cargar archivo de voz ===
fs, data = wavfile.read("HOMBRE-1.wav")

# Si el audio es estéreo, tomar un canal
if data.ndim > 1:
    data = data[:, 0]

# Convertir a float para cálculos
data = data.astype(float)

# === 2. Graficar en el dominio del tiempo ===
t = np.linspace(0, len(data)/fs, len(data))

plt.figure(figsize=(10, 4))
plt.plot(t, data, color="teal")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal de voz en el dominio del tiempo")
plt.grid()
plt.show()

# === 3. Calcular Transformada de Fourier ===
N = len(data)
Y = np.fft.fft(data)
f = np.fft.fftfreq(N, 1/fs)

# Solo frecuencias positivas
Y_mag = np.abs(Y[:N//2])
f_pos = f[:N//2]

plt.figure(figsize=(10, 4))
plt.plot(f_pos, Y_mag, color="darkorange")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.title("Espectro de magnitudes (FFT)")
plt.grid()
plt.show()

# === 4. Calcular características ===

# Frecuencia fundamental (pico principal)
fundamental_freq = f_pos[np.argmax(Y_mag[1:]) + 1]  # saltamos f=0

# Frecuencia media (ponderada)
freq_mean = np.sum(f_pos * Y_mag) / np.sum(Y_mag)

# Brillo (centroide espectral)
brillo = np.sum(f_pos * Y_mag) / np.sum(Y_mag)

# Intensidad (energía)
energia = np.sum(data**2)

# === 5. Mostrar resultados ===
print("===== Características de la voz =====")
print(f"Frecuencia fundamental (F0): {fundamental_freq:.2f} Hz")
print(f"Frecuencia media: {freq_mean:.2f} Hz")
print(f"Brillo espectral: {brillo:.2f} Hz")
print(f"Energía total: {energia:.2f}")
```

### RESULTADOS OBTENIDOS PARA HOMBRE
![SEÑAL DE VOZ HOMBRE](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/SE%C3%91AL%20DE%20VOZ%20HOMBRE.PNG?raw=true)

![ESPECTRO DE MAGNITUDES](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/ESPECTRO%20DE%20MAGNITUDES%20HOMBRE.PNG?raw=true)

![RESULTADO CARACTERISTICAS](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/CARACTERISTICAS%20DE%20LA%20VOZ.PNG?raw=true)

### RESULTADOS OBTENIDOS PARA MUJER

![SEÑAL DE VOZ MUJER](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/se%C3%B1al%20de%20voz%20mujer.PNG?raw=true)

![ESPECTRO DE MAGNITUDES](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/espectro%20magnitudes%20mujer.PNG?raw=true)

![RESULTADO CARACTERISTICAS](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/caracteristicas%20rta%20mujer.PNG?raw=true)

### PARTE B – Medición de Jitter y Shimmer 

En esta sección se realiza un análisis detallado de la estabilidad de la voz a partir de las grabaciones obtenidas en la Parte A. Se selecciona una muestra de voz masculina y una femenina, a las cuales se aplica un filtro pasa–banda FIR dentro del rango típico de frecuencias de cada género (80–400 Hz para hombres y 150–500 Hz para mujeres) con el fin de eliminar componentes de ruido no deseados.

Posteriormente, se evalúan las variaciones temporales y de amplitud de las señales de voz mediante el cálculo del Jitter (fluctuación en la frecuencia fundamental entre ciclos consecutivos) y el Shimmer (variación en la amplitud pico a pico). Estos parámetros permiten cuantificar la regularidad de la vibración de las cuerdas vocales y, por tanto, sirven como indicadores de calidad y estabilidad vocal.

## Diagrama de flujo
![Diagrama Parte A](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/Beige%20Minimal%20Flowchart%20Infographic%20Graph%20(1).png?raw=true)

### PARTA B.1

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin, lfilter, freqz

# =========================================================
# 1. Leer el archivo de audio
# =========================================================
fs, data = wavfile.read("MUJER-1.wav")   # Cambia por tu archivo de voz

# Si el audio está en estéreo, tomar un solo canal
if data.ndim > 1:
    data = data[:,0]

# Convertir a float y normalizar
data = data.astype(float)
if data.dtype == np.int16:
    data = data / 32768.0

# =========================================================
# 2. Parámetros del filtro FIR
# =========================================================
orden = 220
ventana = 'blackman'

# --- Filtro pasa-banda para voz masculina ---
f_low_h = 80
f_high_h = 400

# --- Filtro pasa-banda para voz femenina ---
f_low_m = 150
f_high_m = 500

# =========================================================
# 3. Diseño de los filtros FIR
# =========================================================
b_hombre = firwin(numtaps=orden+1, cutoff=[f_low_h, f_high_h],
                  window=ventana, pass_zero=False, fs=fs)

b_mujer = firwin(numtaps=orden+1, cutoff=[f_low_m, f_high_m],
                 window=ventana, pass_zero=False, fs=fs)


# =========================================================
# 4. Aplicar el filtro
# =========================================================
voz_filtrada = lfilter(b_mujer, 1, data)

# Guardar resultado
wavfile.write("voz_filtrada_femenina.wav", fs, (voz_filtrada * 32768).astype(np.int16))

# =========================================================
# 5. Visualizar señal original y filtrada (superpuestas)
# =========================================================
t = np.linspace(0, len(data)/fs, len(data))

plt.figure(figsize=(12,5))
plt.plot(t, data, color='gray', alpha=0.6, label='Señal original')
plt.plot(t, voz_filtrada, color='red', linewidth=1.2, label='Señal filtrada (150–500 Hz)')
plt.title("Comparación de señal original y filtrada (superpuestas)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================


```
### RESULTADOS OBTENIDOS
![SEÑAL DE VOZ MUJER FILTRADA](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/se%C3%B1al%20filtrada%20mujer.PNG?raw=true)

En este caso lo unico que cambia respecto al codigo anterior es la siguiente linea, la cual se encarga de aplicar el filtro a una voz de hombre: 
```python
# =========================================================
# 4. Aplicar el filtro 
# =========================================================
voz_filtrada = lfilter(b_hombre, 1, data)
```
![SEÑAL DE VOZ HOMBRE FILTRADA](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/SE%C3%91AL%20DE%20VOZ%20HOMRE%20FILTRADA.PNG?raw=true)

### PARTE B.2 
#### PROCEDIMIENTO 
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
import pandas as pd

# =========================================================
# Función para calcular jitter y shimmer
# =========================================================
def calcular_jitter_shimmer(ruta_archivo, graficar=False):
    fs, data = wavfile.read(ruta_archivo)
    data = data.astype(float)

    # Normalizar a [-1, 1]
    if data.dtype == np.int16:
        data = data / 32768.0
    if data.ndim > 1:
        data = data[:, 0]

    # Detectar picos principales (vibraciones)
    peaks, _ = find_peaks(data, height=0, distance=fs/500)

    if len(peaks) < 3:
        return 0, 0, 0, 0

    tiempos = peaks / fs
    Ti = np.diff(tiempos)       # Periodos sucesivos
    Ai = data[peaks]            # Amplitudes en los picos

    # Jitter
    N = len(Ti)
    jitter_abs = np.sum(np.abs(Ti[:-1] - Ti[1:])) / (N - 1)
    jitter_rel = (jitter_abs / np.mean(Ti)) * 100

    # Shimmer
    M = len(Ai)
    shimmer_abs = np.sum(np.abs(Ai[:-1] - Ai[1:])) / (M - 1)
    shimmer_rel = (shimmer_abs / np.mean(Ai)) * 100

    # Graficar si se desea
    if graficar:
        t = np.arange(len(data)) / fs
        plt.figure(figsize=(12, 5))
        plt.plot(t, data, color='gray', label='Señal de voz')
        plt.plot(peaks/fs, data[peaks], 'ro', label='Picos detectados')
        plt.title(f'Picos de vibración - {ruta_archivo}')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return jitter_abs, jitter_rel, shimmer_abs, shimmer_rel


# =========================================================
# 2. Archivos a analizar
# =========================================================
archivos = [
    "HOMBRE-1.wav",
    "HOMBRE-2.wav",
    "HOMBRE-3.wav",
    "MUJER-1.wav",
    "MUJER-2_1.wav",
    "MUJER-3_1.wav"
]

# =========================================================
# 3. Calcular y almacenar resultados
# =========================================================
resultados = []
for archivo in archivos:
    jitter_abs, jitter_rel, shimmer_abs, shimmer_rel = calcular_jitter_shimmer(archivo)
    resultados.append({
        "Archivo": archivo,
        "Jitter_abs (s)": jitter_abs,
        "Jitter_rel (%)": jitter_rel,
        "Shimmer_abs": shimmer_abs,
        "Shimmer_rel (%)": shimmer_rel
    })

# Convertir a DataFrame para visualizar tabla
df_resultados = pd.DataFrame(resultados)
print("\n=== RESULTADOS DE JITTER Y SHIMMER ===")
print(df_resultados.round(6))

# =========================================================
# 4. (Opcional) Guardar resultados en CSV
# =========================================================
df_resultados.to_csv("resultados_jitter_shimmer.csv", index=False)
print("\nResultados guardados en 'resultados_jitter_shimmer.csv'")
```

### RESULTADOS SHIMMER Y JITTER

![RESUTALDOS SHIMMER Y JITTER](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/RESULTADOS%20SHIMMER%20Y%20JITTER.PNG?raw=true)

###  PARTE C.
####  COMPARACIÓN Y CONCUSIONES
En esta sección se realiza la comparación de los resultados obtenidos entre las voces masculinas y femeninas analizadas previamente. Se observan diferencias en la frecuencia fundamental, el brillo, la frecuencia media y la intensidad, con el fin de identificar las características acústicas que distinguen ambos tipos de voz.

Además, se discute la importancia clínica de los parámetros de Jitter y Shimmer, los cuales permiten evaluar la estabilidad y regularidad de la producción vocal. Estos indicadores son especialmente útiles en el diagnóstico de alteraciones fonatorias y en el seguimiento terapéutico de pacientes con trastornos de la voz.

- ¿Qué diferencias se observan en la frecuencia fundamental?:
En las mediciones realizadas se analiza que la frecuencia fundamental (F₀) de la voz masculina (HOMBRE-1) fue de 227.79 Hz, mientras que la de la voz femenina (MUJER-2) alcanzó 447.75 Hz.
Estos resultados señalan que la voz de la mujer vibra aproximadamente el doble de rápido que la del hombre. Esto quiere decir que la diferencia se debe a las características anatómicas de las cuerdas vocales de cada género: las de los hombres son más largas y gruesas, produciendo vibraciones más lentas y tonos más graves; mientras que las de las mujeres son más cortas y delgadas, generando una vibración más rápida y, por lo tanto, un tono más agudo. Además de que tipicamente la voz masculina se encuentra entre el rango de 80Hz a 150Hz, mientras que la voz femenina se encuentra entre el rango de 180Hz a 260Hz, lo cual nos da a entender que los resultados obtenidos poseen la trayectoria correcta.

- ¿Qué otras diferencias notan en términos de brillo, media o intensidad?:
En cuanto al brillo espectral y la frecuencia media, la voz masculina presentó valores de 2703.02 Hz, mientras que la femenina mostró 1653.18 Hz.
Aunque normalmente las voces femeninas tienden a sonar más brillantes, en este caso el hombre tuvo un valor de brillo mayor. Esto puede deberse al tipo de emisión de voz o al esfuerzo vocal durante la grabación, que pudo realzar los armónicos superiores.
Respecto a la intensidad o energía total, la voz femenina (2.09×10¹²) fue ligeramente superior a la del hombre (1.50×10¹²), lo cual indica que la grabación de la mujer tuvo una amplitud mayor o una proyección de voz más fuerte. En resumen, la voz masculina mostró mayor brillo, mientras que la voz femenina destacó por su energía e intensidad.
  
- Redactar conclusiones sobre el comportamiento de la voz en hombres y mujeres a partir de los análisis realizados:
1. La frecuencia fundamental es la característica clave entre géneros, siendo claramente más alta en las mujeres, lo que coincide con el tono más agudo percibido en sus voces.
2. Las voces masculinas mostraron una mayor variabilidad en frecuencia y amplitud, reflejada en los valores más altos de jitter y shimmer. Esto sugiere una vibración general menos regular, probablemente por diferencias anatómicas o por la forma de producir el sonido.
3. Las voces femeninas, fueron más estables, con menor jitter y shimmer, lo que las hace percibirse como más claras y uniformes.
4. Tanto el brillo como la intensidad pueden variar por factores individuales como la técnica vocal, el esfuerzo y las condiciones del entorno, no siendo necesariamente exclusivos de un género.
5. En general, se confirma que las voces femeninas tienen una vibración más rápida y estable, mientras que las masculinas tienden a tener tonos más graves y con mayor variación en sus ciclos vocales.
  
- Discuta la importancia clínica del jitter y shimmer en el análisis de la voz:
El jitter y el shimmer son parámetros clínicos fundamentales para evaluar la calidad y estabilidad de la voz.
1. El jitter mide las variaciones pequeñas en la frecuencia fundamental entre un ciclo de vibración y otro. Un valor elevado puede indicar alteraciones en el control muscular laríngeo o presencia de trastornos vocales, como disfonía, nódulos o fatiga vocal.
2. El shimmer evalúa la variación en la amplitud de los ciclos. Un shimmer alto puede relacionarse con cierre glotal incompleto, irregularidades en las cuerdas vocales o problemas respiratorios durante la fonación.

En el ámbito clínico, estos indicadores son muy útiles para detectar disfunciones laríngeas, realizar diagnósticos tempranos y monitorear la evolución de tratamientos fonoaudiológicos o quirúrgicos. En personas con una voz sana, los valores de jitter y shimmer suelen ser bajos y estables, mientras que incrementos notables en estos parámetros pueden señalar la presencia de alguna alteración vocal.
