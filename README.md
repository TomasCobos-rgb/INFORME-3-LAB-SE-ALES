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
# 4. Aplicar el filtro (ejemplo con el femenino)
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
# 4. Aplicar el filtro (ejemplo con el femenino)
# =========================================================
voz_filtrada = lfilter(b_hombre, 1, data)
```
![SEÑAL DE VOZ HOMBRE FILTRADA](https://github.com/TomasCobos-rgb/INFORME-3-LAB-SE-ALES/blob/main/IMAGENES/SE%C3%91AL%20DE%20VOZ%20HOMRE%20FILTRADA.PNG?raw=true)
