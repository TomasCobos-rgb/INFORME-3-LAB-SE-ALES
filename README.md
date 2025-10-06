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
![SEÑAL DE VOZ HOMBRE]()

![ESPECTRO DE MAGNITUDES]()

![RESULTADO CARACTERISTICAS]()
