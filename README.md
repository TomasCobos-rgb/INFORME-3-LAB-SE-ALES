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
