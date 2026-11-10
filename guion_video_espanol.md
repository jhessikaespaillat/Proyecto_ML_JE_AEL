# Guion Completo del Video Explicativo en Español (Duración Estimada: 5 Minutos)

**Proyecto Final:** Asistente de Inteligencia Artificial para la Planificación Docente  
**Programa:** Maestría en Inteligencia Artificial Aplicada a la Educación e Investigación  
**Asignatura:** Aprendizaje Automático (`IAE-2501`)  
**Docente:** Feibert Guzmán  
**Estudiante / Expositora:** Jhessika Lidabed Espaillat Espaillat  
**Institución:** Universidad Federico Henríquez y Carvajal (UFHEC)

---

## Estructura de Tiempos del Video (00:00 – 05:15)

| Bloque | Tiempo | Tema Principal | Recurso Visual en Pantalla |
| :--- | :--- | :--- | :--- |
| **1. Portada y Problema Educativo** | `00:00 – 00:45` | Contextualización, problema de reprobación/retiro y necesidad de planificación docente anticipada. | Diapositiva 1 / Portada de la Landing Page y Gráfico de Distribución OULAD (`fig1`). |
| **2. Dataset OULAD** | `00:45 – 01:30` | Descripción de `studentInfo.csv` (`32,593` registros), variables predictoras y variable objetivo binaria `target`. | Diapositiva 2 / Tabla de variables originales y esquema de clases `0` vs. `1`. |
| **3. Procesamiento (ETL Sin Fuga) y EDA** | `01:30 – 02:20` | Imputación/codificación en `Pipeline` tras `train_test_split`, perfilado con YData Profiling y hallazgos clave del EDA. | Diapositiva 3 / Cuaderno Jupyter `Proyecto_ML_JE_AEI.ipynb` (Etapas 3, 4 y 5) y figuras `fig2`, `fig3`, `fig4`. |
| **4. Modelos Entrenados y Métricas Reales** | `02:20 – 03:15` | Comparación entre `RandomForestClassifier`, `LogisticRegression` y `DecisionTreeClassifier`, validación cruzada y cohorte `2014J`. | Diapositiva 4 / Pestaña 1 de Streamlit con tabla de métricas, matrices de confusión (`fig5`) y curvas ROC (`fig6`). |
| **5. Demostración del Simulador en Streamlit** | `03:15 – 04:00` | Recorrido en vivo por la Pestaña 2 de `app.py`, selección de escenario e ingreso de características de un estudiante. | Diapositiva 5 / Pestaña 2 de la aplicación Streamlit (`app.py`). |
| **6. Interpretación de Predicción y Reglas Pedagógicas** | `04:00 – 04:40` | Distinción entre probabilidad del modelo y motor determinista de recomendaciones en las 7 dimensiones docentes. | Diapositiva 6 / Tarjetas desplegables de recomendaciones pedagógicas en Streamlit. |
| **7. Conclusiones, Ética y Limitaciones** | `04:40 – 05:15` | Síntesis de hallazgos, limitaciones del corte inicial (`studentInfo.csv`) y aviso ético de uso orientativo. | Diapositiva 7 / Cierre institucional UFHEC y aviso ético. |

---

## Narración Íntegra Paso a Paso (Voz en Off / Exposición en Español)

### Bloque 1: Presentación y Problema Educativo Abordado (`00:00 – 00:45`)
> **Narración:**  
> «Un cordial saludo al docente Feibert Guzmán y a la comunidad académica de la Universidad Federico Henríquez y Carvajal (UFHEC). Mi nombre es **Jhessika Lidabed Espaillat Espaillat**, participante de la Maestría en Inteligencia Artificial Aplicada a la Educación e Investigación, y presento el proyecto final de la asignatura Aprendizaje Automático (`IAE-2501`), titulado: **"Asistente de Inteligencia Artificial para la Planificación Docente"**.  
>  
> En la educación superior contemporánea, uno de los mayores desafíos para el cuerpo docente es que la detección de estudiantes en riesgo de reprobación o abandono suele ocurrir de manera reactiva, cuando el cuatrimestre ya ha concluido. Nuestro proyecto aborda esta problemática transformando los datos históricos de matrícula y contexto estudiantil en un sistema preventivo que apoya al docente desde el primer día de clases para planificar tutorías, recursos de refuerzo y evaluaciones formativas oportunas.»

---

### Bloque 2: El Dataset Open University Learning Analytics Dataset — OULAD (`00:45 – 01:30`)
> **Narración:**  
> «Para garantizar rigor empírico y reproducibilidad, utilizamos el conjunto de datos abierto **Open University Learning Analytics Dataset (OULAD)**, publicado por Kuzilek y colaboradores. Trabajamos específicamente sobre la tabla maestra `studentInfo.csv`, compuesta por **32,593 registros estudiantiles** distribuidos en siete módulos académicos y cuatro cohortes semestrales entre 2013 y 2014.  
>  
> Formulamos un problema de clasificación binaria supervisada a partir del resultado final del estudiante:  
> - La **Clase `0`** agrupa los casos de **fracaso académico o retiro** (*Fail* y *Withdrawn*), que representan el **52.79%** de la matrícula (**17,208 estudiantes**).  
> - La **Clase `1`** agrupa los casos de **éxito académico** (*Pass* y *Distinction*), equivalentes al **47.21%** (**15,385 estudiantes**).  
>  
> Como variables predictoras utilizamos diez características disponibles al inicio del curso: dos numéricas —el número de intentos fallidos previos (`num_of_prev_attempts`) y los créditos matriculados (`studied_credits`)— y ocho categóricas que incluyen módulo, cohorte, género, región geográfica, nivel educativo previo, banda de privación socioeconómica (`imd_band`), rango de edad y condición de discapacidad declarada.»

---

### Bloque 3: Procesamiento de Datos Sin Fuga, EDA y YData Profiling (`01:30 – 02:20`)
> **Narración:**  
> «En el cuaderno Jupyter definitivo implementamos una arquitectura de extracción, carga y transformación libre de fuga de información (*Data Leakage*). Primero separamos los datos de forma estratificada en un **80% para entrenamiento (26,074 registros)** y un **20% para prueba (6,519 registros)**. Únicamente sobre el conjunto de entrenamiento ajustamos el pipeline `ColumnTransformer`, imputando los **1,111 valores faltantes** de `imd_band` mediante la moda de entrenamiento, escalando las variables numéricas y aplicando codificación *One-Hot Encoding* que genera **56 atributos transformados**.  
>  
> El Análisis Exploratorio de Datos y el informe automatizado de **YData Profiling** revelaron tres hallazgos fundamentales para la pedagogía:  
> 1. Primero, el historial de intentos previos reduce drásticamente la tasa de aprobación: de un **49.4%** en quienes cursan por primera vez a menos del **34%** a partir del segundo intento fallido.  
> 2. Segundo, la combinación de sobrecarga de créditos inscritos y menor escolaridad formal previa incrementa el abandono.  
> 3. Y tercero, existe una brecha socioeconómica clara: los estudiantes en los deciles de mayor privación (`0-20%` del índice IMD) presentan tasas de éxito cercanas al **36%**, frente a más del **57%** en los deciles más favorecidos.»

---

### Bloque 4: Modelo Entrenado y Métricas Principales (`02:20 – 03:15`)
> **Narración:**  
> «Entrenamos como modelo principal un **Random Forest Classifier** regularizado con balanceo de clases por submuestra, y lo contrastamos rigurosamente frente a una **Regresión Logística** y un **Árbol de Decisión** individual.  
>  
> En el conjunto de prueba independiente de **6,519 estudiantes**, el modelo **Random Forest Classifier** alcanzó el mejor desempeño global en capacidad discriminativa, con un área bajo la curva **ROC-AUC de `0.680`**, una exactitud (**Accuracy**) cercana al **`63.6%`**, una **Balanced Accuracy de `63.6%`** y una sensibilidad (**Recall**) superior al **`61.8%`** para detectar a los estudiantes en riesgo de fracaso o retiro.  
>  
> Además, validamos el modelo mediante **Validación Cruzada Estratificada de 5 pliegues** (`ROC-AUC medio de 0.679`) y mediante una **evaluación por cohorte temporal**, entrenando con cohortes históricas y evaluando en la cohorte futura independiente **`2014J`** (`11,260 estudiantes`), donde el modelo mantuvo su estabilidad predictiva. El análisis de importancia de variables confirmó que los **créditos estudiados**, los **intentos previos**, el **nivel educativo previo** y el **módulo cursado** son los factores de mayor peso.»

---

### Bloque 5: Demostración del Simulador en Streamlit (`03:15 – 04:00`)
> **Narración:**  
> «Para transferir estos hallazgos al aula, desarrollamos una aplicación web modular en **Streamlit** organizada en dos pestañas. En la primera pestaña el docente consulta las gráficas exploratorias, el informe YData Profiling y las métricas verificadas del modelo.  
>  
> En la segunda pestaña se encuentra el **Simulador Inteligente de Planificación Docente**. Aquí podemos seleccionar escenarios preconfigurados o ingresar manualmente los datos de un estudiante. Por ejemplo, al evaluar a un estudiante inscrito en el módulo `DDD` con **120 créditos matriculados**, **2 intentos previos**, escolaridad inferior a *A Level*, índice socioeconómico `0-10%` y discapacidad declarada, el sistema valida los dominios, aplica el pipeline entrenado y calcula en tiempo real la estimación probabilística.»

---

### Bloque 6: Interpretación de la Predicción y Recomendaciones Pedagógicas (`04:00 – 04:40`)
> **Narración:**  
> «Para este caso, el modelo estima la **Clase `0` (Fracaso académico o retiro)** con una probabilidad de riesgo superior al **75%**, activando un nivel de **Alerta Temprana Crítica**.  
>  
> Es fundamental destacar que nuestra arquitectura separa explícitamente la **inferencia estadística del modelo** del **motor determinista de recomendaciones pedagógicas basado en reglas**. A partir de las condiciones del perfil, el motor activa recomendaciones concretas en **siete dimensiones docentes**:  
> 1. **Tutorías y acompañamiento individual** quincenal por contar con intentos previos.  
> 2. **Actividades de refuerzo** propedéutico por su nivel educativo de ingreso.  
> 3. **Recursos educativos complementarios** bajo Diseño Universal para el Aprendizaje (DUA) en atención a su condición de discapacidad.  
> 4. **Evaluaciones formativas** semanales de bajo impacto para gestionar sus 120 créditos.  
> 5. **Retroalimentación personalizada** temprana en las primeras dos semanas.  
> 6. **Seguimiento de la participación** en el entorno virtual considerando su contexto socioeconómico.  
> 7. Y **ajustes en la planificación docente** para flexibilizar su carga académica.»

---

### Bloque 7: Conclusiones y Limitaciones Éticas (`04:40 – 05:15`)
> **Narración:**  
> «En conclusión, el **Asistente de Inteligencia Artificial para la Planificación Docente** demuestra que es posible anticipar riesgos de reprobación y abandono desde el momento mismo de la matrícula utilizando analítica de aprendizaje reproducible.  
>  
> Reconocemos como limitación metodológica que las variables iniciales de matrícula explican una parte importante pero no total del rendimiento (`ROC-AUC de 0.680`); como trabajo futuro, la integración de los registros semanales de interacción en el entorno virtual (`studentVle.csv`) y las calificaciones parciales permitirá elevar la precisión conforme avanza el semestre.  
>  
> Finalmente, reafirmamos nuestro principio ético rector: **toda predicción generada por este sistema es una estimación probabilística orientativa sujeta a incertidumbre**. Nunca debe utilizarse para etiquetar, excluir ni tomar decisiones automáticas sobre un estudiante, sino como una brújula pedagógica al servicio del criterio humano del docente. ¡Muchas gracias!»

