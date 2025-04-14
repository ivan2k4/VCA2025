# Informe de Práctica: Clasificación de Eventos de Interés en Entornos Portuarios

## Introducción

Esta práctica se enmarca en el contexto de plataformas inteligentes para la gestión de infraestructuras portuarias, en particular en la integración de un sistema de videovigilancia automatizado dentro del concepto Smartports. Uno de los retos operativos más relevantes es la identificación precisa y eficiente de eventos de interés en el entorno del muelle, como la presencia o atraco de embarcaciones. A partir del análisis de imágenes provenientes de cámaras CCTV, se busca implementar un sistema de clasificación que sea capaz de detectar estos eventos con alta fiabilidad.

Para ello, disponemos de un conjunto de datos que incluye 294 imágenes tomadas en diferentes condiciones de luz y perspectiva, etiquetadas tanto con la presencia de barcos como con su estado de atracado. A partir de estas imágenes, se propone el entrenamiento de modelos de clasificación binaria utilizando redes neuronales profundas, explorando diferentes configuraciones de entrenamiento, uso de modelos preentrenados, y técnicas de aumento de datos. Este informe presenta el enfoque seguido, los modelos desarrollados y los resultados obtenidos.

---

## Desarrollo de la Tarea 1: Implementación del Dataset

La primera parte de la práctica consistió en la creación de una clase `Dataset` personalizada, utilizando PyTorch, que permitiera una carga eficiente y flexible de los datos de entrada. Esta clase fue diseñada para funcionar tanto con las etiquetas de presencia de barco (`ship`) como con las de barco atracado (`docked`), permitiendo conmutar fácilmente entre uno y otro objetivo de clasificación.

El diseño de la clase permite cargar las imágenes desde directorios estructurados, leer las etiquetas desde ficheros CSV, y aplicar transformaciones tanto básicas como de aumento de datos. Esta flexibilidad fue fundamental para facilitar las distintas configuraciones de entrenamiento exploradas más adelante. La integración del parámetro `transform` en la clase permitió aplicar estrategias de `data augmentation` de forma controlada, sin tener que duplicar lógica de preprocesado.

Desde una perspectiva crítica, este diseño modular resultó ser una buena decisión, ya que permitió replicar el mismo pipeline tanto para la tarea de clasificación de barcos en escena como para la de barcos atracados, cambiando únicamente el archivo de etiquetas y las transformaciones.

---

## Desarrollo de la Tarea 2: Clasificación Ship / No-ship

La segunda parte de la práctica tuvo como objetivo desarrollar un modelo capaz de identificar la presencia o ausencia de barcos en las imágenes. Se utilizó como base la arquitectura SqueezeNet1_0, la cual escogimos por su balance entre eficiencia computacional y capacidad de aprendizaje.

Se plantearon cuatro configuraciones de entrenamiento diferentes:

1. Entrenamiento desde cero, sin técnicas de data augmentation.
2. Entrenamiento desde cero, utilizando data augmentation.
3. Refinamiento de un modelo preentrenado (transfer learning), sin data augmentation.
4. Refinamiento de un modelo preentrenado, con data augmentation.

El uso de `data augmentation` se diseñó cuidadosamente considerando las características del entorno portuario. Se aplicaron transformaciones como giros horizontales aleatorios, ajustes de brillo y contraste, y rotaciones leves, con el objetivo de simular las variaciones naturales de punto de vista, iluminación y condiciones atmosféricas que se encuentran en entornos reales.

Los resultados obtenidos muestran de forma clara que el uso de modelos preentrenados proporciona una ventaja significativa, especialmente cuando se combinan con técnicas de data augmentation. El modelo preentrenado con `data augmentation` logró no solo una mayor precisión, sino también una mejor capacidad de generalización en el conjunto de validación, demostrando una menor pérdida y mayor estabilidad durante el entrenamiento.

### Resultados de la clasificación Ship / No-ship

| Configuración                         | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|--------------------------------------|--------------|----------------|-------------|----------------|
| Desde cero - Sin Augmentation        | 74.58        | 81.25          | 74.29       | 77.61          |
| Desde cero - Con Augmentation        | 81.36        | 83.33          | 85.71       | 84.51          |
| Preentrenado - Sin Augmentation      | 91.53        | 87.50          | 100.00      | 93.33          |
| Preentrenado - Con Augmentation      | 91.53        | 89.47          | 97.14       | 93.15          |

---

## Desarrollo de la Tarea 4 (opcional): Clasificación Docked / Undocked

También hemos realizado la tarea de clasificación del estado de atracado de las embarcaciones. Esta tarea presenta un nivel adicional de complejidad respecto a la detección de barcos, ya que requiere no solo reconocer la presencia de un barco, sino también detectar su posición relativa al muelle, lo cual puede ser menos evidente visualmente, especialmente en imágenes con ángulos lejanos o de baja resolución.

Reutilizamos el pipeline diseñado para la Tarea 2, adaptándolo al nuevo conjunto de etiquetas (`docked.csv`). De nuevo, exploramos las mismas cuatro configuraciones (entrenamiento desde cero y transferencia de aprendizaje, con y sin `data augmentation`).

Los resultados obtenidos siguieron una tendencia similar: los modelos preentrenados con `data augmentation` generalmente alcanzaron mejores métricas, aunque las diferencias entre las distintas configuraciones fueron ligeramente menores que en la Tarea 2. Lo cual puede deberse a la complejidad de la tarea o a la posible ambigüedad en algunas imágenes sobre el estado real de atraco.

### Resultados de la clasificación Docked / Undocked

| Configuración                         | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) |
|--------------------------------------|--------------|----------------|-------------|----------------|
| Desde cero - Sin Augmentation        | 67.57        | 66.67          | 73.68       | 70.00          |
| Desde cero - Con Augmentation        | 78.38        | 76.19          | 84.21       | 80.00          |
| Preentrenado - Sin Augmentation      | 72.97        | 69.57          | 84.21       | 76.19          |
| Preentrenado - Con Augmentation      | 78.38        | 78.95          | 78.95       | 78.95          |

Teniendo en cuenta los resultados obtenidos, visiblemente inferiores a los de la Tarea 2 hemos indagado qué métodos podrían mejorarlos, aunque no los hemos llevado a cabo debido a la complejidad de los mismos.
Como el método más oportuno nos gusaría resaltar un etiquetado más detallado de las imágenes, en el que dividiríamos las imágenes entre ejemplos más "claros" con las que realizaríamos un primer entrenamiento, para después realizar un segundo entrenamiento con las imágenes más ambiguas.
Otras opciones sería emplear arquitecturas más potentes que SqueezNet (pero resultaría en una pérdida de la eficiencia), segmentar varias zonas en la imágen para separar el muelle del barco (que sería nuestra región de interés), o emplear imágenes con una mayor resolución que 128x128.

---

## Conclusiones

Esta práctica nos ha permitido diseñar e implementar un sistema de clasificación visual robusto para tareas críticas en el entorno portuario. La modularidad del pipeline y la correcta utilización de buenas prácticas en el preprocesado, partición de datos y entrenamiento han resultado claves para la obtención de buenos resultados.

Entre las principales conclusiones de esta práctica, se destaca que el uso de modelos preentrenados ha supuesto una mejora clara con respecto al entrenamiento desde cero. Estos modelos, al haber sido entrenados previamente sobre conjuntos de datos extensos, ofrecen una base sólida que permite obtener buenos resultados incluso en contextos con volúmenes de datos reducidos, como es el caso de este dataset portuario.

De la misma manera, las técnicas de aumento de datos han demostrado ser especialmente efectivas. Al introducir transformaciones que simulan variaciones típicas del entorno portuario (como cambios en la iluminación, ángulos de visión o rotaciones) hemos logrado aumentar la capacidad de generalización de los modelos, reduciendo el riesgo de sobreajuste y mejorando el rendimiento en los conjuntos de validación y prueba.

Por último, cabe mencionar que la tarea de detección de barcos atracados, a pesar de ser más compleja que la detección de presencia de barcos, puede abordarse de forma eficaz mediante una adaptación cuidadosa del mismo enfoque. Aunque los resultados en esta segunda tarea han sido más modestos, la metodología empleada ha demostrado ser válida y nos ofrece una base más que prometedora sobre la que seguir mejorando.

Debido a la dependencia de los datos para el entrenamiento en esta práctica, supondría una gran mejora en los resultados contar con una mayor cantidad de imágenes y con una mejor resolución, así como variedad de escenarios. Si los recursos nos lo permitieran también nos ayudaría experimentar con arquitecturas más avanzadas que pueden capturar relaciones espaciales más complejas.