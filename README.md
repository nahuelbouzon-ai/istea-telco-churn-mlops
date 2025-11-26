Este proyecto corresponde al trabajo de MLOps de la materia laboratorio de minería de datos de  ISTEA. La idea general es armar un pipeline completo y reproducible para predecir churn de clientes de telecomunicaciones usando buenas prácticas de versionado de datos, limpieza, entrenamiento de un modelo y ejecución automática mediante DVC.

El dataset utilizado es telco_churn.csv con alrededor de diez mil registros. La variable objetivo es churn (1 significa que el cliente se dio de baja y 0 que no). Las primeras etapas del proyecto incluyen toda la configuración del entorno, el versionado del dataset crudo, la generación del dataset limpio y el entrenamiento de un modelo baseline.

La estructura del repositorio es simple. Dentro de la carpeta data/raw se encuentra el dataset crudo versionado con DVC. En data/processed se guarda la versión limpia del dataset, también manejada por DVC. En models se guarda el modelo entrenado. En src están los scripts: uno para limpieza (data_prep.py) y otro para entrenamiento del modelo (train.py). El resto del proyecto contiene los archivos de configuración como .dvc, .gitignore, .dvcignore, requirements.txt y este README.

Los archivos csv y pkl reales no se suben al repositorio, sino que los maneja DVC, así que para usarlos hay que ejecutar dvc pull, que descarga todo directamente desde DagsHub.

Para instalar el entorno lo que hay que hacer es crear un environment de conda con python e instalar los paquetes listados en requirements.txt. Después de eso, simplemente usando dvc pull se recuperan el dataset crudo, el dataset procesado y el modelo baseline.

El pipeline está dividido en las primeras tres etapas del proyecto. En la etapa 1 se hace toda la parte de setup del proyecto, incluyendo Git, DVC y la integración con DagsHub, además del versionado del dataset crudo. En la etapa 2 se realiza la limpieza de datos usando el script data_prep.py. Ese script carga el dataset original, limpia columnas, ajusta tipos y valores nulos y guarda el archivo procesado. En la etapa 3 se ejecuta el script train.py, que carga el dataset limpio, prepara las features, entrena un modelo Logistic Regression y guarda tanto el modelo como las métricas generadas.

Para reproducir todo desde cero, cualquier persona puede clonar el repositorio desde GitHub, entrar a la carpeta del proyecto, instalar dependencias y luego ejecutar dvc pull para descargar todos los datos versionados. Después de eso puede volver a generar el dataset limpio con python src/data_prep.py o directamente re-entrenar el modelo con python src/train.py.

La reproducibilidad completa del proyecto funciona porque el pipeline está definido dentro de dvc.yaml, así que también se puede ejecutar dvc repro, que ejecuta automáticamente todos los stages definidos y genera el mismo dataset procesado, el mismo modelo y las mismas métricas que se generaron originalmente.

Los repositorios del proyecto son los siguientes:
GitHub: https://github.com/nahuelbouzon-ai/istea-telco-churn-mlops

DagsHub: https://dagshub.com/nahuel.bouzon/istea-telco-churn-mlops

En DagsHub se puede ver el versionado de datasets y modelos, y más adelante se usaría para visualizar experimentos adicionales si se continuara con el proyecto.
25/11