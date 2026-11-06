# Cartografía panóptica de Bucaramanga

Proyecto para adaptar las anotaciones, entrenar y comparar Mask2Former y
EfficientPS con el dataset de cartografía automatizada. Esta guía describe el
entorno de anotación, entrenamiento y exploración para Ubuntu 22.04.

## Inicio rápido: instalar y ejecutar Streamlit

Estos pasos parten de una copia nueva del repositorio. Los comandos se ejecutan
desde la carpeta raíz `cartografia_panoptica`; no es necesario tener el dataset
original para probar la aplicación. El repositorio incluye las anotaciones
panópticas procesadas y el reporte. Los pesos actuales se descargan por separado
desde una GitHub Release para evitar incluir archivos de cientos de megabytes en
el historial Git.

Clona el proyecto e ingresa a la carpeta:

```bash
cd "$HOME"
git clone https://github.com/JuanChaustre/cartografia_panoptica.git
cd "$HOME/cartografia_panoptica"
```

En Ubuntu 22.04 instala Python, `venv` y las bibliotecas del sistema:

```bash
cd "$HOME/cartografia_panoptica"
sudo apt update
sudo apt install -y python3 python3-venv python3-tk git curl
```

Crea el entorno e instala PyTorch. Elige un solo comando de instalación según
el equipo:

```bash
cd "$HOME/cartografia_panoptica"
python3 -m venv .venv-mask2former
.venv-mask2former/bin/python -m pip install --upgrade pip
# Para equipos NVIDIA:
.venv-mask2former/bin/python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Para una instalación solo CPU, ejecuta este comando en lugar del anterior:

```bash
cd "$HOME/cartografia_panoptica"
.venv-mask2former/bin/python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

Instala el resto de las dependencias de Mask2Former y Streamlit:

```bash
cd "$HOME/cartografia_panoptica"
.venv-mask2former/bin/python -m pip install \
  -r training/requirements-mask2former.txt \
  -r requirements-streamlit.txt
```

Descarga y verifica los pesos de las dos ejecuciones comparadas. Se requieren
conexión a Internet y `sha256sum`:

```bash
cd "$HOME/cartografia_panoptica"
bash tools/descargar_modelos.sh
```

Inicia la aplicación:

```bash
cd "$HOME/cartografia_panoptica"
.venv-mask2former/bin/python -m streamlit run streamlit_app.py
```

Abre `http://localhost:8501`. Mask2Former también puede ejecutarse en CPU;
EfficientPS original requiere una GPU NVIDIA y su entorno legado independiente.
Para usar ambos modelos en Ubuntu instala ese entorno antes de iniciar Streamlit:

```bash
cd "$HOME/cartografia_panoptica"
env -u PYTHONPATH bash tools/instalar_efficientps_original.sh
```

El dataset original no se versiona. Para entrenar, descarga el ZIP desde el
notebook incluido, extrae `reto-1-cartografa-automatizada` en la carpeta
inmediatamente superior al repositorio y sigue la sección
[Entrenamiento y evaluación](#entrenamiento-y-evaluación).

## Estado y formato del dataset

- `Dataset segmentacion` contiene 100 imágenes RGB `recorte_N.png` de 500×500 y
  100 máscaras `ground_truth_recorte_N.npy` booleanas de forma `(500, 500, 4)`.
- Los cuatro canales son área caminable, área construida, zona verde y carretera.
- Esas máscaras son etiquetas semánticas por píxel; no incluyen IDs de instancia.
- Las clases se solapan. El conversor aplica la prioridad acordada
  (`zona verde > construida > caminable > carretera`) para producir una máscara
  exclusiva. Esa regla resuelve la propiedad de píxeles compartidos, pero no
  recupera límites de instancia ausentes.
- El conversor identifica componentes conectados como instancias thing y unifica
  la carretera como stuff. Si dos polígonos de la misma clase se tocan o quedaron
  unidos en la máscara, se producirán como una sola instancia y requieren
  revisión/corrección manual.
- `Recortes adicionales` contiene imágenes sin máscaras. Se reservan para
  inferencia, no para entrenamiento supervisado.
- `data_panoptic_auto` contiene las anotaciones preliminares generadas
  localmente. La versión revisada con acera como stuff se distribuye en
  `data_panoptic_stuff`; la validación estructural más reciente pasó con 100
  imágenes y 4.908 segmentos. Esa validación comprueba formato e IDs, no la
  calidad semántica de cada polígono.

## Definición panóptica acordada

| Categoría | Interpretación para anotar | Tipo |
|---|---|---|
| Acera / área caminable | La clase de región acera; todas sus zonas de una imagen se agrupan | Stuff |
| Manzana construida | Cada manzana es una instancia | Thing |
| Zona verde | Cada zona/parche verde es una instancia | Thing |
| Carretera | Una región de carretera por imagen | Stuff |

Una instancia debe dibujarse como polígono independiente. No se debe aceptar una
etiqueta que una dos manzanas o dos parches verdes distintos solo porque
compartan clase. Acera y carretera son stuff: pueden incluir regiones
desconectadas, agrupadas como un segmento por clase e imagen. Las máscaras
`.npy` se muestran únicamente como referencia tenue; el anotador debe decidir
los límites y resolver las superposiciones.

## Conversión automática inicial

Se puede generar un COCO Panoptic preliminar sin alterar las máscaras originales
ni las anotaciones manuales. El conversor hace mutuamente excluyentes las clases
por prioridad (verde > construida > caminable > carretera), asigna un ID a cada
componente conectado de las tres clases thing y agrupa toda la carretera de cada
imagen en un solo segmento stuff. Los píxeles sin etiqueta quedan como void.

Desde la raíz del repositorio:

```bash
cd "$HOME/cartografia_panoptica"
.venv/bin/python tools/convertir_semantico_a_panoptico.py \
  --dataset ../reto-1-cartografa-automatizada \
  --output ./data_panoptic_auto
```

La conectividad por defecto es 8 (también se puede especificar `--connectivity 4`).
Si la carpeta de salida ya contiene archivos, el conversor se detiene para evitar
reemplazarlos; revisa la ruta antes de usar `--overwrite`.

Valida la salida:

```bash
cd "$HOME/cartografia_panoptica"
.venv/bin/python tools/validar_anotaciones.py \
  --dataset ../reto-1-cartografa-automatizada \
  --annotations ./data_panoptic_auto/annotations/panoptic.json
```

Abre el anotador sobre esa salida para revisar y corregir las instancias
generadas, sin regenerar las máscaras:

```bash
cd "$HOME/cartografia_panoptica"
.venv/bin/python tools/anotar_instancias.py \
  --dataset ../reto-1-cartografa-automatizada \
  --output ./data_panoptic_auto
```

Luego inspecciona las máscaras visualmente. La prioridad es una regla acordada
para resolver solapamientos, no información observada sobre alturas reales. Los
componentes conectados son solo una aproximación: una región partida por la
prioridad puede crear instancias adicionales, y polígonos contiguos de la misma
clase pueden fusionarse. Corrige estos casos con el anotador antes de considerar
las etiquetas como ground truth para métricas o entrenamiento.

## Conversión de acera a stuff

Las anotaciones originales de `data_panoptic_auto` se conservan. Para preparar
la ontología nueva, convierte acera (categoría 1) a stuff; el script genera una
copia derivada y fusiona todos los segmentos de acera de cada imagen en un
segmento:

```bash
cd "$HOME/cartografia_panoptica"
env -u PYTHONPATH .venv-mask2former/bin/python \
  tools/convertir_clase_a_stuff.py \
  --annotations data_panoptic_auto/annotations/panoptic.json \
  --output data_panoptic_stuff \
  --category-id 1 --category-name acera
```

La conversión copia también el `splits.json` para mantener los mismos IDs en
train/validation/test. Verifica la copia sin alterar los originales:

```bash
cd "$HOME/cartografia_panoptica"
env -u PYTHONPATH .venv-mask2former/bin/python \
  tools/validar_anotaciones.py \
  --dataset ../reto-1-cartografa-automatizada \
  --annotations data_panoptic_stuff/annotations/panoptic.json
```

La copia migrada conserva los mapas de clase por píxel: se verificaron las 100
máscaras y no hubo píxeles con cambios de categoría. Las 2.257 regiones de acera
originales se agrupan en 99 segmentos (uno por imagen con acera); la imagen
restante no contiene esa clase.

Este cambio redefine la tarea: mAP de instancia deja de incluir acera; cada
imagen tiene como máximo un segmento stuff de acera, aunque sus píxeles estén
desconectados. No modifica las geometrías anotadas, solo agrupa sus IDs. Las
métricas y checkpoints anteriores pertenecen a la definición anterior y no
deben compararse como si fueran la misma tarea.

## Anotación manual

El anotador escribe máscaras RGB codificadas por ID y un archivo COCO Panoptic:

```text
data_panoptic/
└── annotations/
    ├── panoptic.json
    └── panoptic/
        ├── recorte_1.png
        └── ...
```

## Entorno Ubuntu 22.04

El proyecto se probó en Ubuntu 22.04 con Python 3.10, Tkinter y una NVIDIA RTX
3050 Laptop con 4 GB de VRAM. `requirements.txt` contiene las dependencias del
notebook, la descarga opcional desde Kaggle y el anotador. Las dependencias de
entrenamiento se instalan en entornos separados para aislar las versiones de
PyTorch y sus bibliotecas de modelos.

Desde la raíz del repositorio:

```bash
sudo apt update
sudo apt install -y python3-venv python3-tk
cd "$HOME/cartografia_panoptica"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user \
  --name cartografia-panoptica \
  --display-name "Python (Cartografía panóptica)"
```

Selecciona el kernel **Python (Cartografía panóptica)** para
`Visualización Dataset Cartografía Automatizada.ipynb`, que está en la raíz del
repositorio. El notebook descarga y extrae el dataset junto a la carpeta del
proyecto. Vuelve a ejecutar sus celdas para actualizar rutas y visualizaciones.

Para descargar el ZIP desde Kaggle, se necesita una cuenta y el archivo
`~/.config/kaggle/kaggle.json` (o `~/.kaggle/kaggle.json` en versiones
anteriores de la CLI). Protege las credenciales con:

```bash
cd "$HOME/cartografia_panoptica"
chmod 600 ~/.config/kaggle/kaggle.json
```

Desde la raíz del repositorio, activa el entorno y ejecuta el anotador:

```bash
cd "$HOME/cartografia_panoptica"
source .venv/bin/activate
python tools/anotar_instancias.py \
  --dataset ../reto-1-cartografa-automatizada \
  --output ./data_panoptic
```

Controles:

- Seleccionar categoría y dibujar con clic izquierdo.
- `Enter` o clic derecho aplica el polígono.
- `Borrar / ignorar` elimina el área del polígono y la deja como void.
- `Ctrl+S` guarda la imagen y actualiza `panoptic.json`.
- `Anterior` y `Siguiente` navegan por las 100 imágenes anotables.

Anota cada imagen y guarda el progreso con frecuencia. Los PNG se guardan por
imagen y el JSON se actualiza atómicamente, por lo que un cierre no elimina las
anotaciones ya guardadas. Se puede cerrar y volver a abrir el anotador para
continuar.

Valida la exportación antes de usarla:

```bash
cd "$HOME/cartografia_panoptica"
python tools/validar_anotaciones.py \
  --dataset ../reto-1-cartografa-automatizada \
  --annotations ./data_panoptic/annotations/panoptic.json
```

La validación comprueba que cada imagen anotada tenga máscara RGB válida, que los
IDs del PNG y `segments_info` coincidan, y que las imágenes referenciadas existan.
No determina si un polígono representa correctamente una manzana; se necesita
revisión visual humana.

## Preparación de Mask2Former y EfficientPS

El sistema dispone de una RTX 3050 Laptop con 4 GB de VRAM y driver NVIDIA
580.178.04 (CUDA 13.0 reportado por el driver). Mask2Former usa PyTorch 2.5.1
con CUDA 12.1. El código oficial de EfficientPS se conserva en
`third_party/EfficientPS` (DeepSceneSeg/EfficientPS, revisión
`e1c92c301b8d2a9c582797ab3cad203909f2fa9d`, licencia GPL-3.0) y se prepara en un
entorno separado con Python 3.7, PyTorch 1.9.0/CUDA 11.1 y GCC 8. El código
oficial recomienda PyTorch 1.7/CUDA 10.2, pero esa combinación no ejecuta
kernels en la RTX 3050 (sm_86); se usa la combinación 1.9.0/0.10.0 con CUDA
11.1 para soportar esta GPU. Es una adaptación experimental del entorno, no
una afirmación de que el stack haya sido validado por los autores. Para caber en
4 GB, su configuración de cartografía usa EfficientNet-B0 y resolución 384×384;
es el código/arquitectura oficiales con un backbone reducido, no la configuración
EfficientNet-B5 del resultado publicado.

El instalador compila las extensiones para sm_86 y ejecuta una prueba NMS en
GPU; solo continúa si esa extensión funciona. No se mezclan las dependencias
antiguas con el entorno de Mask2Former.

### Entorno de Mask2Former

Mask2Former usa la implementación de Hugging Face del modelo en un entorno
aislado Python 3.10; las ruedas PyTorch incluyen CUDA 12.1, compatible con el
driver instalado. EfficientPS original usa el entorno legado descrito arriba.

Desde la raíz del repositorio, crea el entorno:

```bash
cd "$HOME/cartografia_panoptica"
python3 -m venv .venv-mask2former
.venv-mask2former/bin/python -m pip install --upgrade pip
.venv-mask2former/bin/python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
.venv-mask2former/bin/python -m pip install \
  -r training/requirements-mask2former.txt \
  -r requirements-streamlit.txt
```

Verifica `torch.cuda.is_available()` antes de entrenar. Esta comprobación solo
confirma que CUDA funciona, no que el modelo quepa en 4 GB.

### EfficientPS oficial

El repositorio original y su licencia están en
[`third_party/EfficientPS`](./third_party/EfficientPS).
El instalador crea el entorno legado aislado, compila las extensiones CUDA del
repositorio y prueba NMS en la GPU:

```bash
cd "$HOME/cartografia_panoptica"
env -u PYTHONPATH bash tools/instalar_efficientps_original.sh
```

La creación/compilación puede tardar y descargar varios GB. Si falla la prueba
NMS con un error de arquitectura CUDA, no ejecutes entrenamiento: el stack
original no quedó operativo en esta RTX 3050. El código de entrenamiento y la
conversión de datos siguen disponibles para ejecutarse en una GPU compatible.
El instalador obtiene el runtime conda y el compilador `nvcc` del instalador
oficial CUDA 11.1.1 (3,53 GB), verifica tamaño y MD5 del archivo y lo instala solo
dentro del entorno del proyecto; no modifica el CUDA del sistema ni los
entornos de los otros modelos. El archivo temporal se elimina al completar esa
instalación.
Por compatibilidad con el compilador legado, el entorno se crea por defecto en
`~/.cache/cartografia-panoptica/efficientps-original`, fuera de la ruta del
proyecto (que contiene espacios). `EFFICIENTPS_ENV_PREFIX` permite elegir otra
ruta sin espacios.

## Particiones del dataset

Con el archivo panóptico revisado y validado, crea una partición reproducible
por imagen (70 % train, 15 % validation, 15 % test; semilla 42):

```bash
cd "$HOME/cartografia_panoptica"
.venv-mask2former/bin/python tools/crear_particiones.py \
  --annotations ./data_panoptic_auto/annotations/panoptic.json \
  --output ./data_panoptic_auto/splits.json
```

El archivo de particiones se comparte entre ambos modelos para que las métricas
se comparen con los mismos ejemplos. La división es aleatoria por imagen; si
los recortes representan ubicaciones contiguas o solapadas, una partición por
grupos espaciales debe preferirse para evitar fuga entre train y test.

## Entrenamiento y evaluación

Después de convertir acera a stuff, los scripts usan
`data_panoptic_stuff/annotations/panoptic.json` y `splits.json` por defecto.
Revalida ese JSON después de cambiar anotaciones; no entrenes ni reportes
métricas con componentes automáticos sin revisar. Las imágenes se buscan en
`../reto-1-cartografa-automatizada` desde la raíz del repositorio;
cambia `--dataset-root` si tu extracción está en otra ruta.

Mask2Former inicia a 256×256 para limitar el uso de memoria de la RTX 3050 de
4 GB y usa lote 1 con AMP en CUDA. El EfficientPS oficial usa lote 1, 384×384,
un máximo de 40 épocas y early stopping con paciencia 8 sobre `validation_loss`.
Conserva el checkpoint con menor pérdida de validation. La evaluación de test
se ejecuta al terminar y no interviene en la selección del checkpoint.

Entrena desde cero con la nueva ontología. Usa estos comandos desde la raíz del
repositorio; `--epochs 40` es el máximo total de épocas y `--patience
8` activa early stopping. No reanudes los checkpoints anteriores, ya que fueron
entrenados con acera como thing:

```bash
cd "$HOME/cartografia_panoptica"
env -u PYTHONPATH .venv-mask2former/bin/python \
  training/train_mask2former.py \
  --epochs 40 --patience 8

env -u PYTHONPATH bash tools/entrenar_efficientps_original.sh
```

En Mask2Former puedes exigir una reducción mínima de pérdida de 0.01 con
`--min-delta 0.01`. Consulta `--help` para ajustar resolución, checkpoint,
dataset o particiones. Si CUDA informa falta de memoria, baja `--image-size`
(por ejemplo a 192 en Mask2Former). En EfficientPS oficial cambia `img_scale`
en su config; no
cambies las particiones entre ejecuciones. Para conservar una ejecución previa,
indica otro directorio, por ejemplo:
`WORK_DIR="$PWD/training/runs/efficientps_original_run2"`.

Cada modelo guarda en `training/runs/<modelo>/`:

- `loss_history.csv` y `loss_curve.png`: pérdida media train/validation por época.
- `last_training.pt`: estado de la última época para reanudar con `--resume`.
- `best_model/` (Mask2Former): pesos con menor pérdida de validación.
- EfficientPS oficial: `loss_history.csv`, `loss_curve.png`, `training.log`,
  los dos checkpoints más recientes y `best_validation_loss.pth`.
- EfficientPS original: `loss_history.csv`, `loss_curve.png`, `training.log`,
  los checkpoints recientes, `best_validation_loss.pth` y `test_metrics.json`.
- `test_metrics.json`: IoU y precisión macro por píxel, F1 macro, mAP de máscaras
  para categorías thing a IoU 0.50:0.95 y PQ.
- `test_predictions/`: predicciones panópticas COCO del split test.

EfficientPS oficial puede continuar una ejecución interrumpida siempre que se
conserven juntos el checkpoint y `loss_history.csv`. Desde la raíz del
repositorio, por ejemplo:

```bash
cd "$HOME/cartografia_panoptica"
WORK_DIR="$PWD/training/runs/efficientps_original_reintento" \
RESUME_FROM="$PWD/training/runs/efficientps_original_reintento/latest.pth" \
  env -u PYTHONPATH bash tools/entrenar_efficientps_original.sh
```

La reanudación restaura el optimizador, la época y el historial de pérdida que
determina el early stopping. Para empezar desde cero, usa un `WORK_DIR` nuevo y
no definas `RESUME_FROM`. EfficientPS conserva los dos checkpoints más recientes
y copia aparte el de menor pérdida de validation.

Para inspeccionar las predicciones junto a la imagen original y el ground truth,
genera overlays comparativos de las 15 imágenes de test:

```bash
cd "$HOME/cartografia_panoptica"
env -u PYTHONPATH .venv-mask2former/bin/python \
  training/visualizar_predicciones.py --model both
```

La hoja completa se guarda en
`training/runs/visualizations_stuff/both/contact_sheet.png`, y cada ejemplo en
`image_<id>_comparison.png` dentro de esa misma carpeta. El color representa la
clase y los bordes oscuros delimitan segmentos/instancias. Para generar solo un
modelo, usa `--model mask2former` o `--model efficientps_original`; para una
previsualización pequeña agrega `--max-images 3`.
`--model both` compara Mask2Former con EfficientPS original.

Cuando ambos modelos hayan terminado, genera el reporte comparativo:

```bash
cd "$HOME/cartografia_panoptica"
env -u PYTHONPATH .venv-mask2former/bin/python \
  training/compare_results.py
```

Por defecto toma las ejecuciones recientes `mask2former_stuff_reentrenado` y
`efficientps_original_desde_cero`. El reporte se guarda en
`training/runs/comparison_final/` e incluye `comparison.md`, `comparison.csv`,
gráficas de métricas y pérdidas, y comparaciones visuales de los 15 ejemplos de
test con su ground truth. Para limitar la hoja visual, agrega `--max-images 3`.
Se pueden indicar otras ejecuciones con `--mask2former-run`, `--efficientps-run`
y `--output-dir`. IoU, F1 y precisión son métricas macro por píxel; mAP se
calcula solo sobre las categorías `thing` (manzana construida y zona verde en
esta ontología). El backbone B0 y la resolución reducida de EfficientPS original
deben describirse en el informe; no son la configuración publicada del paper.

## Aplicación web Streamlit

La app permite cargar una imagen, generar predicciones de Mask2Former y
EfficientPS original y consultar el reporte comparativo con gráficas y
predicciones frente al ground truth. Ejecuta las inferencias en secuencia para
mantener el uso de VRAM bajo; EfficientPS original se ejecuta en su entorno
Python/CUDA aislado. La pestaña de prueba permite recortar opcionalmente las
imágenes rectangulares a un cuadrado, con controles de posición para seleccionar
la región conservada. El escalado posterior se hace a 256×256 para Mask2Former
y 384×384 para EfficientPS original. El recorte coincide con la proporción de
las imágenes de entrenamiento, pero no garantiza mejores predicciones y descarta
la parte exterior; se puede desactivar.

Desde la raíz del repositorio:

```bash
cd "$HOME/cartografia_panoptica"
.venv-mask2former/bin/python -m pip install -r requirements-streamlit.txt
.venv-mask2former/bin/python -m streamlit run streamlit_app.py
```

La aplicación abre localmente en `http://localhost:8501`. Mask2Former carga
desde `training/runs/mask2former_stuff_reentrenado` y EfficientPS desde
`training/runs/efficientps_original_desde_cero`. Después de entrenar, reinicia
Streamlit para cargar los nuevos pesos. Cada modelo puede probarse por separado
o ambos sobre la misma imagen. La pestaña **Reporte comparativo** muestra el
contenido de `training/runs/comparison_final/comparison.md` junto con sus
gráficas y comparaciones visuales.

## Datos, modelos y archivos publicados

El repositorio incluye `data_panoptic_stuff` (anotaciones procesadas y
preparación de etiquetas para EfficientPS), el notebook de descarga y
`training/runs/comparison_final` (reporte, gráficas y comparaciones de test).
No incluye el dataset original, entornos virtuales ni artefactos temporales de
entrenamiento. Los dos checkpoints usados por la aplicación se distribuyen en
la GitHub Release `modelos-v1.0.0` como un archivo comprimido; ejecuta
`bash tools/descargar_modelos.sh` para descargarlos y ubicarlos en las rutas que
esperan Streamlit.

El dataset original debe descargarse con el notebook
`Visualización Dataset Cartografía Automatizada.ipynb` y extraerse como
`../reto-1-cartografa-automatizada` con respecto a la raíz del repositorio. La
aplicación de Streamlit solo requiere los pesos descargados y las anotaciones
procesadas incluidas en Git; el dataset original sí es necesario para volver a
entrenar.

`.gitignore` excluye los entornos, el dataset original y los resultados de
entrenamiento; conserva únicamente el reporte comparativo. Los pesos grandes
se mantienen fuera del historial Git para que clonar el código no descargue
cientos de megabytes automáticamente. Revisa licencias y condiciones de uso del
dataset, pesos preentrenados y dependencias antes de redistribuirlos. El código
upstream de EfficientPS incluido en `third_party/EfficientPS` conserva su
licencia GPL-3.0; este proyecto no declara una licencia propia.

El entrenamiento de EfficientPS requiere CUDA y su entorno legado documentado
en `training/efficientps_original/environment.yml`; Mask2Former usa
`training/requirements-mask2former.txt` y Streamlit se instala con
`requirements-streamlit.txt`.
