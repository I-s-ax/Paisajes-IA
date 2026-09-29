# Paisajes-IA

Clasificador y agrupador de paisajes con **DINOv2 + CLIP**, pensado para ejecutar la IA en Lightning AI y mantener las fotos bajo control de Termux/Android.

## Flujo recomendado: Termux + Lightning AI

Las fotos **no tienen que quedarse almacenadas en Lightning**.

El flujo automático es:

```text
Android / Termux
carpeta de fotos
      │
      │ lote temporal
      ▼
Lightning AI
DINOv2 + CLIP
      │
      ├─ devuelve etiqueta + embedding
      └─ elimina las copias temporales
      │
      ▼
Termux
SQLite local + grupos persistentes
      │
      └─ copia o mueve el archivo original localmente
```

El worker de Lightning permanece encendido con DINOv2 y CLIP cargados en memoria, así no se vuelven a cargar los modelos para cada foto.

### 1. Preparar Lightning AI

En el Studio:

```bash
git clone https://github.com/I-s-ax/Paisajes-IA.git
cd Paisajes-IA
bash setup.sh
python src/download_models.py
```

Cuando vayas a procesar fotos, activa una GPU y comprueba:

```bash
python src/check_env.py
```

Luego inicia el worker:

```bash
bash cloud/start_worker.sh
```

Ver el log:

```bash
tail -f .runtime/worker.log
```

El worker usa por defecto:

```text
/tmp/paisajes-ia/incoming/
/tmp/paisajes-ia/results/
```

Cuando termina un lote, elimina inmediatamente la carpeta temporal que contenía las fotos. El JSON de resultado queda hasta que Termux confirma que lo recibió.

### 2. Obtener los datos SSH de Lightning

En el Studio usa **Connect via SSH** y toma los valores de host, usuario, puerto y, si aplica, la clave privada.

Primero comprueba desde Termux que el comando SSH proporcionado por Lightning funciona.

### 3. Preparar Termux

En Android:

```bash
git clone https://github.com/I-s-ax/Paisajes-IA.git
cd Paisajes-IA
bash termux/setup.sh
```

El cliente Termux solo usa Python estándar + OpenSSH.

### 4. Procesar una carpeta local

Ejemplo:

```bash
python termux/client.py \
  --input /storage/emulated/0/DCIM/Camera \
  --output /storage/emulated/0/PaisajesClasificados \
  --host TU_HOST \
  --user TU_USUARIO \
  --port TU_PUERTO \
  --batch-size 12 \
  --move
```

Si Lightning usa una clave SSH:

```bash
python termux/client.py \
  --input /storage/emulated/0/DCIM/Camera \
  --output /storage/emulated/0/PaisajesClasificados \
  --host TU_HOST \
  --user TU_USUARIO \
  --port TU_PUERTO \
  --identity ~/.ssh/TU_CLAVE \
  --move
```

Sin `--move`, el cliente **copia** las fotos y conserva los originales.

### Qué guarda Termux

Dentro de la salida:

```text
PaisajesClasificados/
├── GRUPO_0001/
├── GRUPO_0002/
├── ...
├── resultados.csv
└── .paisajes-ai/
    └── estado.sqlite3
```

La base local recuerda:

- hashes SHA-256;
- imágenes ya procesadas;
- grupos persistentes;
- centroides DINOv2;
- etiquetas CLIP;
- destinos locales.

Por eso, si se corta Internet o cierras Termux, puedes ejecutar el mismo comando otra vez y continuar.

### Agrupar o clasificar por etiqueta

Por defecto organiza por grupo visual:

```bash
--organize-by group
```

Solo por etiqueta CLIP:

```bash
--organize-by label
```

Etiqueta y grupo:

```bash
--organize-by label-group
```

Ejemplo:

```text
PaisajesClasificados/
└── playa/
    └── GRUPO_0003/
        ├── IMG_001.jpg
        └── IMG_002.jpg
```

### Ajustar similitud de los grupos

El cliente usa por defecto:

```bash
--similarity 0.86
```

Un valor más alto exige imágenes más parecidas. Un valor más bajo agrupa de forma más amplia.

Ejemplo:

```bash
--similarity 0.90
```

## Modo directo en Lightning

También existe `src/analyze.py` para el caso en que quieras subir una carpeta completa al Studio y procesarla allí.

```bash
python src/analyze.py \
  --input /workspace/fotos \
  --output /workspace/resultados
```

Por defecto organiza por similitud visual:

```text
resultados/
├── grupos/
│   ├── GRUPO_001/
│   ├── GRUPO_002/
│   └── ...
├── .paisajes-ai/
│   └── paisajes.sqlite3
└── resultados.csv
```

El CSV también incluye la etiqueta CLIP de cada imagen. Si además quieres carpetas por etiqueta:

```bash
python src/analyze.py \
  --input /workspace/fotos \
  --output /workspace/resultados \
  --organize-by both
```

Por seguridad, este modo también **copia** las imágenes de forma predeterminada. Usa `--move` solo si quieres mover los originales.

## Etiquetas personalizadas

Puedes editar `labels.txt`. Cada línea tiene este formato:

```text
nombre_en_espanol|prompt en inglés para CLIP
```

Ejemplo:

```text
playa|a photo of a beach
montana|a photo of mountains
bosque|a photo of a forest
```

## Privacidad

El repositorio no necesita contener fotografías.

En el modo Termux + Lightning, cada lote se transfiere temporalmente a la máquina de Lightning. El worker elimina las copias del lote después de obtener DINOv2 + CLIP. El original permanece en Android y solo Termux lo mueve o copia.

Eliminar el archivo temporal del Studio no constituye una garantía de borrado forense de toda la infraestructura del proveedor.

## Modelos

- `facebook/dinov2-base`
- `openai/clip-vit-base-patch32`

La primera ejecución descarga los modelos desde Hugging Face. Luego quedan en la caché local del Studio mientras el almacenamiento persista.
