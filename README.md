# Paisajes-IA

Clasificador y agrupador de paisajes con **DINOv2 + CLIP**, pensado para ejecutarse en Lightning AI con GPU.

## Qué hace

- DINOv2 crea embeddings visuales para medir similitud entre imágenes.
- CLIP asigna etiquetas semánticas como playa, montaña, bosque, ciudad, nieve, atardecer, etc.
- Agrupa automáticamente imágenes visualmente parecidas.
- Guarda una caché SQLite por SHA-256 para no recalcular imágenes ya procesadas.
- Puede copiar (por defecto) o mover las imágenes a carpetas de resultados.
- Reduce imágenes grandes antes de procesarlas para controlar el consumo de RAM.
- Funciona por lotes en GPU y usa FP16 cuando CUDA está disponible.

## Lightning AI

En la terminal del Studio:

```bash
git clone https://github.com/I-s-ax/Paisajes-IA.git
cd Paisajes-IA
bash setup.sh
```

Comprueba la GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Uso rápido

Crea una carpeta local para las fotos, por ejemplo:

```text
/workspace/fotos
```

y ejecuta:

```bash
python src/analyze.py \
  --input /workspace/fotos \
  --output /workspace/resultados
```

El resultado será parecido a:

```text
resultados/
├── grupos/
│   ├── GRUPO_001/
│   ├── GRUPO_002/
│   └── ...
├── etiquetas/
│   ├── playa/
│   ├── montana/
│   └── ...
├── resultados.csv
└── paisajes.sqlite3
```

Por seguridad, el programa **copia** las imágenes de forma predeterminada. Usa `--move` solo si quieres mover los originales.

## Número de grupos

Por defecto `--clusters auto` estima automáticamente una cantidad razonable de grupos.

También puedes fijarlo:

```bash
python src/analyze.py --input /workspace/fotos --output /workspace/resultados --clusters 12
```

## Solo analizar, sin copiar archivos

```bash
python src/analyze.py --input /workspace/fotos --output /workspace/resultados --dry-run
```

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

Las imágenes se procesan en la máquina de Lightning AI. El repositorio no necesita contener fotografías. Las carpetas `photos/`, `images/`, `data/`, `outputs/` y archivos de caché están excluidos por `.gitignore`.

## Modelos

- `facebook/dinov2-base`
- `openai/clip-vit-base-patch32`

La primera ejecución descarga los modelos desde Hugging Face. Luego quedan en la caché local del Studio mientras el almacenamiento persista.
