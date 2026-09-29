#!/usr/bin/env python3
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)

DINO_MODEL = "facebook/dinov2-base"
CLIP_MODEL = "openai/clip-vit-base-patch32"

print("Preparando DINOv2...")
AutoImageProcessor.from_pretrained(DINO_MODEL)
AutoModel.from_pretrained(DINO_MODEL)
print("DINOv2 listo en caché.")

print("Preparando CLIP...")
CLIPProcessor.from_pretrained(CLIP_MODEL)
CLIPModel.from_pretrained(CLIP_MODEL)
print("CLIP listo en caché.")

print("Modelos descargados. Ya puedes cambiar el Studio a GPU cuando quieras procesar imágenes.")
