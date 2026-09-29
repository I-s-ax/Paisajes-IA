#!/usr/bin/env python3
import torch
import transformers

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    print(f"GPU {index}:", torch.cuda.get_device_name(index))

if torch.cuda.is_available():
    print("Estado: listo para ejecutar Paisajes-IA con GPU.")
else:
    print("Estado: funciona en CPU, pero conviene activar GPU para analizar fotos.")
