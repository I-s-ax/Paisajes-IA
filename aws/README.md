# AWS EC2

## Fase 1: preparar una instancia CPU

Usa una instancia **x86_64**, por ejemplo `t3.small`, con Ubuntu Server LTS x86_64.

Después de entrar a la instancia:

```bash
git clone https://github.com/I-s-ax/Paisajes-IA.git
cd Paisajes-IA
bash aws/setup_cpu.sh
```

Comprueba:

```bash
uname -m
python3 --version
```

`uname -m` debe mostrar:

```text
x86_64
```

No instales CUDA ni PyTorch GPU mientras la máquina siga siendo CPU.

## Fase 2: cuando AWS apruebe la cuota G/VT

1. Detener la instancia.
2. Cambiar el tipo a `g4dn.xlarge` si aparece como compatible.
3. Iniciar la instancia.
4. Instalar/verificar el controlador NVIDIA.
5. Instalar el entorno Python para DINOv2 + CLIP.
6. Ejecutar el servidor de Paisajes-IA.
7. Conectar Termux por HTTPS.

Si EC2 no permite cambiar directamente el tipo, se creará una nueva instancia GPU compatible usando la misma arquitectura x86_64 y se migrará el proyecto.
