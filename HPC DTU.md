# Receta conexión HPC DTU

- `ssh studentid@login.hpc.dtu.dk`
- `a100sh` (GPU) *or* `linuxsh` (CPU)
- Crear entorno virtual si es la primera vez (sección "Creating a project" en el tutorial)
  - Si ya lo tenemos, activarlo usando `source myenv/bin/activate`
- Abrir **FileZilla** y conectarse usando _Host = transfer.gbar.dtu.dk, Username = studentid, Port = 22_
- Subir a FileZilla `run_display.sh` y `display.py`
- Lanzar job usando `bsub < run_display.sh`
- Si todo va bien, debería guardar la imagen en el propio directorio

### Ejecutar Jupyter Notebooks

- En terminal 1: `jupyter notebook --no-browser --port=8888`
- En terminal 2: `ssh -N -L 8889:localhost:8889 s243345@login.hpc.dtu.dk`
- **Importante:** Siempre cerrar las libretas usando Ctrl + C en la Terminal 1. Ejecutar `jupyter server list` para comprobar que hemos cerrado la libreta correctamente
