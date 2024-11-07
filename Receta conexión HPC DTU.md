# Receta conexión HPC DTU

- `ssh s243345@login.hpc.dtu.dk`
- `a100sh` (GPU) *or* `linuxsh` (CPU)
- Crear entorno virtual si es la primera vez (sección "Creating a project" en el tutorial)
  - Si ya lo tenemos, activarlo usando `source myenv/bin/activate`
- Transferir imágenes a nuestro directorio: `cp -r /zhome/70/5/14854/nobackup/deeplearningf24/forcebiology/data ~/02456/project` (tarda un rato)
- Abrir **FileZilla** y conectarse usando _Host = transfer.gbar.dtu.dk, Username = student id, Port = 22_
- Subir a FileZilla `run_display.sh` y `display.py`, guardándolos en el mismo directorio en el que se encuentra data. Cambiar los paths en `display.sh` para que cuadren con su directorio
- Lanzar job usando `bsub < run_display.sh`
- Si todo va bien, debería guardar la imagen en el propio directorio

### Ejecutar Jupyter Notebooks

- En terminal 1: `jp` (`jupyter notebook --no-browser --port=8888`) o `alias jp='jupyter notebook --no-browser --port=8888'`
- En terminal 2: `ssh -N -L 8889:localhost:8889 s243345@login.hpc.dtu.dk`
- Al cerrar (Ctrl + C), ejecutar `jupyter server list` para comprobar que hemos cerrado la libreta