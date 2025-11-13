# alc_2024c2

## Profiling

Hay una prueba de profiling para svd. 

En un enviroment, instalar snakeviz

```
pip install snakeviz
```

Correr el programa una vez.

```
python -m cProfile -o output.prof prof.py 
```

Lanzar la interfaz web de snakeviz:

```
snakeviz output.prof
```