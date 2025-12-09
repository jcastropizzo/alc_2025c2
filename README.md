# alc_2024c2

## Compilar versión en C de MatMul

Leer archivo `[build.aux](Recursos TP-20251107/template_alumnos/build-aux.sh)` dentro del directorio `template_alumnos` y compilar acorde a la plataforma correspondiente.

## Conclusiones

Las conclusiones se encuentran al final de la notebook `[eval.ipybin](Recursos TP-20251107/template_alumnos/eval.ipybin)` en el directorio `template_alumnos`.

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
