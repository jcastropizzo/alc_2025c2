#! /bin/bash


# Si estan en Mac usen esto
clang -shared -dynamiclib -fvisibility=default -o aux.dylib ffun/aux.c

# Usen esto para Linux
gcc -c -fPIC ffun/aux.c -o aux.o
gcc -shared -o aux.so aux.o