#! /bin/bash

clang -O3  -shared -dynamiclib -fvisibility=default -o aux.dylib ffun/aux.c
