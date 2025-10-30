#!/bin/bash
# Compile learn_bn.cpp (which includes em_algorithm.cpp via #include)
g++ -std=c++17 -O3 learn_bn.cpp -o learn_bn 