#!/bin/bash

n_cores=50

for i in $(seq 1 $n_cores); do
    python scenarios.py &
done

wait
