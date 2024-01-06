#! /bin/bash

# This script is used to run the RRL algorithm on the given dataset.
python experiment.py -d bank -bs 32 -s 10@64 -e401 -lrde 200 -lr 0.002 -ki 0 -i 0 -wd 0.001 --print_rule --weighted
