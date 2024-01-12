#! /bin/bash

# This script is used to run the RRL algorithm on the given dataset.
python experiment.py -d bank -bs 32 -s 10@64 -e401 -lrde 200 -lr 0.002 -ki 0 -i 0 -wd 0.0001 --print_rule --save_best

python experiment.py -d bank -bs 32 -s 10@64@16 -e401 -lrde 200 -lr 0.002 -ki 0 -i 0 -wd 0.00001 --print_rule --save_best

python experiment.py -d raw_bank -bs 32 -s 10@64@16 -e401 -lrde 200 -lr 0.002 -ki 0 -i 0 -wd 0.00001 --print_rule --save_best

python experiment.py -d bank_processed -bs 32 -s 10@64@16 -e401 -lrde 200 -lr 0.002 -ki 0 -i 0 -wd 0.00001 --print_rule --save_best

python experiment.py -d bank_processed -bs 128 -s 10@64@16 -e401 -lrde 40 -lr 0.02 -ki 0 -i 0 -wd 0.000001 --print_rule --save_best

python experiment.py -d bank_processed -bs 256 -s 10@64@16 -e401 -lrde 20 -lr 0.01 -ki 0 -i 0 -wd 0.000001 --print_rule --save_best

python experiment.py -d raw_bank -bs 256 -s 10@64@16 -e401 -lrde 20 -lr 0.01 -ki 0 -i 0 -wd 0.000001 --print_rule --save_best