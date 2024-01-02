export GPU=2

# Boston Housing
python3 experiment.py -d BostonHousing -r -bs 32 -s 1@16 -e401 -lrde 200 -lr 0.002 -ki 0 -i $GPU -wd 0.0001 --print_rule