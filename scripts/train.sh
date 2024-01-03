export GPU=2

# Boston Housing
python3 experiment.py -d BostonHousing -r -bs 32 -s 10@64 -e401 -lrde 200 -lr 0.002 -ki 0 -i $GPU -wd 0.0001 --print_rule

python3 experiment.py -d BostonHousing -r -bs 32 -s 10@64@16 -e401 -lrde 200 -lr 0.002 -ki 0 -i $GPU -wd 0.0001 --print_rule

python3 experiment.py -d WineQuality -r -bs 32 -s 10@64 -e401 -lrde 200 -lr 0.002 -ki 0 -i $GPU -wd 0.0001 --print_rule

python3 experiment.py -d RedWineQuality -r -bs 32 -s 10@64 -e401 -lrde 200 -lr 0.002 -ki 0 -i $GPU -wd 0.0001 --print_rule

python3 experiment.py -d RedWineQuality -r -bs 32 -s 10@64@16 -e401 -lrde 200 -lr 0.002 -ki 0 -i $GPU -wd 0.0001 --print_rule

python3 experiment.py -d RedWineQuality -r -bs 32 -s 10@64@16 -e401 -lrde 40 -lr 0.002 -ki 0 -i $GPU -wd 0.0001 --print_rule --save_best

python3 experiment.py -d OnlineNewsPopularity -r -bs 32 -s 10@64 -e401 -lrde 200 -lr 0.02 -ki 0 -i $GPU -wd 0.0001 --print_rule