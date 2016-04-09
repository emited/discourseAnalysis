
#!usr/bin/bash

# path to use : ~/Documents/s2/tal/discourseAnalysis/data
~/Documents/s2/tal/discourseAnalysis/data

# A executer depuis le dossier DPLP sous peine d'echec !
rm ../data/*.txt.*


../coreNLP/corenlp_2.sh  ../data

python convert.py ../data

python segmenter.py ../data

python analyse.py ../data

#python rstparser.py ../data False



