
#!usr/bin/bash

# A executer depuis le dossier DPLP sous peine d'echec !
# types : argumentative  informative  narrative  

#rm ../data/*.txt.*

bash /tmp/corenlp/torun.sh

python convert.py ~/Documents/s2/tal/discourseAnalysis/data/narrative;
python convert.py ~/Documents/s2/tal/discourseAnalysis/data/argumentative;
python convert.py ~/Documents/s2/tal/discourseAnalysis/data/informative;

python segmenter.py ~/Documents/s2/tal/discourseAnalysis/data/narrative;
python segmenter.py ~/Documents/s2/tal/discourseAnalysis/data/argumentative;
python segmenter.py ~/Documents/s2/tal/discourseAnalysis/data/informative;

python analyse.py ~/Documents/s2/tal/discourseAnalysis/data/narrative;
python analyse.py ~/Documents/s2/tal/discourseAnalysis/data/argumentative;
python analyse.py ~/Documents/s2/tal/discourseAnalysis/data/informative;


