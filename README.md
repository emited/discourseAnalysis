# discourseAnalysis

##Text types
Textual types refer to the following four basic aspects of writing: descriptive, narrative, expository, and argumentative.

**Descriptive text type:**

**Narrative test type:**

**Expository text type:**

**Argumentative text type:**


https://en.wikipedia.org/wiki/Text_types

**corpus links**: 
       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Quelques idées:
  * classifieurs:
      - Naive Bayes
      - SVM
      - KNN
  * kernels:
      - eucl dist
      - treeKernel
      - mod treeKernel, ou les relations du discours sont pondérés selon leur position dans la représentation de l'arbre.


                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fonctions ecrites :**

  - Collection de relations existantes dans un texte

  (Sous forme de dictionnaire, forme sparse équivalente a un vecteur binaire)

  - Construction de Vecteur d'occurence des relations présentes 

  (Sous forme de dico sparse)

  - Construction de Vecteur d'occurence normalisé des relations présentes 

  (Sous forme de dico sparse normalisé par nombre total de relations comptées)

  - Calcul de distance euclidienne entre 2 textes (tree) pour chacune des 
  trois représentations

  RQ :
 
  - Le tout est disponible dans code/treekernel|.py|_test.py,
  - Lire commentaires dans code/treekernel_test.py pour tester sur 
  les données disponibles dans data/
  - Voir src/readme_2.txt pour tester le pipeline 
    =>traitement des données du .txt à l'écrire des trees dans .csv

  - Pour les prochaines versions : réfléchir aux kernels les plus pertinents à
  implémenter.
   =>Mettre en valeur certaines relations ?
   =>Penaliser fortement abscence de relations dans calcul de distance ?
   
  - Comment tenir compte du Pos tagging ?
