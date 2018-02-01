#!/bin/bash

TESTS_TME1="\
  pobj.motx.tme1.test.CaseTest \
  pobj.motx.tme1.test.GrilleTest \
  pobj.motx.tme1.test.GrillePlacesTest \
"

#TESTS_TME2="\
#  pobj.motx.tme2.test.DictionnaireTest \
#  pobj.motx.tme2.test.GrillePotentielTest \
#  pobj.motx.tme2.test.DictionnaireTest2 \
#  pobj.motx.tme2.test.GrillePotentielTest2 \
#  pobj.motx.tme2.test.GrillePotentielTest3 \
#  pobj.motx.tme2.test.GrillePotentielTest4 \
#  pobj.motx.tme2.test.GrillePotentielTest5 \
#  pobj.motx.tme2.test.GrillePotentielTest6 \
#"

#TESTS_TME3="\
#  pobj.motx.tme3.test.GrilleSolverTest \
#"

TESTS="$TESTS_TME1 $TESTS_TME2 $TESTS_TME3"


#########

CP=scripts-ci/pobj-ci.jar:/usr/share/java/junit4.jar:/usr/share/java/hamcrest-core-1.3.jar

SRC=./src

echo "Date :"
date
echo

echo "Répertoire courant :"
pwd
echo

echo "Liste des fichiers :"
ls -lR
echo

echo "Lancement des tests"
echo java -cp $CP:. pobj.ci.Runner -c -cp $CP -sp $SRC $TESTS
java -cp $CP:. pobj.ci.Runner -c -cp $CP -sp $SRC $TESTS
