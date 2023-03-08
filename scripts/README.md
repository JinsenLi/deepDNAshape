# Scripts of training the model and utilities

## Convert DNA shape features to DNA structures (.pdb)

Firstly, predict 12 DNA shape features for one sequence and put them into one single sub-directory

Secondly, use `preditions2par.py` to convert the predictions to PAR file.

Thirdly, use `rebuild` program in `X3DNA` (https://x3dna.org/) package to rebuild DNA structure from the PAR file. 

A bash script is provided to rebuild one sequence to PDB. Checkout `rebuild` from X3DNA for details. 
```
./pre-rebuild.sh AGTGATAG test.par
rebuild -atomic test.par test.pdb
```

## Train a model from scratch

comming soon!