#!/bin/bash

SEQ=$1
OUTPUT=$2


deepDNAshape --feature Shear --seq $SEQ --showseq > tmp_Shear.txt
deepDNAshape --feature Stretch --seq $SEQ --showseq > tmp_Stretch.txt
deepDNAshape --feature Stagger --seq $SEQ --showseq > tmp_Stagger.txt
deepDNAshape --feature Buckle --seq $SEQ --showseq > tmp_Buckle.txt
deepDNAshape --feature ProT --seq $SEQ --showseq > tmp_ProT.txt
deepDNAshape --feature Opening --seq $SEQ --showseq > tmp_Opening.txt
deepDNAshape --feature Shift --seq $SEQ --showseq > tmp_Shift.txt
deepDNAshape --feature Slide --seq $SEQ --showseq > tmp_Slide.txt
deepDNAshape --feature Rise --seq $SEQ --showseq > tmp_Rise.txt
deepDNAshape --feature Tilt --seq $SEQ --showseq > tmp_Tilt.txt
deepDNAshape --feature Roll --seq $SEQ --showseq > tmp_Roll.txt
deepDNAshape --feature HelT --seq $SEQ --showseq > tmp_HelT.txt

python predictions2par.py --fileprefix tmp --seq $SEQ > $OUTPUT