#!/bin/bash

SEQ=$1
OUTPUT=$2


deepDNAshape --feature Shear --seq $SEQ > tmp_Shear.txt
deepDNAshape --feature Stretch --seq $SEQ > tmp_Stretch.txt
deepDNAshape --feature Stagger --seq $SEQ > tmp_Stagger.txt
deepDNAshape --feature Buckle --seq $SEQ > tmp_Buckle.txt
deepDNAshape --feature ProT --seq $SEQ > tmp_ProT.txt
deepDNAshape --feature Opening --seq $SEQ > tmp_Opening.txt
deepDNAshape --feature Shift --seq $SEQ > tmp_Shift.txt
deepDNAshape --feature Slide --seq $SEQ > tmp_Slide.txt
deepDNAshape --feature Rise --seq $SEQ > tmp_Rise.txt
deepDNAshape --feature Tilt --seq $SEQ > tmp_Tilt.txt
deepDNAshape --feature Roll --seq $SEQ > tmp_Roll.txt
deepDNAshape --feature HelT --seq $SEQ > tmp_HelT.txt

python predictions2par.py --fileprefix tmp --seq $SEQ > $OUTPUT