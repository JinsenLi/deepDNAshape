# Welcome to Deep DNAshape

The package includes an executable `deepDNAshape` to predict DNA shape features for any sequences. 

It also includes all the components of Deep DNAshape. You may incoorporate Deep DNAshape into your pipeline or modify it to fit your needs.

## Installation

Prerequsite: `tensorflow >= 2.0` `numpy`
### Download and install through pip
```
wget https://github.com/JinsenLi/deepDNAshape/blob/main/release/deepDNAshape.zip
unzip deepDNAshape.zip
cd deepdnashape
pip install .
```

## Quickstart

Pre-trained models are provided with the package. You don't need to train anything to predict DNA shape features!

* `deepDNAshape -h` - Print help message and exit.
* `deepDNAshape --seq [SEQ] --feature [FEATURE]` - Specify the DNA shape feature and the sequence to be predicted. DNA shape features include MGW, Shear, Stretch, Stagger, Buckle, ProT, Opening, Shift, Slide, Rise, Tilt, Roll, HelT. Add "-FL" to the feature name to predict DNA shape fluctuations, e.g. MGW-FL.

### Predict any DNA shape for a single sequence

`deepDNAshape --seq AAGGTT --feature MGW` - This command will predict minor groove width for the sequence AAGGTT on all 6 positions.

To select layers:

Use `--layer [l]` to select the layer number. `[l]` must be `0 - 7`, integers. 

Example 1:

* `deepDNAshape --seq AGAGATACGATACGA --feature ProT --layer 2`

* This example will predict propeller twist (ProT) of sequence `AGAGATACGATACGA` by considering only 2bp of flanking regions. 

Example 2:

* `deepDNAshape --seq AGAGATACGATACGA --feature ProT --layer 7`

* This example will predict propeller twist (ProT) of sequence `AGAGATACGATACGA` by considering 7bp of flanking regions. 


### Predict any DNA shape from a sequence file
`deepDNAshape --file [FILE] --feature MGW` - This command will predict minor groove width for all the sequences in [FILE].


Use `--file [FILE]` to replace `--seq [SEQ]`. 

`[FILE]` format: each line is one sequence. 
```
AAAAAACCCCCGGG
CCGTGCAGGGATATTTAGACCCAT
AAAAA
```

Results will be:
```
AAAAAACCCCCGGG 5.456438 4.6564693 4.0487256 3.7174146 3.7821176 3.9350023 3.829193 4.4738736 4.8066416 5.043952 5.3840685 5.3597145 4.9772162 4.829335
CCGTGCAGGGATATTTAGACCCAT 4.8822823 5.098533 5.235756 5.8786955 6.113864 6.084464 5.7162333 5.055209 4.7080736 4.8015795 4.8796396 4.9851036 4.444648 4.0474467 4.7741375 5.873541 6.1201353 5.47472 4.915975 4.4750524 4.7644296 5.3036046 5.545209 5.43421
AAAAA 5.456438 4.6639423 4.0483274 3.631318 3.635215
```

To choose output file:

Use `deepDNAshape --file [FILE] --output [OUTPUTFILE]` to specify an output file to store the predictions instead of stdout.

