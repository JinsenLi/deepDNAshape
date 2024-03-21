# Welcome to Deep DNAshape

The package includes an executable `deepDNAshape` to predict DNA shape features for any sequences. 

It also includes all the components of Deep DNAshape. You may incoorporate Deep DNAshape into your pipeline or modify it to fit your needs.

Please also check out our webserver for predicting of DNA shape features in real time, <https://deepdnashape.usc.edu/>.

## Installation

Prerequsite: `tensorflow >= 2.6.0` `numpy<1.24`
For tensorflow version >= 2.16, please use keras 2 (see <https://blog.tensorflow.org/2024/03/whats-new-in-tensorflow-216.html>)

### Download and install through pip
```
git clone https://github.com/JinsenLi/deepDNAshape
cd deepDNAshape
pip install .
```
Installation time should be minimal depending on the time to install the prerequsites. 

## Quickstart

Pre-trained models are provided with the package. You don't need to train anything to predict DNA shape features! If you want to use the model to train other data, please go to "scripts" folder.

Run time of the program depends on the amount of inputs. For a single sequence, run time should be a couple seconds. If you are processing large data, please consider using `--file` option which will expedite the process. 

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


### Predict any DNA shape from a line-separated sequence txt file
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
5.456438 4.6564693 4.0487256 3.7174146 3.7821176 3.9350023 3.829193 4.4738736 4.8066416 5.043952 5.3840685 5.3597145 4.9772162 4.829335
4.8822823 5.098533 5.235756 5.8786955 6.113864 6.084464 5.7162333 5.055209 4.7080736 4.8015795 4.8796396 4.9851036 4.444648 4.0474467 4.7741375 5.873541 6.1201353 5.47472 4.915975 4.4750524 4.7644296 5.3036046 5.545209 5.43421
5.456438 4.6639423 4.0483274 3.631318 3.635215
```

To choose output file:

Use `deepDNAshape --file [FILE] --output [OUTPUTFILE]` to specify an output file to store the predictions instead of stdout.

### Predict any DNA shape from a fasta sequence file
`deepDNAshape --file [FASTA_FILE] --feature MGW` - This command will predict minor groove width for all the sequences in [FASTA_FILE].


`[FASTA_FILE]` format: starts with `>XXX`
```
>test1
ACGTAAAAGGGGATAACCG
>test2
CCGTAGGG
>test3
GGTGAGGGGGGGGGGGGGG
```

Results will be in the same format as above if use stdout or output to regular text file:
```
5.335149 4.919947 5.1440744 5.9646835 5.8556986 4.9728765 4.2535486 4.315494 4.355875 4.689518 4.7436676 4.923707 5.141595 5.7708316 5.6300097 4.841404 4.490379 5.00844 5.259532
4.879819 5.13968 5.2515874 5.8307476 5.9487104 5.065263 4.6507463 4.719969
4.977294 4.8510094 5.546277 5.7830486 5.477006 5.0075583 4.778365 4.883775 4.9586406 4.956913 4.956913 4.956913 4.956913 4.956913 4.956913 4.950612 4.9112043 4.8351297 4.8283052
```


Results will be in FASTA format if `--output [OUTPUTFILE]` is used and `[OUTPUTFILE]` endswith `.fa` or `.fasta`:
```
>test1
5.335149,4.919947,5.1440744,5.9646835,5.8556986,4.9728765,4.2535486,4.315494,4.355875,4.689518,4.7436676,4.923707,5.141595,5.7708316,5.6300097,4.841404,4.490379,5.00844,5.259532
>test2
4.879819,5.13968,5.2515874,5.8307476,5.9487104,5.065263,4.6507463,4.719969
>test3
4.977294,4.8510094,5.546277,5.7830486,5.477006,5.0075583,4.778365,4.883775,4.9586406,4.956913,4.956913,4.956913,4.956913,4.956913,4.956913,4.950612,4.9112043,4.8351297,4.8283052
```


## Windows usage

For users trying to use the package in Windows environment. Please download `deepDNAshape` in the `bin/` directory to local after installing the package and change the run command `deepDNAshape` to `python deepDNAshape ...`
