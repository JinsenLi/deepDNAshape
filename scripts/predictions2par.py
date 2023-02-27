import argparse
parser = argparse.ArgumentParser(description='Merge all predictions file into one par file for each sequence in the prediciton files.')
parser.add_argument("--fileprefix",dest = "fileprefix", required = True, help = "Filename that starts with; Will look for [FILEPREFIX]_Roll.txt, _ProT.txt, etc.")
parser.add_argument("--output", dest = "output", default = "", help = "If this value is set (and seq is set), will output the required seq to this file; If seq is not set, will output all to this folder. ")
parser.add_argument("--seq", dest = "seq", default = "", help = "If set, find the required seq in all fileprefix files.")
parser.add_argument("--alllayers", dest = "alllayers", action = "store_true", help = "If set, will treat the output to contain all layers output. ")
parser.add_argument("--layernum", dest = "layernum", default = 0, type = int, help = "If [alllayers] is true, will output this [layernum] layer.")
parser.add_argument("--forplot", dest = "forplot", action = "store_true")
parser.add_argument("--reverse", dest = "reverse", action = "store_true")
args = parser.parse_args()

from collections import defaultdict
import sys, os

x_axis_features = set(['Shift', 'Tilt', 'Shear', 'Buckle'])
intrabase_features = ["Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening"]
if args.forplot:
    intrabase_features += ["MGW", "EP"]
if_reverse = args.reverse
interbase_features = ["Shift", "Slide", "Rise", "Tilt", "Roll", "HelT"]
data = defaultdict(dict)
for feature in intrabase_features + interbase_features:
    with open(args.fileprefix + "_" + feature + ".txt") as fin:
        for line in fin:
            items = line.strip().split()
            seq = items[0]
            if args.alllayers:
                layer = int(items[1])
                features = items[2:]
            else:
                layer = 0
                features = items[1:]
            if layer not in data[seq]:
                data[seq][layer] = {}
            data[seq][layer][feature] = features

def output(data, seq, fout, layernum):
    pairs = {"A":"T", "T": "A", "C": "G", "G": "C", "N": "N", "M": "g", "g": "M"}
    fout.write(str(len(seq))+"\n")
    fout.write("0\n")
    fout.write("#        Shear    Stretch   Stagger   Buckle   Prop-Tw   Opening     Shift     Slide     Rise      Tilt      Roll      Twist\n")
    for i in range(len(seq)):
        fout.write(seq[i] +"-" + pairs[seq[i]] + "\t")
        for feature in intrabase_features:
            fout.write(data[seq][layernum][feature][i] + "\t")
        for feature in interbase_features:
            if i == 0:
                fout.write("0.0\t")
            else:
                fout.write(data[seq][layernum][feature][i - 1] + "\t")
        fout.write("\n")

def plotoutput(data, seq, fout, layernum):
    #pairs = {"A":"T", "T": "A", "C": "G", "G": "C"}
    fout.write("seq\t" + "\t".join(intrabase_features) + "\t" + "\t".join(interbase_features) + "\n")
    for i in range(len(seq)):
        fout.write(seq[i] + "\t")
        for feature in intrabase_features:
            fout.write(data[seq][layernum][feature][i] + "\t")
        for feature in interbase_features:
            if i == 0:
                fout.write("0.0\t")
            else:
                fout.write(data[seq][layernum][feature][i - 1] + "\t")
        fout.write("\n")
#print(data[args.seq].keys())

def reverse(data, seq, layernum):
    pairs = {"A":"T", "T": "A", "C": "G", "G": "C", "N": "N", "M": "g", "g": "M"}
    opposeq = seq[::-1]
    opposeq = "".join(list(map(lambda x: pairs[x], opposeq)))
    if opposeq in data:
        return None
    data[opposeq] = {}
    data[opposeq][layernum] = {}
    for feature in intrabase_features + interbase_features:
        data[opposeq][layernum][feature] = data[seq][layernum][feature][::-1]
        if feature in x_axis_features:
            data[opposeq][layernum][feature] = list(map(lambda x: str(-float(x)), data[opposeq][layernum][feature]))
    return opposeq

if args.forplot:
    func = plotoutput
else:
    func = output
if args.seq != "":
    if args.output != "":
        fout = open(args.output, "w")
    else:
        fout = sys.stdout
    func(data, args.seq, fout, args.layernum)
    if args.output != "":
        fout.close()
else:
    for seq in list(data.keys()):
        with open(os.path.join(args.output, seq + ".par"), "w") as fout:
            func(data, seq, fout, args.layernum)
        if if_reverse:
            opposeq = reverse(data, seq, args.layernum)
            if opposeq:
                with open(os.path.join(args.output, opposeq + ".par"), "w") as fout:
                    func(data, opposeq, fout, args.layernum)

