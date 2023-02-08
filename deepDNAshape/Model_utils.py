import numpy as np
import itertools

def getBasesMapping(if_phychem = False, include_5mc = False):
    if if_phychem:
        purine = [0]
        pyrimidine = [1]
        weakbond = [0]
        strongbond = [1]
        nongroup = [0,0,0]
        nh2 = [0,1,0]
        met = [1,0,0]
        ketone = [0,0,1]
        pairs = {"A": np.array([purine + pyrimidine + weakbond + nh2 + nongroup + ketone + met]).flatten(),
            "C": np.array([pyrimidine + purine + strongbond + nh2 + nongroup + ketone + nh2]).flatten(),
            "G": np.array([purine + pyrimidine + strongbond + ketone + nh2 + nh2 + nongroup]).flatten(),
            "T": np.array([pyrimidine + purine + weakbond + ketone + met + nh2 + nongroup]).flatten()}
        #h_acceptor = [0,0,0,1]
        #h_donor = [0,0,1,0]
        #met = [0,1,0,0]
        #nonpolar = [1,0,0,0]
        #pairs = {"A": np.array([h_acceptor, h_donor, h_acceptor, met, h_acceptor, nonpolar, h_acceptor]).flatten(),
        #     "C": np.array([nonpolar, h_donor, h_acceptor, h_acceptor, h_acceptor, h_donor, h_acceptor]).flatten(),
        #     "G": np.array([h_acceptor, h_acceptor, h_donor, nonpolar, h_acceptor, h_donor, h_acceptor]).flatten(),
        #     "T": np.array([met, h_acceptor, h_donor, h_acceptor, h_acceptor, nonpolar, h_acceptor]).flatten()}
        bppool = ["A", "C", "G", "T", "N"]
        if include_5mc:
            pairs["M"] = np.array([pyrimidine + purine + strongbond + nh2 + met + ketone + nh2]).flatten()
            pairs["g"] = np.array([purine + pyrimidine + strongbond + ketone + nh2 + nh2 + met]).flatten()
            #pairs["M"] = np.array([met, h_donor, h_acceptor, h_acceptor, h_acceptor, h_donor, h_acceptor])
            #pairs["g"] = np.array([h_acceptor, h_acceptor, h_donor, met, h_acceptor, h_donor, h_acceptor])
            bppool = bppool + ["M", "g"]
        pairs["N"] = np.mean(np.array(list(pairs.values())), axis = 0)
        diPairs = {}
        for bp1 in bppool:
            for bp2 in bppool:
                diPairs[(bp1, bp2)] = np.concatenate((pairs[bp1], pairs[bp2]))
    else:
        if include_5mc:
            bits = 6
            bases = ["A", "C", "M", "g", "G", "T"]
        else:
            bits = 4
            bases = ["T", "G", "C", "A"]
        pairs = {}
        i = 0
        for base in bases:
            pairs[base] = np.zeros(bits)
            pairs[base][i] = 1
            i += 1
        pairs["N"] = np.ones(bits, dtype = float) / bits
        diPairs = {("N", "N"): np.ones((bits * bits), dtype = float) / bits / bits}
        for i, di in enumerate(itertools.product(bases, repeat = 2)):
            diPairs[di] = np.zeros((bits * bits))
            diPairs[di][i] = 1
        for bp1 in bases:
            diPairs[(bp1, "N")] = np.zeros((bits * bits))
            diPairs[("N", bp1)] = np.zeros((bits * bits))
            for bp2 in bases:
                diPairs[(bp1, "N")] += diPairs[(bp1, bp2)]
                diPairs[("N", bp1)] += diPairs[(bp2, bp1)]
            diPairs[(bp1, "N")] = diPairs[(bp1, "N")] / bits
            diPairs[("N", bp1)] = diPairs[("N", bp1)] / bits
    return pairs, diPairs