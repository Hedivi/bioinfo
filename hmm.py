from collections import Counter
from Bio import SeqIO
import numpy as np

nucleotides = ['A', 'T', 'G', 'C']

def initial_prob (seq):
    count = Counter(seq)
    total = len(seq)
    return np.array([count[n] / total for n in nucleotides])

def transition_matrix (seq):
    matrix = np.zeros((4,4))
    count = {n: 0 for n in nucleotides}

    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i+1]
        if a in nucleotides and b in nucleotides:
            i_idx = nucleotides.index(a)
            j_idx = nucleotides.index(b)
            matrix[i_idx][j_idx] += 1
            count[a] += 1

    for i, n in enumerate(nucleotides):
        if count[n] > 0:
            matrix[i] /= count[n]

    return matrix.flatten () # vira vetor de 16 posições

def emission_prob (seq):
    count = Counter (seq)
    total = len(seq)
    return np.array([count[n] / total for n in nucleotides])

def extract_features (seq_cds, seq_ncds):
    pi = initial_prob (seq_cds + seq_ncds)
    trans = transition_matrix (seq_cds + seq_ncds)

    e_cds = emission_prob(seq_cds)
    e_ncds = emission_prob(seq_ncds)

    return np.concatenate([pi, trans, e_cds, e_ncds])
