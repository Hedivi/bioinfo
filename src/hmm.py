from collections import Counter
from Bio import SeqIO
import numpy as np

nucleotides = ['A', 'T', 'G', 'C']

def initial_prob(seq_completa):
    """
    Probabilidade inicial π(i) — Equação 4 do artigo.
    Distribuição dos nucleotídeos na sequência completa (CDS + NCDS).
    Representa: qual a chance de a sequência começar em cada estado oculto.
    Retorna vetor de 4 valores: [P(A), P(T), P(G), P(C)]
    """
    count = Counter(seq_completa)
    total = len(seq_completa)
    return np.array([count[n] / total for n in nucleotides])


def transition_matrix(seq_completa):
    """
    Matriz de transição P(i,j) — Equação 5 do artigo.
    Probabilidade de ir do estado oculto i para o estado oculto j.
    Ou seja: dado que o nucleotídeo atual é X, qual a chance do próximo ser Y?
    Calculada na sequência completa (CDS + NCDS juntas).
    Retorna vetor de 16 valores (matriz 4x4 achatada).
    """
    matrix = np.zeros((4, 4))
    count = {n: 0 for n in nucleotides}

    for i in range(len(seq_completa) - 1):
        a, b = seq_completa[i], seq_completa[i + 1]
        if a in nucleotides and b in nucleotides:
            i_idx = nucleotides.index(a)
            j_idx = nucleotides.index(b)
            matrix[i_idx][j_idx] += 1
            count[a] += 1

    # Normaliza cada linha pela quantidade de vezes que aquele nucleotídeo aparece
    for i, n in enumerate(nucleotides):
        if count[n] > 0:
            matrix[i] /= count[n]

    return matrix.flatten()


def emission_prob(seq_regiao):
    """
    Probabilidade de emissão E(i) — Equação 6 do artigo.
    Dado que estamos numa região observável (CDS ou NCDS),
    qual a probabilidade de cada nucleotídeo ser emitido?
    
    Chamada duas vezes:
        emission_prob(seq_cds)  → E(A|CDS), E(T|CDS), E(G|CDS), E(C|CDS)
        emission_prob(seq_ncds) → E(A|NCDS), E(T|NCDS), E(G|NCDS), E(C|NCDS)
    
    Retorna vetor de 4 valores.
    """
    count = Counter(seq_regiao)
    total = len(seq_regiao)
    return np.array([count[n] / total for n in nucleotides])


def extract_features(seq_cds, seq_ncds):
    """
    Monta o vetor Θ = {π(i), P(i,j), E_CDS(i), E_NCDS(i)} — Passo 2 do artigo.
    
    Estados ocultos:  A, T, G, C  (os nucleotídeos)
    Estados observados: CDS e NCDS (as regiões)
    
    Composição do vetor final de 28 features:
        [0:4]   → probabilidade inicial       π(i)        — 4 valores
        [4:20]  → matriz de transição         P(i,j)      — 16 valores
        [20:24] → emissão na região CDS       E_CDS(i)    — 4 valores
        [24:28] → emissão na região NCDS      E_NCDS(i)   — 4 valores
    """
    seq_completa = seq_cds + seq_ncds

    # π(i): distribuição inicial dos nucleotídeos
    pi = initial_prob(seq_completa)

    # P(i,j): transições entre nucleotídeos (estados ocultos)
    trans = transition_matrix(seq_completa)

    # E(i|CDS): probabilidade de cada nucleotídeo dado que estamos no CDS
    e_cds = emission_prob(seq_cds)

    # E(i|NCDS): probabilidade de cada nucleotídeo dado que estamos no NCDS
    e_ncds = emission_prob(seq_ncds)

    return np.concatenate([pi, trans, e_cds, e_ncds])