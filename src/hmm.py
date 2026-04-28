from collections import Counter
from Bio import SeqIO
import numpy as np

# Lista de nucleotídeos possíveis
nucleotides = ['A', 'T', 'G', 'C']

# Calcula as probabilidades iniciais
def initial_prob (seq):

    # Conta quantos nucleotídeos tem de cada tipo na sequência
    count = Counter(seq)

    # Calcula quantos nucleotídeos no total da sequência
    total = len(seq)

    # Retorna um array com a probailidade incial de cada nucleotídeo
    return np.array([count[n] / total for n in nucleotides])

# Calcula a matriz de probabilidades de transição dada uma sequência
def transition_matrix (seq):

    # Define uma matriz 4x4 incializada com 0
    matrix = np.zeros((4,4))

    # Define um dicionário para os nucleotídeos, incializado com 0
    count = {n: 0 for n in nucleotides}

    # Para cada nucleotídeo da sequência
    for i in range(len(seq) - 1):

        # Define o nucleotídeo (a) e seu posterior (b)
        a, b = seq[i], seq[i+1]

        # Se a e b forem nucleotídeos
        if a in nucleotides and b in nucleotides:
            # Seleciona o índice de a e b na matriz
            i_idx = nucleotides.index(a)
            j_idx = nucleotides.index(b)

            # Soma mais 1 na matriz 
            matrix[i_idx][j_idx] += 1

            # Soma mais um na lista de nucleotídeos
            count[a] += 1

    # Para cada posição (i) e nucleotídeo (n) na lista de nucleotídeos
    for i, n in enumerate(nucleotides):

        # Se a soma do nucleotídeo for maior que 0
        if count[n] > 0:
            # Calcula a probabilidade
            matrix[i] /= count[n]

    # Transforma a matriz 4x4 em um vetor de 16 posições
    return matrix.flatten () 

# Calcula a probabilidade de emissão dada uma sequência
def emission_prob (seq):

    # Conta quantos de cada nucleotídeos a sequência possui
    count = Counter (seq)

    # Calcula o comprimento total da sequência
    total = len(seq)

    # Retorna um array com a probabilidade de emissão da sequência
    return np.array([count[n] / total for n in nucleotides])

# Extrai o vetor de características de acordo com as sequências codificante e não condificante
def extract_features (seq_cds, seq_ncds):

    # Calcula as probabilidades inciais
    pi = initial_prob (seq_cds + seq_ncds)

    # Calcula as probailidades de transição
    trans = transition_matrix (seq_cds + seq_ncds)

    # Calcula as probabilidades de emissão para cada sequência (codificante e não codificante)
    e_cds = emission_prob(seq_cds)
    e_ncds = emission_prob(seq_ncds)

    # Concatena as probabilidades em um vetor
    return np.concatenate([pi, trans, e_cds, e_ncds])
