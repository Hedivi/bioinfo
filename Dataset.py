import torch
from torch.utils.data import Dataset
import numpy as np
import os

from utils import extract_cds_ncds
from hmm import extract_features

# Define a classe de dataset
class GenDataset:

    # Inicializa o dataset
    def __init__ (self, cancer_dir, normal_dir):

        # Inicializa a lista de features (vetor de 28 características)
        self.X = []

        # Inicializa a lista de labels
        self.Y = []

        # Carrega os dados de câncer e não câncer
        self.load_data(cancer_dir, label=1)
        self.load_data(normal_dir, label=0)

        # Transforma os dados em um array
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    # Carrega os dados
    def load_data (self, folder, label):

        # Para cada arquivo em uma pasta
        for file in os.listdir(folder):

            # Se o arquivo estiver no formato GenBank
            if file.endswith(".gb") or file.endswith(".gbk"):

                # Define o caminho do arquivo
                path = os.path.join(folder, file)

                # Extrai as sequências codificantes e não codificantes de um arquivo GenBank
                cds, ncds = extract_cds_ncds(path)

                # Se o comprimento das sequências for igual a zero, continue
                if len(cds) == 0 and len(ncds) == 0:
                    continue

                # Extrai o vetor de 28 posições das sequências
                features = extract_features(cds, ncds)

                # Adiciona a feature e a label nos vetores
                self.X.append(features)
                self.Y.append(label)

    # Coleta um dado
    def get_data(self):
        return self.X, self.Y
    
    # Define o tamanho do dicionário
    def __len__(self):
        return len(self.X)

    # Reliza a divisão do dataset em K conjunto. Retorna apenas os índices.
    def cross_validation_split (self, k=10):

        # Define o tamanho de cada conjunto
        fold_size = self.len() // k

        # Define uma lista de índices
        indices = np.arrange(self.len())

        # Define a lista de índices, inicialmente vazia
        folds = []
        # Para cada K
        for i in range(k):

            # Separa os índices de teste e treino. De teste, é o conjunto atual (i). Para treino, é o resto.
            test = indices[i * fold_size: (i + 1) * fold_size]
            train = np.concatenate([indices[:i * fold_size], indices[(i+1) * fold_size:]])

            # Adiciona uma tupla de índices na lista
            folds.append((train, test))

        # Retorna a lista de índices
        return folds
    
    # Obtém os dados relacionados a uma lista de índices
    def get_fold (self, train_index, test_index):

        # Define as listas
        train_X = [], train_Y = []
        test_X = [], test_Y = []

        # Para cada dado do dataset
        for i in range(self.len()):

            # Se o índice estiver presente na lista de índices de treino, adiciona na lista de dados de treino
            if i in train_index:
                train_X.append(self.X[i])
                train_Y.append(self.Y[i])

            # Se o índice estiver presente na lista de índices de teste, adiciona na lista de dados de teste
            elif i in test_index:
                test_X.append(self.X[i])
                test_Y.append(self.Y[i])

        # Retorna as listas com os dados
        return train_X, train_Y, test_X, test_Y
       
    