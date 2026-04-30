"""

Estrutura esperada dos arquivos:
    sequences/
        VCV001686624_2_BRCA1_CDS.fasta
        VCV001686624_2_BRCA1_NCDS.fasta
        VCV001327189_3_TP53_CDS.fasta
        ...

Os arquivos CDS e NCDS de um mesmo VCV são emparelhados automaticamente
pelo prefixo (tudo antes de _CDS ou _NCDS).

Labels:
    cancer_vcvs  → label 1  (malignant)
    normal_vcvs  → label 0  (non-malignant)

"""

import os
import numpy as np
from pathlib import Path
from Bio import SeqIO
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from hmm import extract_features   # extrai vetor de 28 features HMM



# Lê um arquivo FASTA e retorna a sequência como string."""
def ler_fasta(caminho: str | Path) -> str:
    record = SeqIO.read(str(caminho), "fasta")
    return str(record.seq).upper()


def encontrar_pares(seq_dir: str | Path) -> dict:
    """
    Varre seq_dir e agrupa arquivos CDS/NCDS pelo prefixo base.

    Retorna:
        {
          "VCV001686624_2_BRCA1": {
              "cds":  Path(..._CDS.fasta),
              "ncds": Path(..._NCDS.fasta)
          },
          ...
        }
    """
    seq_dir = Path(seq_dir)
    pares = {}

    for arquivo in sorted(seq_dir.glob("*.fasta")):
        nome = arquivo.stem   # ex: VCV001686624_2_BRCA1_CDS

        if nome.endswith("_CDS"):
            base = nome[:-4]   # remove _CDS
            pares.setdefault(base, {})["cds"] = arquivo

        elif nome.endswith("_NCDS"):
            base = nome[:-5]   # remove _NCDS
            pares.setdefault(base, {})["ncds"] = arquivo

    # Mantém só pares completos
    completos = {k: v for k, v in pares.items()
                 if "cds" in v and "ncds" in v}

    incompletos = set(pares) - set(completos)
    if incompletos:
        print(f"[Aviso] {len(incompletos)} arquivo(s) sem par CDS/NCDS ignorado(s):")
        for k in incompletos:
            print(f"  - {k}")

    return completos


class GenDataset(Dataset):

    """
    O construtor recebe duas pastas: uma com sequências de genes associados a câncer e outra com sequências normais. 
    Cada pasta tem seus arquivos FASTA. O X vai guardar os vetores de características e o Y vai guardar os rótulos correspondentes
    """
    def __init__(self, cancer_seq_dir: str, normal_seq_dir: str):
        self.X: list = []
        self.Y: list = []
        self.nomes: list = []   # guarda o nome base para rastreabilidade

        print("[Dataset] Carregando sequências malignas...")
        self._carregar_diretorio(cancer_seq_dir, label=1)

        print("[Dataset] Carregando sequências não-malignas...")
        self._carregar_diretorio(normal_seq_dir, label=0)

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.int64)

        print(f"[Dataset] Total: {len(self.Y)} amostras "
              f"| Malignas: {self.Y.sum()} "
              f"| Não-malignas: {(self.Y == 0).sum()}")


    """
    Para cada par de arquivos, lê as duas sequências e passa para extract_features, que é importada do módulo hmm. 
    Essa função é o coração científico desse código — ela recebe o CDS e o NCDS e devolve um vetor de 28 números 
    que representam as probabilidades do modelo HMM treinado nessas sequências. Esses 28 números são as características que o classificador vai usar.
    """
    def _carregar_diretorio(self, seq_dir: str, label: int):
        pares = encontrar_pares(seq_dir)

        if not pares:
            print(f"  [Aviso] Nenhum par CDS/NCDS encontrado em: {seq_dir}")
            return

        erros = 0
        for base, arquivos in pares.items():
            try:
                cds  = ler_fasta(arquivos["cds"])
                ncds = ler_fasta(arquivos["ncds"])

                # Extrai vetor de 28 features HMM (inicial + transição + emissão)
                features = extract_features(cds, ncds)

                self.X.append(features)
                self.Y.append(label)
                self.nomes.append(base)

            except Exception as e:
                erros += 1
                print(f"  [ERRO] {base}: {e}")

        ok = len(pares) - erros
        print(f"  ✓ {ok}/{len(pares)} amostras carregadas de {seq_dir}")


    def __len__(self):
        return len(self.X)


    """Retorna (features: Tensor, label: int)."""
    def __getitem__(self, idx):
        import torch
        return torch.tensor(self.X[idx], dtype=torch.float32), int(self.Y[idx])


    """Retorna (X, Y) como arrays NumPy."""
    def get_data(self):
        return self.X, self.Y


    """
    Divide em k folds estratificados.
    Retorna lista de (train_index, test_index).
    """
    def cross_validation_split(self, k: int = 10):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        return list(skf.split(self.X, self.Y))

    """
    Retorna (X_train, Y_train, X_test, Y_test) para um fold.
    Usa indexação NumPy — muito mais rápido que loop Python.
    """
    def get_fold(self, train_index, test_index):
        X_train = self.X[train_index]
        Y_train = self.Y[train_index]
        X_test  = self.X[test_index]
        Y_test  = self.Y[test_index]
        return X_train, Y_train, X_test, Y_test


    """Imprime um resumo do dataset."""
    def info(self):
        print(f"\n{'─'*45}")
        print(f"  Amostras totais   : {len(self.Y)}")
        print(f"  Malignas  (1)     : {int(self.Y.sum())}")
        print(f"  Não-malignas (0)  : {int((self.Y == 0).sum())}")
        print(f"  Dimensão features : {self.X.shape[1]}")
        print(f"{'─'*45}\n")