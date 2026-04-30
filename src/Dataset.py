"""
dataset.py
==========
Dataset PyTorch compatível com os arquivos gerados por vcv_to_genbank.py.

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

    Se você não tiver os dois grupos separados, veja a seção
    "USO COM ARQUIVO CSV" no final deste arquivo.
"""

import os
import numpy as np
from pathlib import Path
from Bio import SeqIO
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from hmm import extract_features   # extrai vetor de 28 features HMM



def ler_fasta(caminho: str | Path) -> str:
    """Lê um arquivo FASTA e retorna a sequência como string."""
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


# ══════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════

class GenDataset(Dataset):
    """
    Parâmetros
    ----------
    cancer_seq_dir : str | Path
        Pasta com os FASTAs das sequências MALIGNAS (label=1).
        Deve conter pares *_CDS.fasta e *_NCDS.fasta.

    normal_seq_dir : str | Path
        Pasta com os FASTAs das sequências NÃO-MALIGNAS (label=0).
        Mesma estrutura.

    Exemplo de uso
    --------------
    Se você ainda não tem os dois grupos separados (só tem uma pasta),
    consulte a seção ao final deste arquivo.
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

    # ── Carga ──────────────────────────────────────────────────────

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

    # ── Interface Dataset ──────────────────────────────────────────

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """Retorna (features: Tensor, label: int)."""
        import torch
        return torch.tensor(self.X[idx], dtype=torch.float32), int(self.Y[idx])

    def get_data(self):
        """Retorna (X, Y) como arrays NumPy."""
        return self.X, self.Y

    # ── Cross-validation ───────────────────────────────────────────

    def cross_validation_split(self, k: int = 10):
        """
        Divide em k folds estratificados.
        Retorna lista de (train_index, test_index).
        """
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        return list(skf.split(self.X, self.Y))

    def get_fold(self, train_index, test_index):
        """
        Retorna (X_train, Y_train, X_test, Y_test) para um fold.
        Usa indexação NumPy — muito mais rápido que loop Python.
        """
        X_train = self.X[train_index]
        Y_train = self.Y[train_index]
        X_test  = self.X[test_index]
        Y_test  = self.Y[test_index]
        return X_train, Y_train, X_test, Y_test

    # ── Utilidades ─────────────────────────────────────────────────

    def info(self):
        """Imprime um resumo do dataset."""
        print(f"\n{'─'*45}")
        print(f"  Amostras totais   : {len(self.Y)}")
        print(f"  Malignas  (1)     : {int(self.Y.sum())}")
        print(f"  Não-malignas (0)  : {int((self.Y == 0).sum())}")
        print(f"  Dimensão features : {self.X.shape[1]}")
        print(f"{'─'*45}\n")


# ══════════════════════════════════════════════════════
# VARIANTE: carregar direto de um CSV (sem duas pastas)
# ══════════════════════════════════════════════════════

class GenDatasetFromCSV(Dataset):
    """
    Alternativa quando você tem um CSV com colunas:
        vcv_id, label, cds_fasta, ncds_fasta

    Exemplo de CSV:
        vcv_id,label,cds_fasta,ncds_fasta
        VCV001686624.2,1,sequences/VCV001686624_2_BRCA1_CDS.fasta,...
        VCV001327189.3,0,sequences/VCV001327189_3_TP53_CDS.fasta,...
    """

    def __init__(self, csv_path: str):
        import csv
        self.X = []
        self.Y = []
        self.nomes = []

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cds  = ler_fasta(row["cds_fasta"])
                    ncds = ler_fasta(row["ncds_fasta"])
                    features = extract_features(cds, ncds)
                    self.X.append(features)
                    self.Y.append(int(row["label"]))
                    self.nomes.append(row["vcv_id"])
                except Exception as e:
                    print(f"  [ERRO] {row.get('vcv_id','?')}: {e}")

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.int64)

    def __len__(self):        return len(self.X)
    def __getitem__(self, i):
        import torch
        return torch.tensor(self.X[i], dtype=torch.float32), int(self.Y[i])
    def get_data(self):       return self.X, self.Y
    def cross_validation_split(self, k=10):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        return list(skf.split(self.X, self.Y))
    def get_fold(self, train_index, test_index):
        return (self.X[train_index], self.Y[train_index],
                self.X[test_index],  self.Y[test_index])


# ══════════════════════════════════════════════════════
# EXEMPLO DE USO
# ══════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Opção 1: duas pastas separadas ────────────────
    # dataset = GenDataset(
    #     cancer_seq_dir = "genbank_output/sequences/cancer",
    #     normal_seq_dir = "genbank_output/sequences/normal",
    # )

    # ── Opção 2: uma pasta com todos os FASTAs ────────
    # (todos os VCVs estão na mesma pasta; você precisa
    #  separar manualmente quais são cancer/normal)

    # ── Opção 3: via CSV ──────────────────────────────
    # dataset = GenDatasetFromCSV("labels.csv")

    # ── Rodar cross-validation ────────────────────────
    # dataset.info()
    # folds = dataset.cross_validation_split(k=10)
    # for fold_num, (train_idx, test_idx) in enumerate(folds):
    #     X_tr, Y_tr, X_te, Y_te = dataset.get_fold(train_idx, test_idx)
    #     print(f"Fold {fold_num+1}: treino={len(Y_tr)}, teste={len(Y_te)}")

    print("dataset.py carregado com sucesso.")
