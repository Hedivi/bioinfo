"""
Dataset para extração de features HMM de sequências de DNA.

Estrutura esperada dos arquivos:
    genbank/
    ├── cancer/
    │   ├── BRCA1/sequences/VCV001_BRCA1_CDS.fasta
    │   ├── BRCA1/sequences/VCV001_BRCA1_NCDS.fasta
    │   ├── TP53/sequences/VCV002_TP53_CDS.fasta
    │   └── ...
    └── nao_cancer/
        ├── ATM/sequences/VCV003_ATM_CDS.fasta
        └── ...

Os arquivos CDS e NCDS de um mesmo prefixo são emparelhados automaticamente.

Labels:
    cancer_base_dir  → label 1  (malignant)
    normal_base_dir  → label 0  (non-malignant)
"""

import numpy as np
from pathlib import Path
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold

from hmm import extract_features

NUCLEOTIDES = set("ATGC")


def ler_fasta(caminho: Path) -> str:
    """
    Lê um arquivo FASTA com um ou mais registros.
    Concatena todos os registros, converte para maiúsculas
    e filtra nucleotídeos ambíguos (N, R, Y, etc.).
    """
    sequencia = ""
    for record in SeqIO.parse(str(caminho), "fasta"):
        seq = str(record.seq).upper()
        seq = "".join(c for c in seq if c in NUCLEOTIDES)
        sequencia += seq
    return sequencia


def encontrar_pares(seq_dir: Path) -> dict:
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
    pares = {}

    for arquivo in sorted(seq_dir.glob("*.fasta")):
        nome = arquivo.stem

        if nome.endswith("_CDS"):
            base = nome[:-4]
            pares.setdefault(base, {})["cds"] = arquivo
        elif nome.endswith("_NCDS"):
            base = nome[:-5]
            pares.setdefault(base, {})["ncds"] = arquivo

    completos   = {k: v for k, v in pares.items() if "cds" in v and "ncds" in v}
    incompletos = set(pares) - set(completos)

    if incompletos:
        print(f"  [Aviso] {len(incompletos)} arquivo(s) sem par CDS/NCDS ignorado(s):")
        for k in sorted(incompletos):
            print(f"    - {k}")

    return completos


def _coletar_seq_dirs(base_dir: Path) -> list:
    """
    Dado um diretório raiz (ex: genbank/cancer/), retorna a lista de
    subpastas sequences/ de cada gene encontrado.
    """
    seq_dirs = []
    for gene_dir in sorted(base_dir.iterdir()):
        if not gene_dir.is_dir():
            continue
        seq_dir = gene_dir / "sequences"
        if seq_dir.exists():
            seq_dirs.append((gene_dir.name, seq_dir))
        else:
            print(f"  [Aviso] {gene_dir.name}: pasta sequences/ não encontrada — ignorado.")
    return seq_dirs


class GenDataset:
    """
    Carrega todos os genes de duas pastas raiz (cancer e nao_cancer),
    extrai o vetor de 28 features HMM de cada par CDS/NCDS e monta X, Y.

    Cada par de arquivos (CDS + NCDS) de cada variante de cada gene
    vira uma amostra (linha) no dataset — igual ao artigo.
    """

    def __init__(self, cancer_base_dir: str, normal_base_dir: str):
        self._X: list = []
        self._Y: list = []
        self.nomes: list = []   # lista de (gene, base) para rastreabilidade

        cancer_path = Path(cancer_base_dir)
        normal_path = Path(normal_base_dir)

        print("[Dataset] Carregando genes malignos...")
        for gene, seq_dir in _coletar_seq_dirs(cancer_path):
            print(f"  → {gene}")
            self._carregar_diretorio(seq_dir, label=1, gene=gene)

        print("\n[Dataset] Carregando genes não-malignos...")
        for gene, seq_dir in _coletar_seq_dirs(normal_path):
            print(f"  → {gene}")
            self._carregar_diretorio(seq_dir, label=0, gene=gene)

        if len(self._X) == 0:
            raise RuntimeError(
                "Nenhuma amostra carregada. Verifique os diretórios e os arquivos FASTA."
            )

        self.X = np.array(self._X, dtype=np.float32)
        self.Y = np.array(self._Y, dtype=np.int64)

        n_maligno = int(self.Y.sum())
        n_normal  = int((self.Y == 0).sum())
        print(f"\n[Dataset] Total: {len(self.Y)} amostras "
              f"| Malignas: {n_maligno} "
              f"| Não-malignas: {n_normal}")

        min_classe = min(n_maligno, n_normal)
        if min_classe < 10:
            print(f"  [Aviso] Menor classe tem {min_classe} amostras. "
                  f"Use k <= {min_classe} no cross_validation_split.")

    def _carregar_diretorio(self, seq_dir: Path, label: int, gene: str):
        pares = encontrar_pares(seq_dir)

        if not pares:
            print(f"    [Aviso] Nenhum par CDS/NCDS encontrado em: {seq_dir}")
            return

        erros = 0
        for base, arquivos in pares.items():
            try:
                cds  = ler_fasta(arquivos["cds"])
                ncds = ler_fasta(arquivos["ncds"])

                if len(cds) == 0 or len(ncds) == 0:
                    print(f"    [Aviso] {base}: sequência vazia após filtragem — ignorado.")
                    erros += 1
                    continue

                features = extract_features(cds, ncds)
                self._X.append(features)
                self._Y.append(label)
                self.nomes.append((gene, base))

            except Exception as e:
                erros += 1
                print(f"    [Erro] {base}: {e}")

        ok = len(pares) - erros
        print(f"    ✓ {ok}/{len(pares)} amostras carregadas")

    def __len__(self):
        return len(self.Y)

    def cross_validation_split(self, k: int = 10):
        """Divide em k folds estratificados."""
        min_classe = int(min(self.Y.sum(), (self.Y == 0).sum()))
        if k > min_classe:
            raise ValueError(
                f"k={k} inviável: menor classe tem {min_classe} amostras. "
                f"Use k <= {min_classe}."
            )
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        return list(skf.split(self.X, self.Y))

    def get_fold(self, train_index, test_index):
        """Retorna (X_train, Y_train, X_test, Y_test) para um fold."""
        return (
            self.X[train_index], self.Y[train_index],
            self.X[test_index],  self.Y[test_index],
        )

    def info(self):
        print(f"\n{'─'*45}")
        print(f"  Amostras totais   : {len(self.Y)}")
        print(f"  Malignas  (1)     : {int(self.Y.sum())}")
        print(f"  Não-malignas (0)  : {int((self.Y == 0).sum())}")
        print(f"  Dimensão features : {self.X.shape[1]}")

        print(f"\n  Distribuição por gene:")
        gene_counts: dict = {}
        for (gene, _), label in zip(self.nomes, self.Y):
            gene_counts.setdefault(gene, {0: 0, 1: 0})
            gene_counts[gene][label] += 1
        for gene, counts in sorted(gene_counts.items()):
            print(f"    {gene:<12} maligno={counts[1]:>3}  normal={counts[0]:>3}")
        print(f"{'─'*45}\n")
