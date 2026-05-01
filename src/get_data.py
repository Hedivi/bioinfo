"""
get_data_v2.py
==============
Coleta múltiplas variantes genômicas (NG_) por gene do GenBank,
gerando um dataset com várias amostras por gene — igual ao artigo.

Usa requests diretamente para as chamadas à API do NCBI (mais estável
que o wrapper Biopython para esearch/esummary).

Saída:
  genbank/
    cancer/BRCA1/sequences/
        NG_005905_2_BRCA1_CDS.fasta
        NG_005905_2_BRCA1_NCDS.fasta
        ...
    nao_cancer/ATM/sequences/
        ...
"""

import time
import csv
import requests
from pathlib import Path
from io import StringIO
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
import sys

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ──────────────────────────────────────────────────────────────────────────────

EMAIL      = "heloisa.viotto@ufpr.br"
API_KEY    = ""           # recomendado: https://www.ncbi.nlm.nih.gov/account/

Entrez.email   = EMAIL    # usado apenas no efetch via Biopython
Entrez.api_key = API_KEY

OUTPUT_DIR   = Path("genbank")
DELAY        = 0.4        # segundos entre chamadas (0.1 com api_key)
MAX_VARIANTS = 1         # máximo de variantes NG_ por gene

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ──────────────────────────────────────────────────────────────────────────────
# GENES
# ──────────────────────────────────────────────────────────────────────────────

def get_genes (mal, nao_mal):

    files = [mal, nao_mal]
    types = ["cancer", "nao_cancer"]
    genes = []
    for i in range(len(files)):
        file = files[i]
        classe = types[i]
        with open (file, "r") as f:
            for line in f:
                genes.append((line.strip(), classe))

    return genes

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS HTTP
# ──────────────────────────────────────────────────────────────────────────────

def _params_base() -> dict:
    p = {"email": EMAIL}
    if API_KEY:
        p["api_key"] = API_KEY
    return p


def _get(endpoint: str, params: dict, timeout: int = 60) -> requests.Response:
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(
        url,
        params=params,
        headers={"User-Agent": f"Python/get_data_v2 ({EMAIL})"},
        timeout=timeout,
    )
    r.raise_for_status()
    time.sleep(DELAY)
    return r


# ──────────────────────────────────────────────────────────────────────────────
# BUSCA DE VARIANTES NO GENBANK
# ──────────────────────────────────────────────────────────────────────────────

def buscar_uids(accession: str) -> list[str]:
    params = {
        **_params_base(),
        "db": "nuccore",
        "term": f"{accession}[Accession]",
        "retmode": "json",
    }

    r = _get("esearch.fcgi", params)
    data = r.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    return ids


def uid_para_accession(uid: str) -> str:
    """Converte um UID numérico no accession correspondente (ex: NG_005905.2)."""
    params = {
        **_params_base(),
        "db":      "nuccore",
        "id":      uid,
        "retmode": "json",
    }
    r = _get("esummary.fcgi", params)
    result = r.json().get("result", {})
    return result.get(uid, {}).get("accessionversion", "")


def buscar_accessions(simbolo: str) -> list[str]:
    """
    Retorna lista de accessions NG_ disponíveis para o gene.
    """
    print(f"  Buscando variantes de {simbolo}...")

    try:
        uids = buscar_uids(simbolo)
    except Exception as e:
        print(f"  [Erro] esearch falhou para {simbolo}: {e}")
        return []

    if not uids:
        print(f"  [Aviso] Nenhuma entrada encontrada para {simbolo}.")
        return []

    accessions = []
    for uid in uids:
        try:
            acc = uid_para_accession(uid)
            if acc.startswith("NG_"):
                accessions.append(acc)
        except Exception as e:
            print(f"  [Aviso] Falha ao converter UID {uid}: {e}")

    accessions = uids

    print(f"  ✓ {len(accessions)} accessions NG_ encontrados")
    return accessions


# ──────────────────────────────────────────────────────────────────────────────
# DOWNLOAD DO GENBANK
# ──────────────────────────────────────────────────────────────────────────────

def baixar_genbank(accession: str, caminho: Path) -> SeqRecord | None:
    """Baixa um registro GenBank usando requests, com cache local."""
    if caminho.exists():
        try:
            return SeqIO.read(str(caminho), "genbank")
        except Exception:
            caminho.unlink()

    try:
        params = {
            **_params_base(),
            "db":      "nuccore",
            "id":      accession,
            "rettype": "gb",
            "retmode": "text",
        }
        r = _get("efetch.fcgi", params, timeout=120)
        caminho.write_text(r.text, encoding="utf-8")

        record = SeqIO.read(str(caminho), "genbank")
        print(f"    ✓ {accession}: {len(record.seq):,} bp")
        return record

    except Exception as e:
        print(f"    [Erro] Download {accession}: {e}")
        if caminho.exists():
            caminho.unlink()
        return None


# ──────────────────────────────────────────────────────────────────────────────
# EXTRAÇÃO CDS / NCDS
# ──────────────────────────────────────────────────────────────────────────────

def extrair_cds_ncds(record: SeqRecord) -> tuple[str, str]:
    """
    Extrai CDS (exons concatenados) e NCDS (introns + flanqueadoras).
    Retorna ("", "") se o registro não for adequado.
    """
    acc = record.id
    mol = record.annotations.get("molecule_type", "").lower()

    if "rna" in mol or acc.startswith("NM_") or acc.startswith("NR_"):
        return "", ""

    seq     = str(record.seq).upper()
    pos_cds = set()

    for feat in record.features:
        if feat.type == "CDS":
            for parte in feat.location.parts:
                pos_cds.update(range(int(parte.start), int(parte.end)))

    if not pos_cds:
        return "", ""

    cds  = "".join(seq[i] for i in sorted(pos_cds))
    ncds = "".join(seq[i] for i in range(len(seq)) if i not in pos_cds)

    if not cds or not ncds:
        return "", ""

    # Descarta se >95% é CDS — suspeito de mRNA
    if len(cds) / len(seq) > 0.95:
        return "", ""

    return cds, ncds


def salvar_fasta(caminho: Path, seq: str, header: str):
    with open(caminho, "w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE POR GENE
# ──────────────────────────────────────────────────────────────────────────────

def processar_gene(simbolo: str, classe: str) -> list[dict]:
    print(f"\n{'─'*60}")
    print(f"  {simbolo} | {classe}")
    print(f"{'─'*60}")

    base_dir = OUTPUT_DIR / classe / simbolo
    gb_dir   = base_dir / "genbank_files"
    seq_dir  = base_dir / "sequences"
    gb_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)

    accessions = buscar_accessions(simbolo)
    if not accessions:
        return []

    resultados = []
    for acc in accessions:
        nome_gb = f"{acc.replace('.', '_')}.gb"
        record  = baixar_genbank(acc, gb_dir / nome_gb)
        if record is None:
            continue

        cds, ncds = extrair_cds_ncds(record)
        if not cds:
            print(f"    [Ignorado] {acc}: CDS/NCDS inválido")
            continue

        base = f"{acc.replace('.', '_')}_{simbolo}"
        salvar_fasta(seq_dir / f"{base}_CDS.fasta",  cds,  f"{acc}|{simbolo}|CDS")
        salvar_fasta(seq_dir / f"{base}_NCDS.fasta", ncds, f"{acc}|{simbolo}|NCDS")

        resultados.append({
            "gene":      simbolo,
            "classe":    classe,
            "accession": acc,
            "len_total": len(record.seq),
            "len_cds":   len(cds),
            "len_ncds":  len(ncds),
        })

    print(f"  → {len(resultados)}/{len(accessions)} variantes salvas para {simbolo}")
    return resultados


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    todos = []

    GENES = get_genes(sys.argv[1], sys.argv[2])

    for (simbolo, classe) in GENES:
        try:
            resultados = processar_gene(simbolo, classe)
            todos.extend(resultados)
        except Exception as e:
            print(f"\n  [ERRO] {simbolo}: {e}")

    if todos:
        csv_path = Path("variantes.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(todos[0].keys()))
            writer.writeheader()
            writer.writerows(todos)
        print(f"\nCSV salvo: {csv_path}")

    cancer     = [r for r in todos if r["classe"] == "cancer"]
    nao_cancer = [r for r in todos if r["classe"] == "nao_cancer"]

    print(f"\n{'='*60}")
    print(f"RESUMO FINAL")
    print(f"{'='*60}")
    print(f"  Total de amostras : {len(todos)}")
    print(f"  Malignas          : {len(cancer)}")
    print(f"  Não-malignas      : {len(nao_cancer)}")
    print()

    from collections import Counter
    por_gene = Counter(r["gene"] for r in todos)
    print(f"  {'Gene':<10} {'Classe':<12} {'Amostras':>10}")
    print(f"  {'-'*34}")
    for (simbolo, classe) in GENES:
        n = por_gene.get(simbolo, 0)
        print(f"  {simbolo:<10} {classe:<12} {n:>10}")


if __name__ == "__main__":
    main()