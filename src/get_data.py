"""
VCV ClinVar → GenBank → Extração CDS e NCDS
============================================
Replicação do pipeline do artigo:
  "Cancer detection with various classification models:
   A comprehensive feature analysis using HMM to extract a nucleotide pattern"
  Kalal & Jha, Computational Biology and Chemistry, 2024

Requisitos:
    pip install biopython requests

Uso:
    python vcv_to_genbank.py
"""

import os
import time
import csv
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
import sys


Entrez.email   = "heloisa.viotto@ufpr.br"
Entrez.api_key = ""
OUTPUT_DIR     = Path("../Genbank")
DELAY          = 0.4                             # segundos entre chamadas. Evita sobrecarga
INPUT = sys.argv[1]

# Le um arquivo informado como argumento do código python3. E armazena os códigos ID das variantes
VCV_IDS = []
with open(INPUT, "r") as f:
    for line in f:
        VCV_IDS.append(line.strip())



#─────────────────────────────────────────────


# ══════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════

# A API do clinvar só aceita os números do ID da variante. Sendo assim, precisamos fazer uma limpeza no ID: 'VCV001239650.4' → '1239650'
def limpeza_vcv(vcv_id: str) -> str:
    return vcv_id.replace("VCV", "").split(".")[0].lstrip("0")


def _p() -> dict:
    """Params base para Entrez REST."""
    p = {"email": Entrez.email}
    if Entrez.api_key:
        p["api_key"] = Entrez.api_key
    return p


def _h() -> dict:
    """Headers para requests."""
    return {"User-Agent": f"Python/vcv_pipeline ({Entrez.email})"}


# ══════════════════════════════════════════════
# PASSO 1 — INFO CLINVAR (3 estratégias)
# ══════════════════════════════════════════════

# Primeiramente, apenas armazena a númeração do ID da variante em um dicionário.
def _info_vazia(vcv_numero):
    return {"vcv_numero": vcv_numero, "variante_nome": "",
            "gene_simbolo": "", "gene_id": "", "refseq_ids": []}


"""
Faz uma requisição GET para a API do NCBI pedindo um resumo em JSON. 
O resultado vem como um dicionário Python. 
O código navega por dentro desse dicionário buscando os campos genes (símbolo e ID do gene) e sequence_locations dentro de variation_set (os accessions RefSeq)
"""
def estrategia_esummary(vcv_numero: str) -> dict:
    """esummary JSON — rápido, retorna dados básicos."""
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        params={**_p(), "db": "clinvar", "id": vcv_numero, "retmode": "json"},
        headers=_h(), timeout=20)
    r.raise_for_status()
    time.sleep(DELAY)

    # Formata a requisição em json
    data   = r.json().get("result", {})

    # Obtem o número da variante
    record = data.get(vcv_numero) or data.get(str(int(vcv_numero)))
    if not record:
        return _info_vazia(vcv_numero)

    info = _info_vazia(vcv_numero)

    # Obtem o nome da variante
    info["variante_nome"] = record.get("title", "")

    # Obtem as informações de genes (Simbolo e ID)
    genes = record.get("genes", [])
    if genes:
        info["gene_simbolo"] = genes[0].get("symbol", "")
        info["gene_id"]      = str(genes[0].get("geneid", ""))

    # Capturamos o accessions. Referencia do gene que possui a região codificante e não codificante.
    for vs in record.get("variation_set", []):
        for m in vs.get("measures", []):
            for sl in m.get("sequence_locations", []):
                acc = sl.get("accession", "")
                if acc and acc not in info["refseq_ids"]:
                    info["refseq_ids"].append(acc)
    return info



"""
Outra forma de obter os dados da variante, porém agora em arquivos XML (linguagem de marcação)
"""
def estrategia_efetch_xml(vcv_numero: str) -> dict:
    """
    efetch XML — mais completo.
    O id numérico do ClinVar é o mesmo que o UID do banco.
    """
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={**_p(), "db": "clinvar", "id": vcv_numero,
                "rettype": "vcv", "retmode": "xml"},
        headers=_h(), timeout=30)
    r.raise_for_status()
    time.sleep(DELAY)

    root = ET.fromstring(r.text)
    info = _info_vazia(vcv_numero)

    for e in root.iter("VariationName"):
        info["variante_nome"] = e.text or ""; break
    for e in root.iter("Gene"):
        info["gene_simbolo"] = e.get("Symbol", "")
        info["gene_id"]      = e.get("GeneID", ""); break
    for e in root.iter("SequenceLocation"):
        acc = e.get("Accession", "")
        if acc and acc not in info["refseq_ids"]:
            info["refseq_ids"].append(acc)
    for e in root.iter("HGVSExpression"):
        acc = e.get("AccessionVersion", "").split(":")[0]
        if acc and acc not in info["refseq_ids"]:
            info["refseq_ids"].append(acc)
    return info


#  Usa a API REST mais moderna do ClinVar, que retorna JSON com uma estrutura diferente das outras duas.
def estrategia_rest_clinvar(vcv_numero: str) -> dict:
    """API REST moderna do ClinVar."""
    r = requests.get(
        "https://clinvar.ncbi.nlm.nih.gov/api/rest/vcv",
        params={"variation_id": vcv_numero, "format": "json"},
        headers=_h(), timeout=20)
    r.raise_for_status()
    time.sleep(DELAY)

    data = r.json()
    info = _info_vazia(vcv_numero)
    info["variante_nome"] = data.get("variation_name", "")

    for g in data.get("genes", []):
        info["gene_simbolo"] = g.get("symbol", "")
        info["gene_id"]      = str(g.get("gene_id", "")); break

    for allele in [data.get("variation", {})]:
        if not allele: continue
        for loc in allele.get("allele", {}).get("sequence_locations", []):
            acc = loc.get("accession", "")
            if acc and acc not in info["refseq_ids"]:
                info["refseq_ids"].append(acc)
    return info


def buscar_info_clinvar(vcv_numero: str) -> dict:
    """Tenta 3 estratégias; retorna o primeiro resultado com dados úteis."""
    estrategias = [
        ("esummary",   estrategia_esummary),
        ("efetch XML", estrategia_efetch_xml),
        ("REST API",   estrategia_rest_clinvar),
    ]
    ultimo_erro = None
    for nome, fn in estrategias:
        try:
            print(f"    → {nome}...")
            info = fn(vcv_numero)
            if info.get("gene_simbolo") or info.get("refseq_ids"):
                print(f"    ✓ Sucesso com {nome}")
                return info
            print(f"    ✗ {nome}: dados insuficientes")
        except requests.HTTPError as e:
            print(f"    ✗ {nome}: HTTP {e.response.status_code}")
            ultimo_erro = e
        except Exception as e:
            print(f"    ✗ {nome}: {e}")
            ultimo_erro = e

    raise RuntimeError(f"Todas as estratégias falharam. Último erro: {ultimo_erro}")


# ══════════════════════════════════════════════
# PASSO 2 — RESOLVER ACCESSION GENBANK
# ══════════════════════════════════════════════

def priorizar_refseq(ids: list) -> str | None:
    for pref in ["NG_", "NM_", "NR_", "NC_"]:
        for acc in ids:
            if acc.startswith(pref):
                return acc
    return ids[0] if ids else None


def buscar_ng_via_elink(gene_id: str) -> str | None:
    if not gene_id: return None
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
            params={**_p(), "dbfrom": "gene", "db": "nuccore", "id": gene_id,
                    "linkname": "gene_nuccore_refseqgene", "retmode": "json"},
            headers=_h(), timeout=20)
        r.raise_for_status(); time.sleep(DELAY)

        uids = []
        for ls in r.json().get("linksets", []):
            for lsdb in ls.get("linksetdbs", []):
                for lk in lsdb.get("links", []):
                    uids.append(str(lk))
        if not uids: return None

        r2 = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={**_p(), "db": "nuccore", "id": uids[0],
                    "rettype": "acc", "retmode": "text"},
            headers=_h(), timeout=20)
        r2.raise_for_status(); time.sleep(DELAY)

        acc = r2.text.strip()
        if acc.startswith("NG_"):
            print(f"    ✓ NG_ via elink: {acc}")
            return acc
    except Exception as e:
        print(f"    ✗ elink falhou: {e}")
    return None


def buscar_ng_via_esearch(simbolo: str) -> str | None:
    if not simbolo: return None
    try:
        r = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={**_p(), "db": "nuccore", "retmax": 5, "retmode": "json",
                    "term": f"{simbolo}[GENE] AND RefSeqGene[KEYWORD] AND Homo sapiens[ORGN]"},
            headers=_h(), timeout=20)
        r.raise_for_status(); time.sleep(DELAY)

        ids = r.json().get("esearchresult", {}).get("idlist", [])
        for uid in ids[:5]:
            r2 = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={**_p(), "db": "nuccore", "id": uid,
                        "rettype": "acc", "retmode": "text"},
                headers=_h(), timeout=20)
            r2.raise_for_status(); time.sleep(DELAY)
            acc = r2.text.strip()
            if acc.startswith("NG_"):
                print(f"    ✓ NG_ via esearch: {acc}")
                return acc
    except Exception as e:
        print(f"    ✗ esearch falhou: {e}")
    return None


def resolver_accession(info: dict) -> str | None:
    acc = priorizar_refseq(info["refseq_ids"])
    if acc and acc.startswith("NG_"):
        return acc
    # Tentar obter NG_ de outras formas
    ng = (buscar_ng_via_elink(info["gene_id"])
          or buscar_ng_via_esearch(info["gene_simbolo"]))
    return ng or acc   # fallback para o que tiver


# ══════════════════════════════════════════════
# PASSO 3 — DOWNLOAD GENBANK
# ══════════════════════════════════════════════

def baixar_genbank(accession: str, caminho: Path) -> SeqRecord:
    if caminho.exists():
        print(f"    → Cache: {caminho.name}")
        return SeqIO.read(caminho, "genbank")

    print(f"    → Baixando: {accession}...")
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={**_p(), "db": "nuccore", "id": accession,
                "rettype": "gb", "retmode": "text"},
        headers=_h(), timeout=120)
    r.raise_for_status(); time.sleep(DELAY)

    caminho.write_text(r.text, encoding="utf-8")
    print(f"    ✓ Salvo: {caminho.name}")
    return SeqIO.read(caminho, "genbank")


# ══════════════════════════════════════════════
# PASSO 4 — EXTRAÇÃO CDS / NCDS
# ══════════════════════════════════════════════

def extrair_cds_ncds(record: SeqRecord) -> tuple:
    seq       = str(record.seq).upper()
    pos_cds   = set()
    n_features = 0

    for feat in record.features:
        if feat.type == "CDS":
            n_features += 1
            for parte in feat.location.parts:
                pos_cds.update(range(int(parte.start), int(parte.end)))

    cds  = "".join(seq[i] for i in sorted(pos_cds))
    ncds = "".join(seq[i] for i in range(len(seq)) if i not in pos_cds)
    return cds, ncds, n_features


def salvar_fasta(caminho: Path, seq: str, header: str):
    with open(caminho, "w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")


# ══════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════

def processar_vcv(vcv_id: str, gb_dir: Path, seq_dir: Path) -> dict:
    r = {"vcv_id": vcv_id, "variante_nome": "", "gene_simbolo": "",
         "accession_usado": "", "comprimento_total": 0,
         "num_features_cds": 0, "comprimento_cds": 0,
         "comprimento_ncds": 0, "status": "pendente"}

    vcv_num = limpeza_vcv(vcv_id)

    print(f"  [1] Info ClinVar...")
    info = buscar_info_clinvar(vcv_num)
    r["variante_nome"] = info["variante_nome"]
    r["gene_simbolo"]  = info["gene_simbolo"]
    print(f"      Gene: {info['gene_simbolo']} | {info['variante_nome']}")
    print(f"      RefSeqs: {info['refseq_ids']}")

    print(f"  [2] Resolvendo accession...")
    acc = resolver_accession(info)
    if not acc:
        r["status"] = "erro: sem accession"; return r
    r["accession_usado"] = acc
    print(f"      → {acc}")

    print(f"  [3] GenBank...")
    nome_gb = f"{vcv_id.replace('.','_')}_{acc.replace('.','_')}.gb"
    record  = baixar_genbank(acc, gb_dir / nome_gb)
    r["comprimento_total"] = len(record.seq)
    print(f"      {len(record.seq):,} bp")

    print(f"  [4] Extraindo CDS/NCDS...")
    cds, ncds, n = extrair_cds_ncds(record)
    r.update({"num_features_cds": n, "comprimento_cds": len(cds),
               "comprimento_ncds": len(ncds)})
    print(f"      CDS: {len(cds):,} nt | NCDS: {len(ncds):,} nt | features: {n}")

    base = f"{vcv_id.replace('.','_')}_{info['gene_simbolo']}"
    salvar_fasta(seq_dir / f"{base}_CDS.fasta",  cds,
                 f"{vcv_id} | {info['gene_simbolo']} | {acc} | CDS")
    salvar_fasta(seq_dir / f"{base}_NCDS.fasta", ncds,
                 f"{vcv_id} | {info['gene_simbolo']} | {acc} | NCDS")

    r["status"] = "ok"
    return r


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    gb_dir  = OUTPUT_DIR / "genbank_files";  gb_dir.mkdir(exist_ok=True)
    seq_dir = OUTPUT_DIR / "sequences";      seq_dir.mkdir(exist_ok=True)

    resultados = []
    for vcv_id in VCV_IDS:
        print(f"\n{'='*60}\n  {vcv_id}\n{'='*60}")
        try:
            res = processar_vcv(vcv_id, gb_dir, seq_dir)
        except Exception as e:
            print(f"  ERRO: {e}")
            res = {k: "" for k in ["vcv_id","variante_nome","gene_simbolo",
                   "accession_usado","comprimento_total","num_features_cds",
                   "comprimento_cds","comprimento_ncds"]}
            res["vcv_id"]  = vcv_id
            res["status"]  = f"erro: {e}"
        resultados.append(res)

    csv_path = OUTPUT_DIR / "resumo.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(resultados[0].keys()))
        writer.writeheader()
        writer.writerows(resultados)

    print(f"\n{'='*75}\nRESUMO\n{'='*75}")
    print(f"{'VCV ID':<26} {'Gene':<8} {'Accession':<16} "
          f"{'Total':>9} {'CDS':>9} {'NCDS':>9}  Status")
    print("-" * 90)
    for res in resultados:
        print(f"{res['vcv_id']:<26} {res['gene_simbolo']:<8} "
              f"{res['accession_usado']:<16} "
              f"{str(res['comprimento_total']):>9} "
              f"{str(res['comprimento_cds']):>9} "
              f"{str(res['comprimento_ncds']):>9}  {res['status']}")
    print(f"\n→ Arquivos: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
