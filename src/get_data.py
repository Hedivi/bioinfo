"""
    VCV ClinVar → GenBank → Extração CDS e NCDS
    ============================================
    Replicação do pipeline do artigo:
    "Cancer detection with various classification models:
    A comprehensive feature analysis using HMM to extract a nucleotide pattern"
    Kalal & Jha, Computational Biology and Chemistry, 2024

    Pipeline do código:
    -Busca informações sobre cada variante
    -Encontra o gene associado
    -Baixa a sequência de DNA no formato GenBank
    -Separa o DNA em duas partes: CDS e NCDS
    -Salva tudo organizado em arquivos
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

nome_input = Path(INPUT).stem

if nome_input.startswith("Nao_Mal_"):
    CLASSE = "nao_cancer"
    GENE = nome_input.replace("Nao_Mal_", "")
elif nome_input.startswith("Mal_"):
    CLASSE = "cancer"
    GENE = nome_input.replace("Mal_", "")
else:
    CLASSE = "desconhecido"
    GENE = nome_input

OUTPUT_DIR = Path("../Genbank") / CLASSE / GENE

# Le um arquivo informado como argumento do código python3. E armazena os códigos ID das variantes
VCV_IDS = []
with open(INPUT, "r") as f:
    for line in f:
        VCV_IDS.append(line.strip())




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
    print ("Extraindo informações usando o esummary...")
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
    print ("Extraindo informações usando o efetch XML...")
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
    print ("Extraindo informações usando o rest_clinvar...")
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


"""
Tenta buscar as informações do ClinVar por 3 estratégias diferentes para código númerico da Variante.
Se tiver Genes e Accessions retorna automaticamente.
"""
def buscar_info_clinvar(vcv_numero: str) -> dict:
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



# Percorre por todos os prefixos de accessions
def priorizar_refseq(ids: list) -> str | None:
    for pref in ["NG_", "NM_", "NR_", "NC_"]:
        for acc in ids:
            if acc.startswith(pref):
                return acc
    return ids[0] if ids else None


# Procura outra forma de conseguir o accession que comece com NG
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


# Procura outra forma de conseguir o accession que comece com NG
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


"""
Tenta 3 maneiras de conseguir o accession, caso ele não consiga código do acession da variante, 
ele tenta por outros meios como pelo ID do gene...
"""
def resolver_accession(info: dict) -> str | None:
    acc = priorizar_refseq(info["refseq_ids"])
    if acc and acc.startswith("NG_"):
        return acc
    # Tentar obter NG_ de outras formas
    ng = (buscar_ng_via_elink(info["gene_id"])
          or buscar_ng_via_esearch(info["gene_simbolo"]))
    return ng or acc   # fallback para o que tiver


# Realiza o download do accession no formato GenBank
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



# Captura as regiões de introns e excns (Regiões codificantes e não codificantes)
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


# Responsavel por todo o pipeline
def processar_vcv(vcv_id: str, gb_dir: Path, seq_dir: Path) -> dict:
    
    # Cria o dicionário de resultado com todos os campos já inicializados
    # como vazios ou zero. Isso garante que mesmo se der erro no meio do
    # caminho, o dicionário terá todos os campos esperados para o CSV final.
    r = {"vcv_id": vcv_id, "variante_nome": "", "gene_simbolo": "",
         "accession_usado": "", "comprimento_total": 0,
         "num_features_cds": 0, "comprimento_cds": 0,
         "comprimento_ncds": 0, "status": "pendente"}

    # Converte o VCV ID do formato "VCV001239650.4" para o número limpo
    # "1239650", que é o formato aceito pelas APIs do NCBI.
    vcv_num = limpeza_vcv(vcv_id)

    # ETAPA 1: busca informações sobre a variante no ClinVar.
    # A função tenta 3 estratégias diferentes (esummary, efetch XML e REST API)
    # e retorna a primeira que trouxer dados úteis como gene e accessions.
    print(f"  [1] Info ClinVar...")
    info = buscar_info_clinvar(vcv_num)
    
    # Salva o nome da variante e o símbolo do gene no dicionário de resultado.
    r["variante_nome"] = info["variante_nome"]
    r["gene_simbolo"]  = info["gene_simbolo"]
    print(f"      Gene: {info['gene_simbolo']} | {info['variante_nome']}")
    print(f"      RefSeqs: {info['refseq_ids']}")

    # ETAPA 2: decide qual accession GenBank será usado para baixar a sequência.
    # O script prioriza NG_ (gene completo), mas se não encontrar tenta via
    # elink e esearch. Se nada funcionar, usa qualquer accession disponível.
    print(f"  [2] Resolvendo accession...")
    acc = resolver_accession(info)
    
    # Se não foi encontrado nenhum accession por nenhuma das estratégias,
    # registra o erro no dicionário e encerra o processamento desse VCV.
    if not acc:
        r["status"] = "erro: sem accession"
        return r
    
    r["accession_usado"] = acc
    print(f"      → {acc}")

    # ETAPA 3: baixa o arquivo GenBank do accession encontrado.
    # Os pontos no VCV ID e no accession são trocados por underscores
    # para evitar problemas no nome do arquivo em diferentes sistemas.
    # Se o arquivo já tiver sido baixado antes, usa o cache local.
    print(f"  [3] GenBank...")
    nome_gb = f"{vcv_id.replace('.','_')}_{acc.replace('.','_')}.gb"
    record  = baixar_genbank(acc, gb_dir / nome_gb)
    
    # Registra o tamanho total da sequência em pares de base (bp).
    r["comprimento_total"] = len(record.seq)
    print(f"      {len(record.seq):,} bp")

    # ETAPA 4: percorre as anotações do arquivo GenBank para identificar
    # quais posições do DNA pertencem a regiões CDS (codificantes).
    # O restante das posições é classificado como NCDS (não-codificante).
    print(f"  [4] Extraindo CDS/NCDS...")
    cds, ncds, n = extrair_cds_ncds(record)
    
    # Atualiza o dicionário de resultado com as estatísticas da extração:
    # quantidade de features CDS encontradas e tamanho em nucleotídeos
    # de cada região.
    r.update({"num_features_cds": n, "comprimento_cds": len(cds),
               "comprimento_ncds": len(ncds)})
    print(f"      CDS: {len(cds):,} nt | NCDS: {len(ncds):,} nt | features: {n}")

    # Monta o nome base dos arquivos FASTA usando o VCV ID e o símbolo do gene.
    # Salva dois arquivos: um com a sequência CDS e outro com a sequência NCDS.
    # O cabeçalho de cada FASTA identifica a origem: VCV, gene e accession usado.
    base = f"{vcv_id.replace('.','_')}_{info['gene_simbolo']}"
    salvar_fasta(seq_dir / f"{base}_CDS.fasta",  cds,
                 f"{vcv_id} | {info['gene_simbolo']} | {acc} | CDS")
    salvar_fasta(seq_dir / f"{base}_NCDS.fasta", ncds,
                 f"{vcv_id} | {info['gene_simbolo']} | {acc} | NCDS")

    # Marca o processamento como bem-sucedido e retorna o dicionário completo
    # com todas as informações coletadas para esse VCV ID.
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
