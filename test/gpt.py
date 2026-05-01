from Bio import Entrez, SeqIO
from xml.etree import ElementTree as ET
import time

Entrez.email = "heloisa.viotto@urfpr.br"

accessions = [
  "NM_015407.4", "NM_021243.2", "NM_024684.2", "NM_032548.3", "NM_148912.2", 
  "NM_015423.2", "NM_024666.4", "NM_130786.3", "NM_138340.4", "NM_013375.3", 
  "NM_001163993.2", "NM_001256847.2", "NM_01163942.1", "NM_178559.5", "NM_001014423.2", 
  "NM_001014422.2", "NM_001271886.1", "NM_015429.3", "NM_020745.3", "NM_001206929.1",
  "NM_017436.4", "NM_018713.2", "NM_001105208.2", "NM_001206484.2", "NM_007003.4", 
  "NM_001287387.1", "NM_005984.4", "NM_001256534.1", "NM_012278.2", "NM_024533.4", 
  "NM_030754.4", "NM_032044.3", "NM_138937.2", "NM_001127380.2", "NM_001159353.1", 
  "NM_001177515.1", "NM_153698.1", "NM_032859.2", "NM_001605.2", "NM_004924.5", 
  "NM_012138.3", "NM_001206673.1", "NM_001024675.1", "NM_001159352.1"
]

# ── Funções ────────────────────────────────────────────────────────────────────

def efetch_with_retry(db, id, rettype, retmode, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            handle = Entrez.efetch(
                db=db,
                id=id,
                rettype=rettype,
                retmode=retmode,
                **kwargs
            )
            return handle

        except Exception as e:
            print(f"Erro efetch (tentativa {attempt+1}): {e}")
            time.sleep(1 + attempt)

    raise RuntimeError("Falha após múltiplas tentativas")

def get_gene_id_from_genbank(acc):
    """Extrai GeneID e nome do gene do registro GenBank."""
    handle = Entrez.efetch(db="nuccore", id=acc, rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()

    gene_id, gene_name = None, None
    for feature in record.features:
        for xref in feature.qualifiers.get("db_xref", []):
            if xref.startswith("GeneID:"):
                gene_id = xref.split(":")[1]
        if not gene_name:
            gene_name = feature.qualifiers.get("gene", [None])[0]

    # AF_/AJ_ sem GeneID: busca pelo nome do gene
    if not gene_id and gene_name:
        organism = record.annotations.get("organism", "Homo sapiens")
        handle = Entrez.esearch(db="gene", term=f"{gene_name}[gene] AND 9606[taxid]")
        rec = Entrez.read(handle)
        handle.close()
        if rec["IdList"]:
            gene_id = rec["IdList"][0]

    return gene_id, gene_name

def get_ng_accession(gene_id):
    """Tenta obter NG_ via elink gene → nuccore refseqgene."""
    handle = Entrez.elink(dbfrom="gene", db="nuccore", id=gene_id,
                          linkname="gene_nuccore_refseqgene")
    link_rec = Entrez.read(handle)
    handle.close()

    ng_ids = [
        link["Id"]
        for linkset in link_rec
        for linksetdb in linkset.get("LinkSetDb", [])
        for link in linksetdb["Link"]
    ]
    if not ng_ids:
        return None

    handle = Entrez.efetch(db="nuccore", id=ng_ids, rettype="acc", retmode="text")
    accs = handle.read().strip().split("\n")
    handle.close()
    ng_list = [a for a in accs if a.startswith("NG_")]
    return ng_list[0] if ng_list else None

def get_coords_esummary(gene_id):
    """Busca coordenadas genômicas (NC_) via esummary XML do Gene."""
    handle = Entrez.esummary(db="gene", id=gene_id, retmode="xml")
    content = handle.read()
    handle.close()

    root = ET.fromstring(content)
    for ds in root.iter("DocumentSummary"):
        for loc in ds.iter("GenomicInfoType"):
            chrom_acc   = loc.findtext("ChrAccVer")
            chrom_start = loc.findtext("ChrStart")
            chrom_stop  = loc.findtext("ChrStop")
            if chrom_acc and chrom_acc.startswith("NC_"):
                start  = int(chrom_start)
                end    = int(chrom_stop)
                strand = "-" if start > end else "+"
                return chrom_acc, min(start, end), max(start, end), strand
    return None, None, None, None

def fetch_nc_region(nc_acc, start, end, flank=5000):
    seq_start = max(0, start - flank)
    seq_end   = end + flank

    # 🔹 GenBank (com retry)
    handle = efetch_with_retry(
        db="nuccore",
        id=nc_acc,
        rettype="gb",
        retmode="text",
        seq_start=seq_start + 1,
        seq_end=seq_end
    )
    record = SeqIO.read(handle, "genbank")
    handle.close()

    # 🔹 FASTA (garantir sequência)
    handle = efetch_with_retry(
        db="nuccore",
        id=nc_acc,
        rettype="fasta",
        retmode="text",
        seq_start=seq_start + 1,
        seq_end=seq_end
    )
    fasta_record = SeqIO.read(handle, "fasta")
    handle.close()

    record.seq = fasta_record.seq

    return record, seq_start, seq_end

def extrair_cds_ncds(record, gene_name):
    seq = str(record.seq).upper()
    pos_cds = set()

    for feat in record.features:
        if feat.type == "CDS":
            gene_feat = feat.qualifiers.get("gene", [""])[0]
            if gene_feat != gene_name:
                continue

            for parte in feat.location.parts:
                pos_cds.update(range(int(parte.start), int(parte.end)))

    if not pos_cds:
        return "", ""

    cds  = "".join(seq[i] for i in sorted(pos_cds))
    ncds = "".join(seq[i] for i in range(len(seq)) if i not in pos_cds)

    return cds, ncds


def salve_fasta(caminho, seq, header):
    with open(caminho, "w") as f:
        f.write(f">{header}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")

# ── Pipeline principal ─────────────────────────────────────────────────────────

results = {}

for acc in accessions:
    print(f"\nProcessando {acc}...")
    try:
        # 1. Obter GeneID
        gene_id, gene_name = get_gene_id_from_genbank(acc)
        if not gene_id:
            results[acc] = "GeneID não encontrado"
            time.sleep(0.4)
            continue
        print(f"  GeneID: {gene_id} ({gene_name})")

        # 2. Tentar NG_ primeiro
        ng = get_ng_accession(gene_id)
        time.sleep(0.3)

        if ng:
            # Busca FASTA do NG_ completo
            handle = Entrez.efetch(db="nuccore", id=ng, rettype="gb", retmode="text")
            record = SeqIO.read(handle, "genbank")
            handle.close()

            # salva genbank
            gb_filename = f"data/genbank/{acc.replace('.', '_')}.gb"
            SeqIO.write(record, gb_filename, "genbank")

            cds_seq, nao_cds_seq = extrair_cds_ncds(record, gene_name)

            # salva CDS
            salve_fasta(
                f"data/fasta/{acc.replace('.', '_')}_cds.fasta",
                f"{acc}_CDS_GeneID{gene_id}",
                cds_seq
            )

            # salva NÃO CDS
            salve_fasta(
                f"data/fasta/{acc.replace('.', '_')}_nao_cds.fasta",
                f"{acc}_NAO_CDS_GeneID{gene_id}",
                nao_cds_seq
            )

            results[acc] = f"OK → {gb_filename}, CDS, NÃO CDS"
        else:
            # Fallback: recortar NC_
            nc_acc, start, end, strand = get_coords_esummary(gene_id)
            if not nc_acc:
                results[acc] = f"NC_ não encontrado (GeneID: {gene_id})"
                time.sleep(0.4)
                continue

            record, seq_start, seq_end = fetch_nc_region(nc_acc, start, end, flank=5000)

            # salva GenBank
            gb_filename = f"data/genbank/{acc.replace('.', '_')}_region.gb"
            SeqIO.write(record, gb_filename, "genbank")

            # extrai CDS / NCDS
            cds_seq, nao_cds_seq = extrair_cds_ncds(record, gene_name)

            if not cds_seq:
                results[acc] = f"NC_ sem CDS válido ({gene_name})"
                continue

            # salva CDS
            salve_fasta(
                f"data/fasta/{acc.replace('.', '_')}_cds.fasta",
                cds_seq,
                f"{acc}_CDS_GeneID{gene_id}"
            )

            # salva NÃO CDS
            salve_fasta(
                f"data/fasta/{acc.replace('.', '_')}_nao_cds.fasta",
                nao_cds_seq,
                f"{acc}_NAO_CDS_GeneID{gene_id}"
            )

            results[acc] = (
                f"NC_ recortado | {nc_acc}:{seq_start+1}-{seq_end} ({strand}) → "
                f"{gb_filename}, CDS, NÃO CDS"
            )

    except Exception as e:
        import traceback
        results[acc] = f"Erro: {e}"
        traceback.print_exc()

    time.sleep(0.8)

print("\n=== Resultados Finais ===")
for acc, res in results.items():
    print(f"{acc} → {res}")

