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
    """Recorta região do NC_ com flancos."""
    seq_start = max(0, start - flank)
    seq_end   = end + flank
    handle = Entrez.efetch(
        db="nuccore", id=nc_acc, rettype="fasta", retmode="text",
        seq_start=seq_start + 1, seq_end=seq_end
    )
    fasta = handle.read()
    handle.close()
    return fasta, seq_start, seq_end

def save_fasta(acc, gene_id, source_label, fasta_text, seq_start, seq_end):
    filename = f"{acc.replace('.', '_')}_genomic.fasta"
    lines = fasta_text.strip().split("\n")
    header = f">{acc}_GeneID{gene_id}_{source_label}_{seq_start+1}-{seq_end}"
    with open(filename, "w") as f:
        f.write(header + "\n" + "\n".join(lines[1:]) + "\n")
    return filename

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
            handle = Entrez.efetch(db="nuccore", id=ng, rettype="fasta", retmode="text")
            fasta = handle.read()
            handle.close()
            filename = f"{acc.replace('.', '_')}_genomic.fasta"
            lines = fasta.strip().split("\n")
            header = f">{acc}_GeneID{gene_id}_{ng}"
            with open(filename, "w") as f:
                f.write(header + "\n" + "\n".join(lines[1:]) + "\n")
            results[acc] = f"NG_ | {ng} → {filename}"

        else:
            # Fallback: recortar NC_
            nc_acc, start, end, strand = get_coords_esummary(gene_id)
            if not nc_acc:
                results[acc] = f"NC_ não encontrado (GeneID: {gene_id})"
                time.sleep(0.4)
                continue

            fasta, seq_start, seq_end = fetch_nc_region(nc_acc, start, end, flank=5000)
            label = f"{nc_acc}_{seq_start+1}-{seq_end}_({strand})"
            filename = save_fasta(acc, gene_id, label, fasta, seq_start, seq_end)
            results[acc] = f"NC_ recortado | {nc_acc}:{seq_start+1}-{seq_end} ({strand}) → {filename}"

    except Exception as e:
        import traceback
        results[acc] = f"Erro: {e}"
        traceback.print_exc()

    time.sleep(0.5)

print("\n=== Resultados Finais ===")
for acc, res in results.items():
    print(f"{acc} → {res}")
