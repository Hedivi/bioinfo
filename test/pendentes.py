from Bio import Entrez, SeqIO
import time

Entrez.email = "seu@email.com"

# AF003934.1 e AF335477.1 são KLK3 (PSA), GeneID 354
# já confirmado pelos AJ_ anteriores que retornaram NG_011653.1
pendentes = {
    "AF003934.1": "354",
    "AF335477.1": "354",
}

def get_coords_esummary(gene_id):
    from xml.etree import ElementTree as ET
    handle = Entrez.esummary(db="gene", id=gene_id, retmode="xml")
    content = handle.read()
    handle.close()

    root = ET.fromstring(content)
    for ds in root.iter("DocumentSummary"):
        for loc in ds.iter("GenomicInfoType"):
            chrom_acc    = loc.findtext("ChrAccVer")
            chrom_start  = loc.findtext("ChrStart")
            chrom_stop   = loc.findtext("ChrStop")
            chrom_strand = loc.findtext("ExonCount")  # debug: ver todos os campos
            
            # Imprime todos os filhos para ver o que está disponível
            print(f"  Campos GenomicInfoType: { {c.tag: c.text for c in loc} }")
            
            if chrom_acc and chrom_acc.startswith("NC_"):
                start  = int(chrom_start)
                end    = int(chrom_stop)
                # Strand: start > end significa minus
                strand = "-" if start > end else "+"
                return chrom_acc, min(start,end), max(start,end), strand
    return None, None, None, None

def fetch_nc_region(nc_acc, start, end, flank=5000):
    seq_start = max(0, start - flank)
    seq_end   = end + flank
    handle = Entrez.efetch(
        db="nuccore", id=nc_acc, rettype="fasta", retmode="text",
        seq_start=seq_start+1, seq_end=seq_end
    )
    fasta = handle.read()
    handle.close()
    return fasta, seq_start, seq_end

for acc, gene_id in pendentes.items():
    print(f"\nProcessando {acc} (GeneID: {gene_id})...")
    try:
        nc_acc, start, end, strand = get_coords_esummary(gene_id)
        print(f"  → NC: {nc_acc}, {start}-{end}, strand: {strand}")

        fasta, seq_start, seq_end = fetch_nc_region(nc_acc, start, end, flank=5000)

        filename = f"{acc.replace('.', '_')}_genomic.fasta"
        with open(filename, "w") as f:
            lines = fasta.strip().split("\n")
            header = f">{acc}_GeneID{gene_id}_{nc_acc}_{seq_start+1}-{seq_end}_({strand})"
            f.write(header + "\n" + "\n".join(lines[1:]) + "\n")

        print(f"  Salvo: {filename}")

    except Exception as e:
        import traceback
        print(f"  Erro: {e}")
        traceback.print_exc()

    time.sleep(0.5)
