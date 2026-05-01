from Bio import Entrez, SeqIO
import time

Entrez.email = "seu@email.com"

problematicos = {
    "NM_015407.4": "25864",
    "AF003934.1":  None,
    "NM_021243.2": "58527",
    "AF335477.1":  None,
    "NM_024684.2": "28971",
    "NM_030754.4": "6289",
    "NM_032044.3": "83998",
}

def get_gene_id_from_acc(acc):
    """Para AF_ sem anotação: tenta via esummary do nuccore."""
    handle = Entrez.esearch(db="nuccore", term=acc)
    rec = Entrez.read(handle)
    handle.close()
    if not rec["IdList"]:
        return None, None
    uid = rec["IdList"][0]

    handle = Entrez.esummary(db="nuccore", id=uid)
    summary = Entrez.read(handle)
    handle.close()
    print(f"  esummary keys: {list(summary[0].keys())}")
    # Tenta extrair gene name do Title
    title = summary[0].get("Title", "")
    print(f"  Title: {title}")
    return None, title

def get_coords_xml(gene_id):
    """Busca coordenadas genômicas via XML do NCBI Gene."""
    handle = Entrez.efetch(db="gene", id=gene_id, rettype="docsum", retmode="xml")
    from xml.etree import ElementTree as ET
    tree = ET.parse(handle)
    handle.close()
    root = tree.getroot()

    # Procura por NC_ nas coordenadas
    nc_acc, start, end, strand = None, None, None, None

    for elem in root.iter():
        if elem.tag == "Seq-interval_from":
            start = int(elem.text)
        elif elem.tag == "Seq-interval_to":
            end = int(elem.text)
        elif elem.tag == "Na-strand" and elem.attrib.get("value"):
            strand = elem.attrib["value"]

    # Busca o NC_ via elink gene → nuccore chromosome
    handle = Entrez.elink(dbfrom="gene", db="nuccore", id=gene_id, linkname="gene_nuccore_refseq")
    link_rec = Entrez.read(handle)
    handle.close()

    nc_ids = []
    for linkset in link_rec:
        for linksetdb in linkset.get("LinkSetDb", []):
            for link in linksetdb["Link"]:
                nc_ids.append(link["Id"])

    if nc_ids:
        handle = Entrez.efetch(db="nuccore", id=nc_ids, rettype="acc", retmode="text")
        accs = handle.read().strip().split("\n")
        handle.close()
        nc_list = [a for a in accs if a.startswith("NC_")]
        if nc_list:
            nc_acc = nc_list[0]

    return nc_acc, start, end, strand

def get_coords_esummary(gene_id):
    """Alternativa: pega NC_ e coords via esummary XML do Gene."""
    handle = Entrez.esummary(db="gene", id=gene_id, retmode="xml")
    from xml.etree import ElementTree as ET
    content = handle.read()
    handle.close()

    root = ET.fromstring(content)

    nc_acc = None
    start  = None
    end    = None
    strand = None

    for ds in root.iter("DocumentSummary"):
        # Localização genômica
        for loc in ds.iter("GenomicInfoType"):
            chrom_acc = loc.findtext("ChrAccVer")
            chrom_start = loc.findtext("ChrStart")
            chrom_stop  = loc.findtext("ChrStop")
            chrom_strand = loc.findtext("Strand") 
            if chrom_acc and chrom_acc.startswith("NC_"):
                nc_acc = chrom_acc
                start  = int(chrom_start)
                end    = int(chrom_stop)
                strand = chrom_strand
                break

    return nc_acc, start, end, strand

def fetch_nc_region(nc_acc, start, end, flank=5000):
    seq_start = max(0, start - flank)
    seq_end   = end + flank
    handle = Entrez.efetch(
        db="nuccore", id=nc_acc, rettype="fasta", retmode="text",
        seq_start=seq_start+1, seq_end=seq_end  # NCBI usa 1-based aqui
    )
    fasta = handle.read()
    handle.close()
    return fasta, seq_start, seq_end

results = {}

for acc, gene_id in problematicos.items():
    print(f"\nProcessando {acc} (GeneID: {gene_id})...")
    try:
        if not gene_id:
            gene_id, title = get_gene_id_from_acc(acc)
            if not gene_id:
                results[acc] = f"GeneID não encontrado (title: {title})"
                time.sleep(0.4)
                continue

        # Tenta esummary primeiro (mais confiável para coordenadas)
        nc_acc, start, end, strand = get_coords_esummary(gene_id)
        print(f"  esummary → NC: {nc_acc}, start: {start}, end: {end}, strand: {strand}")

        if not nc_acc:
            results[acc] = f"NC_ não encontrado (GeneID: {gene_id})"
            time.sleep(0.4)
            continue

        fasta, seq_start, seq_end = fetch_nc_region(nc_acc, start, end, flank=5000)

        filename = f"{acc.replace('.', '_')}_genomic.fasta"
        with open(filename, "w") as f:
            lines = fasta.strip().split("\n")
            header = f">{acc}_GeneID{gene_id}_{nc_acc}_{seq_start+1}-{seq_end}_({strand})"
            f.write(header + "\n" + "\n".join(lines[1:]) + "\n")

        results[acc] = f"Salvo em {filename} | {nc_acc}:{seq_start+1}-{seq_end} ({strand})"

    except Exception as e:
        import traceback
        results[acc] = f"Erro: {e}"
        traceback.print_exc()

    time.sleep(0.5)

print("\n=== Resultados Finais ===")
for acc, res in results.items():
    print(f"{acc} → {res}")