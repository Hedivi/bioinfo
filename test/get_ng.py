from Bio import Entrez, SeqIO
import time

Entrez.email = "heloisa.viotto@ufpr.br"

accessions = [
    "NM_015407.4", "AF003934.1", "NM_012278.2", "NM_021243.2",
    "AF335477.1", "NM_024533.4", "NM_024684.2", "AJ459782.1",
    "NM_030754.4", "NM_032548.3", "AJ459784.1", "NM_032044.3"
]

def get_gene_id_from_genbank(acc):
    """Extrai GeneID ou nome do gene do registro GenBank."""
    handle = Entrez.efetch(db="nuccore", id=acc, rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()

    gene_id = None
    gene_name = None

    for feature in record.features:
        for xref in feature.qualifiers.get("db_xref", []):
            if xref.startswith("GeneID:"):
                gene_id = xref.split(":")[1]
        if not gene_name:
            gene_name = feature.qualifiers.get("gene", [None])[0]

    # Para AF_/AJ_, tentar buscar GeneID pelo nome do gene + organismo
    if not gene_id and gene_name:
        organism = record.annotations.get("organism", "Homo sapiens")
        print(f"  {acc}: sem GeneID direto, buscando por gene '{gene_name}' em '{organism}'")
        handle2 = Entrez.esearch(db="gene", term=f"{gene_name}[gene] AND {organism}[orgn] AND 9606[taxid]")
        rec2 = Entrez.read(handle2)
        handle2.close()
        if rec2["IdList"]:
            gene_id = rec2["IdList"][0]

    return gene_id, gene_name

def get_ng_from_gene_id(gene_id):
    """Tenta múltiplos linknames para achar NG_."""
    linknames = [
        "gene_nuccore_refseqgene",
        "gene_nuccore",
        "gene_nuccore_refseq",
    ]

    for linkname in linknames:
        handle = Entrez.elink(dbfrom="gene", db="nuccore", id=gene_id, linkname=linkname)
        link_record = Entrez.read(handle)
        handle.close()

        ng_ids = []
        for linkset in link_record:
            for linksetdb in linkset.get("LinkSetDb", []):
                for link in linksetdb["Link"]:
                    ng_ids.append(link["Id"])

        if ng_ids:
            handle = Entrez.efetch(db="nuccore", id=ng_ids, rettype="acc", retmode="text")
            ng_accs = handle.read().strip().split("\n")
            handle.close()
            ng_filtered = [a for a in ng_accs if a.startswith("NG_")]
            if ng_filtered:
                return ng_filtered

        time.sleep(0.3)

    # Última tentativa: buscar direto por GeneID no nuccore
    handle = Entrez.esearch(db="nuccore", term=f"{gene_id}[gene] AND NG_[accn]")
    rec = Entrez.read(handle)
    handle.close()
    if rec["IdList"]:
        handle = Entrez.efetch(db="nuccore", id=rec["IdList"], rettype="acc", retmode="text")
        ng_accs = handle.read().strip().split("\n")
        handle.close()
        ng_filtered = [a for a in ng_accs if a.startswith("NG_")]
        if ng_filtered:
            return ng_filtered

    return None

results = {}

for acc in accessions:
    print(f"Processando {acc}...")
    try:
        gene_id, gene_name = get_gene_id_from_genbank(acc)

        if not gene_id:
            results[acc] = f"GeneID não encontrado (gene: {gene_name})"
            time.sleep(0.4)
            continue

        print(f"  GeneID: {gene_id}, gene: {gene_name}")

        ng = get_ng_from_gene_id(gene_id)
        results[acc] = ", ".join(ng) if ng else f"NG_ não encontrado (GeneID: {gene_id})"

    except Exception as e:
        results[acc] = f"Erro: {e}"

    time.sleep(0.4)

print("\n=== Resultados Finais ===")
for acc, ng in results.items():
    print(f"{acc} → {ng}")