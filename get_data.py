from Bio import Entrez
import time

Entrez.email = "heloisa.viotto@ufpr.br"

def download_Genbank (acession, output_file):

    handle = Entrez.efetch(
        db="nucleotide",
        id=acession,
        rettype="gb",
        retmode="text"
    )

    with open (output_file, "w") as f:
        f.write(handle.read())

    handle.close()

if __name__ == "__main__":

    accession = ["L78833"]

    for acc in accession:
        
        type = "cancer" # or normal

        output_file = "data/" + type + "/" + acc + ".gb" 
        download_Genbank(acc, output_file)
        time.sleep(0.5)