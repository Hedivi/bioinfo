from Bio import Entrez
import time

# Email para acessar o NCBI GenBank
Entrez.email = "heloisa.viotto@ufpr.br"

# Baixa o arquivo GenBank em OUTPUT_FILE de acordo com ACCESSION
def download_Genbank (accession, output_file):

    # Define o arquivo que deve ser baixado
    handle = Entrez.efetch(
        db="nucleotide",
        id=accession,
        rettype="gb",
        retmode="text"
    )

    # Baixa o arquivo
    with open (output_file, "w") as f:
        f.write(handle.read())

    # Fecha a conexão
    handle.close()

if __name__ == "__main__":

    # Define uma lista de accesion de exemplo
    accession = ["L78833"]

    # Para cada accesion da lista
    for acc in accession:
        
        # Define o tipo, a pasta em que deve ser baixada
        type = "cancer" # or normal

        # Define o caminho de download
        output_file = "data/" + type + "/" + acc + ".gb" 
        
        # Baixa o arquivo
        download_Genbank(acc, output_file)

        # Tempo para não realizar múltiplas conexões seguidas
        time.sleep(0.5)