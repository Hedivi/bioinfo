from Bio import Entrez
from pathlib import Path
import time


# Email para acessar o NCBI GenBank
Entrez.email = "heloisa.viotto@ufpr.br"

ACCESSION_DIR = Path("../Data/Accessions")
GENBANK_DIR = Path("../Data/Genbank")


# Baixa o arquivo GenBank em OUTPUT_FILE de acordo com código ACCESSION
def download_Genbank(accession, output_file):

    # Define o arquivo que deve ser baixado
    handle = Entrez.efetch(
        db="nucleotide",
        id=accession,
        rettype="gb",
        retmode="text"
    )

    # Baixa o arquivo
    with open(output_file, "w") as f:
        f.write(handle.read())

    # Fecha a conexão
    handle.close()


"""
    Essa função é responsável por fazer a leitura de um arquivo com todos os Accessions
    de um determinado tipo de câncer e retornar os códigos de cada sequência.
"""
def read_accession_file(file):

    list_accession = []

    # Lê o arquivo linha por linha
    with open(file, "r") as f:
        for row in f:
            accession = row.strip()

            # Ignora linhas vazias e comentários
            if not accession or accession.startswith("#"):
                continue

            list_accession.append(accession)

    return list_accession


if __name__ == "__main__":

    # Cria a pasta principal de saída, caso ela não exista
    GENBANK_DIR.mkdir(parents=True, exist_ok=True)

    """
        Leitura de todos os arquivos da pasta Data/Accessions.
        Cada arquivo representa uma lista de accessions de uma classe.
        Ex: Mal_BRCA1, Nao_Mal_BRCA1, Mal_CDKN2A, Nao_Mal_CDKN2A.
    """
    for accession_file in ACCESSION_DIR.glob("*.txt"):

        # O nome da pasta de saída será o mesmo nome do arquivo txt
        type_name = accession_file.stem

        output_class_dir = GENBANK_DIR / type_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        accessions = read_accession_file(accession_file)

        print(f"\nArquivo: {accession_file}")
        print(f"Classe: {type_name}")
        print(f"Total de accessions: {len(accessions)}")

        """
            Extração de cada código presente no arquivo de accession.
            Cada código será usado para realizar o download da sequência no formato GenBank.
        """
        for accession in accessions:

            output_file = output_class_dir / f"{accession}.gb"

            # Evita baixar novamente caso o arquivo já exista
            if output_file.exists():
                print(f"SKIP - Arquivo já existe: {output_file}")
                continue

            try:
                print(f"DOWNLOAD - Baixando {accession}")

                download_Genbank(accession, output_file)

                print(f"OK - Salvo em: {output_file}")

                # Tempo para não realizar múltiplas conexões seguidas
                time.sleep(0.5)

            except Exception as e:
                print(f"ERRO - Não foi possível realizar o download de {accession}: {e}")