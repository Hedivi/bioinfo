from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
from Bio import SeqIO

# Extrai as sequência codificantes e não condificantes de um arquivo
def extract_cds_ncds (file):

    # Abre o arquivo
    record = SeqIO.read(file, "genbank")

    # Seleciona a sequência e coloca em maiúscula
    seq = str(record.seq).upper()
    

    # Incializa as sequência codificante (string vazia) e uma lista de posições
    cds_seq = ""
    cds_pos = []

    # Para cada feature no arquivo
    for feature in record.features:

        # Se a feature for do tipo CDS (codificante)
        if feature.type == "CDS":

            # Define o começo e o fim da sequência
            start = feature.location.start
            end = feature.location.end

            # Concatena o pedaço da sequência codificante 
            cds_seq += seq[start:end]

            # Adiciona a tupla de posições inicial e final
            cds_pos.append((start, end))

    # Define uma máscara do tamanho da sequência com valores verdadeiros
    mask = [True] * len(seq)

    # Para cada posição de início e fim das regiões codificantes
    for start, end in cds_pos:

        # Para cada posição entre start e end
        for i in range(start, end):
            # Muda a máscara na posição i para falso
            mask[i] = False

    # Junta a sequência restante. Se a máscara for positiva, ou seja não está na rgeião codificante, adiciona na sequência não codificante
    ncds_seq = "".join([seq[i] for i in range(len(seq)) if mask[i]])

    # Retorna as sequências codificantes e não condificantes
    return cds_seq, ncds_seq

# Calcula as métricas dado uma sequência de valores de referência e um predito
def generate_score (scores, ref, pred):

    # Calcula a acurácia
    acc = accuracy_score (ref, pred)

    # Calcula o F1-score
    f1 = f1_score(ref, pred)

    # Calcula a especificidade
    spe = recall_score(ref, pred, pos_label=0)

    # Calcula o recall
    rec = recall_score(ref, pred)

    # Calcula a precisão
    prec = precision_score(ref, pred)

    # Calcula a curva ROC
    roc = roc_curve(ref, pred)

    # Adiciona os valores em suas respectivas listas
    scores["accuracy"].append(acc)
    scores["f1-score"].append(f1)
    scores["specifity"].append(spe)
    scores["recall"].append(rec)
    scores["precision"].append(prec)
    scores["roc"].append(roc)
   