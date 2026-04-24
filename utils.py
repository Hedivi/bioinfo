from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
from Bio import SeqIO


def extract_cds_ncds (file):

    record = SeqIO.read(file, "genbank")
    seq = str(record.seq).upper()
    
    cds_seq = ""
    cds_pos = []
    for feature in record.features:
        if feature.type == "CDS":
            start = feature.location.start
            end = feature.location.end
            cds_seq += seq[start:end]
            cds_pos.append((start, end))

    mask = [True] * len(seq)
    for start, end in cds_pos:
        for i in range(start, end):
            mask[i] = False

    ncds_seq = "".join([seq[i] for i in range(len(seq)) if mask[i]])

    return cds_seq, ncds_seq

def generate_score (scores, ref, pred):

    acc = accuracy_score (ref, pred)
    f1 = f1_score(ref, pred)
    spe = recall_score(ref, pred, pos_label=0)
    rec = recall_score(ref, pred)
    prec = precision_score(ref, pred)
    roc = roc_curve(ref, pred)

    scores["accuracy"].append(acc)
    scores["f1-score"].append(f1)
    scores["specifity"].append(spe)
    scores["recall"].append(rec)
    scores["precision"].append(prec)
    scores["roc"].append(roc)
   