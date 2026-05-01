"""
Utilitários de avaliação para classificação binária.
Implementa as métricas usadas no artigo:
    accuracy, f1-score, specificity, recall, precision, ROC AUC
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
)


def specificity_score(y_true, y_pred) -> float:
    """
    Especificidade = TN / (TN + FP)
    O sklearn não oferece isso diretamente — calculamos via confusion_matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    # confusion_matrix para binário retorna [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Fallback para casos degenerados (fold com só uma classe)
    return 0.0


def generate_score(score_dict: dict, y_true, y_pred, y_prob=None):
    """
    Calcula todas as métricas e acumula no dicionário de scores.

    Args:
        score_dict : dicionário com listas para cada métrica
        y_true     : rótulos reais
        y_pred     : rótulos preditos
        y_prob     : probabilidades preditas (necessário para ROC AUC)
                     Se None, ROC não é calculado.
    """
    score_dict["accuracy"].append(
        accuracy_score(y_true, y_pred)
    )
    score_dict["f1-score"].append(
        f1_score(y_true, y_pred, zero_division=0)
    )
    score_dict["recall"].append(
        recall_score(y_true, y_pred, zero_division=0)
    )
    score_dict["precision"].append(
        precision_score(y_true, y_pred, zero_division=0)
    )
    score_dict["specificity"].append(
        specificity_score(y_true, y_pred)
    )

    # ROC AUC só é calculado se probabilidades forem fornecidas
    if y_prob is not None:
        try:
            score_dict["roc"].append(
                roc_auc_score(y_true, y_prob)
            )
        except ValueError:
            # Ocorre quando o fold de teste tem só uma classe
            score_dict["roc"].append(float("nan"))
    else:
        score_dict["roc"].append(float("nan"))
