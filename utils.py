import random
import logging

import numpy as np
import torch
from sklearn.metrics import f1_score

from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    BertConfig,
    BertTokenizer
)

from model import (
    ElectraForBiasClassification,
    BertForBiasClassification
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "koelectra-base": (ElectraConfig, ElectraForBiasClassification, ElectraTokenizer),
    "koelectra-small": (ElectraConfig, ElectraForBiasClassification, ElectraTokenizer),
    "koelectra-base-v2": (ElectraConfig, ElectraForBiasClassification, ElectraTokenizer),
    "koelectra-small-v2": (ElectraConfig, ElectraForBiasClassification, ElectraTokenizer),
    "kcbert-base": (BertConfig, BertForBiasClassification, BertTokenizer),
}


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(pred_bias_labels, pred_hate_labels, gt_bias_labels, gt_hate_labels):
    bias_weighted_f1 = f1_score(gt_bias_labels, pred_bias_labels, average="weighted")
    hate_weighted_f1 = f1_score(gt_hate_labels, pred_hate_labels, average="weighted")

    bias_macro_f1 = f1_score(gt_bias_labels, pred_bias_labels, average="macro")
    hate_macro_f1 = f1_score(gt_hate_labels, pred_hate_labels, average="macro")

    mean_weighted_f1 = (bias_weighted_f1 + hate_weighted_f1) / 2
    return {
        "bias_weighted_f1": bias_weighted_f1,
        "hate_weighted_f1": hate_weighted_f1,
        "mean_weighted_f1": mean_weighted_f1,
        "bias_macro_f1": bias_macro_f1,
        "hate_macro_f1": hate_macro_f1
    }
