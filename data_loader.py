import os
import re
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset
from soynlp.normalizer import emoticon_normalize, repeat_normalize

logger = logging.getLogger(__name__)


def preprocess(title: str,
               comment: str):
    # Erase redundant \" in the start & end of the title
    if title.startswith("\""):
        title = title[1:]
    if title.endswith("\""):
        title = title[:-1]

    # Change quotes
    title = title.replace("“", "\"").replace("”", "\"").replace("‘", "\'").replace("’", "\'")

    # Erase braces in title
    braces = r"\[(.*?)\]"
    braces2 = r"\{(.*?)\}"
    braces3 = r"\【(.*?)\】"
    braces4 = r"\<(.*?)\>"

    title = re.sub(braces, '', title)
    title = re.sub(braces2, '', title)
    title = re.sub(braces3, '', title)
    title = re.sub(braces4, '', title)

    # Normalize the comment
    comment = emoticon_normalize(comment, num_repeats=3)
    comment = repeat_normalize(comment, num_repeats=3)

    return title, comment


class InputExample(object):
    """ A single training/test example for simple sequence classification. """

    def __init__(self,
                 guid,
                 text_a,
                 text_b,
                 bias_label,
                 hate_label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.bias_label = bias_label
        self.hate_label = hate_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 bias_label=None,
                 hate_label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.bias_label = bias_label
        self.hate_label = hate_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class KoreanHateSpeechProcessor(object):
    """Processor for the Korean Hate Speech data set """

    def __init__(self, args):
        self.args = args

    @classmethod
    def get_labels(cls):
        bias_label_lst = ['none', 'gender', 'others']
        hate_label_lst = ['none', 'hate', 'offensive']
        return bias_label_lst, hate_label_lst

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the train, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):  # Except the header
            line = line.split('\t')
            guid = "%s-%s" % (set_type, i)
            title = line[0]
            comment = line[1]
            title, comment = preprocess(title, comment)

            bias_label = None
            hate_label = None
            if set_type != "test":
                bias_label = line[2]
                hate_label = line[3]
            if i % 1000 == 0:
                logger.info([title, comment])

            examples.append(InputExample(guid=guid,
                                         text_a=comment,
                                         text_b=title,
                                         bias_label=bias_label,
                                         hate_label=hate_label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir, file_to_read)), mode)


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length,
):
    bias_label_list, hate_label_list = KoreanHateSpeechProcessor.get_labels()

    bias_label_map = {label: i for i, label in enumerate(bias_label_list)}
    hate_label_map = {label: i for i, label in enumerate(hate_label_list)}

    def label_from_example(example):
        bias_label_id = -1
        hate_label_id = -1
        if example.bias_label is not None:
            bias_label_id = bias_label_map[example.bias_label]
        if example.hate_label is not None:
            hate_label_id = hate_label_map[example.hate_label]
        return bias_label_id, hate_label_id

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        if "token_type_ids" not in inputs:
            inputs["token_type_ids"] = [0] * len(inputs["input_ids"])  # For xlm-roberta, distilkobert

        feature = InputFeatures(**inputs, bias_label=labels[i][0], hate_label=labels[i][1])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: {}".format(example.guid))
        logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
        logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
        logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
        logger.info("bias_label: {}".format(features[i].bias_label))
        logger.info("hate_label: {}".format(features[i].hate_label))

    return features


def load_examples(args, tokenizer, mode):
    processor = KoreanHateSpeechProcessor(args)

    logger.info("Creating features from dataset file at %s", args.data_dir)
    if mode == "train":
        examples = processor.get_examples("train")
    elif mode == "dev":
        examples = processor.get_examples("dev")
    elif mode == "test":
        examples = processor.get_examples("test")
    else:
        raise Exception("For mode, Only train, dev, test is available")

    features = convert_examples_to_features(
        examples,
        tokenizer,
        args.max_seq_len
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_bias_labels = torch.tensor([f.bias_label for f in features], dtype=torch.long)
    all_hate_labels = torch.tensor([f.hate_label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids,
                            all_attention_mask,
                            all_token_type_ids,
                            all_bias_labels,
                            all_hate_labels)
    return dataset
