import torch.nn as nn
from transformers import (
    ElectraPreTrainedModel,
    ElectraModel,
    ElectraConfig,
    BertPreTrainedModel,
    BertModel,
    BertConfig
)
from argparse import Namespace


class BiasClassificationHead(nn.Module):
    """Head for Bias Classification"""

    def __init__(self, config, num_bias_labels):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_bias_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class HateClassificationHead(nn.Module):
    """Head for Hate Classification"""

    def __init__(self, config, num_hate_labels):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_hate_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class ElectraForBiasClassification(ElectraPreTrainedModel):
    def __init__(self,
                 config: ElectraConfig,
                 args: Namespace,
                 bias_label_lst=None,
                 hate_label_lst=None):
        super().__init__(config)
        self.args = args
        self.num_bias_labels = len(bias_label_lst) if bias_label_lst is not None else 0
        self.num_hate_labels = len(hate_label_lst) if hate_label_lst is not None else 0

        self.electra = ElectraModel(config)
        self.bias_classifier = BiasClassificationHead(config, self.num_bias_labels)
        self.hate_classifier = HateClassificationHead(config, self.num_hate_labels)

        self.loss_fct = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        bias_labels=None,
        hate_labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        pooled_output = discriminator_hidden_states[0][:, 0]

        bias_logits = self.bias_classifier(pooled_output)
        hate_logits = self.hate_classifier(pooled_output)

        total_loss = 0

        # 1. Bias
        if bias_labels is not None:
            bias_loss = self.loss_fct(bias_logits.view(-1, self.num_bias_labels), bias_labels.view(-1))
            total_loss += self.args.bias_loss_coef * bias_loss

        # 2. Hate
        if hate_labels is not None:
            hate_loss = self.loss_fct(hate_logits.view(-1, self.num_hate_labels), hate_labels.view(-1))
            total_loss += self.args.hate_loss_coef * hate_loss

        outputs = ((bias_logits, hate_logits),) + discriminator_hidden_states[1:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForBiasClassification(BertPreTrainedModel):
    def __init__(self,
                 config: BertConfig,
                 args: Namespace,
                 bias_label_lst=None,
                 hate_label_lst=None):
        super().__init__(config)
        self.args = args
        self.num_bias_labels = len(bias_label_lst) if bias_label_lst is not None else 0
        self.num_hate_labels = len(hate_label_lst) if hate_label_lst is not None else 0

        self.bert = BertModel(config)
        self.bias_classifier = BiasClassificationHead(config, self.num_bias_labels)
        self.hate_classifier = HateClassificationHead(config, self.num_hate_labels)

        self.loss_fct = nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        bias_labels=None,
        hate_labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        bias_logits = self.bias_classifier(pooled_output)
        hate_logits = self.hate_classifier(pooled_output)

        total_loss = 0

        # 1. Bias
        if bias_labels is not None:
            bias_loss = self.loss_fct(bias_logits.view(-1, self.num_bias_labels), bias_labels.view(-1))
            total_loss += self.args.bias_loss_coef * bias_loss

        # 2. Hate
        if hate_labels is not None:
            hate_loss = self.loss_fct(hate_logits.view(-1, self.num_hate_labels), hate_labels.view(-1))
            total_loss += self.args.hate_loss_coef * hate_loss

        outputs = ((bias_logits, hate_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
