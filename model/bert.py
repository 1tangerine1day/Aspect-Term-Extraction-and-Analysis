from transformers import BertModel
import torch

class bert_ATE(torch.nn.Module):
    def __init__(self, pretrain_model):
        super(bert_ATE, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, tags_tensors, masks_tensors):
        bert_outputs,_ = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors)
        # print(bert_outputs.size())
        linear_outputs = self.linear(bert_outputs)
        # print(linear_outputs.size())

        if tags_tensors is not None:
            tags_tensors = tags_tensors.view(-1)
            linear_outputs = linear_outputs.view(-1,3)
            # print(linear_outputs.size())
            # print(tags_tensors.size())
            loss = self.loss_fn(linear_outputs, tags_tensors)
            return loss
        else:
            return linear_outputs


class bert_ABSA(torch.nn.Module):
    def __init__(self, pretrain_model):
        super(bert_ABSA, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, ids_tensors, lable_tensors, masks_tensors, segments_tensors):
        _, pooled_outputs = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, token_type_ids=segments_tensors)
        # print(bert_outputs.size())
        linear_outputs = self.linear(pooled_outputs)
        # print(linear_outputs.size())

        if lable_tensors is not None:
            # print(linear_outputs.size())
            # print(tags_tensors.size())
            loss = self.loss_fn(linear_outputs, lable_tensors)
            return loss
        else:
            return linear_outputs