# Standard Library Modules
import os
import sys
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
# Huggingface Modules
from transformers import AutoConfig, AutoModel
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import get_huggingface_model_name

class ClassificationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ClassificationModel, self).__init__()
        self.args = args

        if args.model_type in ['bert', 'distilbert', 'roberta', 'roberta_large', 'tinybert']:
            huggingface_model_name = get_huggingface_model_name(self.args.model_type)
            self.config = AutoConfig.from_pretrained(huggingface_model_name)
            if args.model_ispretrained:
                self.model = AutoModel.from_pretrained(huggingface_model_name)
            else:
                self.model = AutoModel.from_config(self.config)

            self.cls_size = self.config.hidden_size
            self.num_classes = self.args.num_classes
        elif args.model_type == 'cnn':
            each_out_size = args.embed_size // 3

            self.embed = nn.Embedding(args.vocab_size, args.embed_size)
            self.conv = nn.ModuleList(
                [nn.Conv1d(in_channels=args.embed_size, out_channels=each_out_size,
                           kernel_size=kernel_size, stride=1, padding='same', bias=False)
                           for kernel_size in [3, 4, 5]]
            )

            self.cls_size = each_out_size * 3
            self.num_classes = self.args.num_classes
        elif args.model_type == 'lstm':
            self.embed = nn.Embedding(args.vocab_size, args.embed_size)

            self.rnn = nn.LSTM(input_size=args.embed_size, hidden_size=args.hidden_size, num_layers=args.num_layers_rnn, batch_first=True, bidirectional=args.rnn_isbidirectional)

            self.cls_size = args.hidden_size * (2 if args.rnn_isbidirectional else 1) * args.num_layers_rnn
            self.num_classes = self.args.num_classes
        else:
            raise NotImplementedError(f"Model type {args.model_type} is not implemented.")

        self.projector = nn.Sequential(
            nn.Linear(self.cls_size, args.projection_size),
            nn.Dropout(args.dropout_rate),
            nn.Tanh(), # Following previous works: https://github.com/tonytan48/MSCL/blob/main/sentiment_analysis/mscl_domain.py#L312
            nn.LayerNorm(args.projection_size)
        )

        self.classifier = nn.Sequential(
            nn.Tanh(), # Following previous works: https://github.com/tonytan48/MSCL/blob/main/sentiment_analysis/mscl_domain.py#L704
            nn.Linear(args.projection_size, self.num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        if self.args.model_type in ['bert', 'distilbert', 'roberta', 'roberta_large', 'tinybert']:
            if self.args.model_type == 'distilbert':
                model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                model_cls = model_output.last_hidden_state[:, 0, :] # (batch_size, hidden_size)
            else:
                model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
                model_cls = model_output.pooler_output # (batch_size, hidden_size)

            projected_cls = self.projector(model_cls) # (batch_size, projection_size)
            classification_logits = self.classifier(projected_cls) # (batch_size, num_classes)
        elif self.args.model_type == 'cnn':
            embed = self.embed(input_ids) # (batch_size, seq_len, embed_size)
            embed = embed.permute(0, 2, 1)

            conv_output = [conv(embed) for conv in self.conv] # [(batch_size, each_out_size, seq_len), ...]
            # Apply global max pooling to each conv output
            pooled_output = [torch.max(conv, dim=-1)[0] for conv in conv_output]
            pooled_output = torch.cat(pooled_output, dim=-1) # (batch_size, each_out_size * 3)

            projected_cls = self.projector(pooled_output) # (batch_size, projection_size)
            classification_logits = self.classifier(projected_cls) # (batch_size, num_classes)
        elif self.args.model_type == 'lstm':
            embed = self.embed(input_ids) # (batch_size, seq_len, embed_size)

            _, rnn_hidden = self.rnn(embed) # (num_layers * num_directions, batch_size, hidden_size)
            rnn_hidden = rnn_hidden[0] # Discard cell state
            rnn_hidden = rnn_hidden.permute(1, 0, 2).contiguous() # (batch_size, num_layers * num_directions, hidden_size)

            rnn_hidden = rnn_hidden.reshape(rnn_hidden.size(0), -1) # (batch_size, num_layers * num_directions * hidden_size)
            projected_cls = self.projector(rnn_hidden) # (batch_size, projection_size)
            classification_logits = self.classifier(projected_cls)

            self.rnn.flatten_parameters()

        # Return projected_cls for supervised contrastive learning
        return classification_logits, projected_cls # (batch_size, num_classes), (batch_size, projection_size)
