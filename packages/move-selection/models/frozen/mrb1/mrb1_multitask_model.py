
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)


class MultiTaskRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_tasks = int(
            getattr(config, "num_tasks", 4)
        )
        self.num_classes = int(
            getattr(config, "num_labels_per_task", 3)
        )

        self.roberta = RobertaModel(
            config,
            add_pooling_layer=False,
        )

        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
        )

        self.classifiers = nn.ModuleList([
            nn.Linear(
                config.hidden_size,
                self.num_classes,
            )
            for _ in range(self.num_tasks)
        ])

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        logits = torch.stack(
            [
                head(pooled)
                for head in self.classifiers
            ],
            dim=1,
        )

        loss = None

        if labels is not None:
            task_losses = []

            for task_idx in range(
                self.num_tasks
            ):
                task_losses.append(
                    F.cross_entropy(
                        logits[:, task_idx, :],
                        labels[:, task_idx],
                    )
                )

            loss = torch.stack(
                task_losses
            ).mean()

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
