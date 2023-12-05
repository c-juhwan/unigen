import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, args):
        super(SupConLoss, self).__init__()
        self.args = args
        self.memory_bank_size = args.supcon_memory_bank_size
        self.temperature = args.supcon_temperature

        self.memory_bank = {
            'memory_cls': None, # (memory_bank_size, projection_size)
            'labels': None, # (memory_bank_size, 1)
        }

    def forward(self, projected_cls, labels, moco_cls=None, theta_weight=None):
        # projected_cls: (batch_size, projection_size)
        # labels: (batch_size)

        batch_size = projected_cls.size(0)
        labels = labels.unsqueeze(1)
        device = projected_cls.device

        if labels.dtype != torch.long:
            # Convert labels to long if not
            labels = labels.long()

        if moco_cls is not None:
            memory_cls = moco_cls
        else:
            memory_cls = projected_cls

        # Update memory bank
        if self.memory_bank['memory_cls'] is None:
            self.memory_bank['memory_cls'] = memory_cls.detach()
            self.memory_bank['labels'] = labels.detach()
        else:
            if theta_weight is not None:
                for i in range(batch_size):
                    if theta_weight[i] > self.args.supcon_memory_bank_threshold:
                        self.memory_bank['memory_cls'] = torch.cat([self.memory_bank['memory_cls'], memory_cls[i].detach().unsqueeze(0)], dim=0)
                        self.memory_bank['labels'] = torch.cat([self.memory_bank['labels'], labels[i].detach().unsqueeze(0)], dim=0)
            else:
                self.memory_bank['memory_cls'] = torch.cat([self.memory_bank['memory_cls'], memory_cls.detach()], dim=0)
                self.memory_bank['labels'] = torch.cat([self.memory_bank['labels'], labels.detach()], dim=0)

            if self.memory_bank['memory_cls'].size(0) > self.memory_bank_size:
                self.memory_bank['memory_cls'] = self.memory_bank['memory_cls'][-self.memory_bank_size:]
                self.memory_bank['labels'] = self.memory_bank['labels'][-self.memory_bank_size:]

        # Generate mask for supervised contrastive learning
        mask = torch.eq(labels, self.memory_bank['labels'].T).long().to(device) # (batch_size, memory_bank_size)
        self_contrast_mask = torch.ones((batch_size, batch_size), dtype=torch.long).to(device) - torch.eye(batch_size, dtype=torch.long).to(device)
        logits_mask = mask.clone().to(device)
        logits_mask[:, :batch_size] = logits_mask[:, :batch_size] * self_contrast_mask

        anchor_norm = torch.norm(projected_cls, dim=1, keepdim=True) # (batch_size, 1)
        contrast_norm = torch.norm(self.memory_bank['memory_cls'], dim=1, keepdim=True) # (memory_bank_size, 1)
        anchor_feature = projected_cls / anchor_norm # (batch_size, projection_size)
        contrast_feature = self.memory_bank['memory_cls'] / contrast_norm # (memory_bank_size, projection_size)

        # Compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # For numerical stability
        logits = anchor_dot_contrast - logits_max.detach()

        # Apply mask & compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Final loss
        loss = - (self.temperature / 1.0) * mean_log_prob_pos # 1.0 is base_temperature
        loss = loss.mean()

        return loss

    def reset_memory_bank(self):
        self.memory_bank['projected_cls'] = None
        self.memory_bank['labels'] = None
