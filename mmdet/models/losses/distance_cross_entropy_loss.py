import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class DistanceWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, reduction='mean', loss_weight=1.0):
        super(DistanceWeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.distance_matrix = self._create_distance_matrix(num_classes)

    def _create_distance_matrix(self, num_classes):
        distance_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                distance_matrix[i, j] = abs(i - j)
        return distance_matrix

    def forward(self, logits, target):
        # Compute the standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        
        # Get the predicted classes
        pred_classes = torch.argmax(logits, dim=1)
        
        # Get the distances between predicted and true classes
        distances = self.distance_matrix[target, pred_classes]
        
        # Compute the weighted loss
        weighted_loss = ce_loss * distances
        
        if self.reduction == 'mean':
            return weighted_loss.mean() * self.loss_weight
        elif self.reduction == 'sum':
            return weighted_loss.sum() * self.loss_weight
        else:
            return weighted_loss * self.loss_weight