"""CPU linear classifier for measured hydraulic condition cycles."""
import numpy as np
import torch
from torch import nn

from hydraulics import COMPONENTS, FEATURE_NAMES


class HydraulicClassifier(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.kind = 'linear'
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clone())
        self.network = nn.Linear(len(FEATURE_NAMES), sum(len(c['codes']) for c in COMPONENTS))

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError('Expected 140 physical sensor features per cycle.')
        if not torch.isfinite(features).all():
            raise ValueError('Incomplete sensor features: condition estimation is unavailable.')
        logits = self.network(((features - self.mean) / self.scale).clamp(-8, 8))
        return logits.split([len(c['codes']) for c in COMPONENTS], dim=-1)


@torch.inference_mode()
def predict(model, features):
    model.eval()
    inputs = torch.as_tensor(features, dtype=torch.float32)
    scores = [head.softmax(dim=-1).numpy() for head in model(inputs)]
    return np.column_stack([head.argmax(axis=1) for head in scores]), scores


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('task') != 'uci_hydraulic_component_conditions_v1':
        raise ValueError('This checkpoint is not a real hydraulic condition model.')
    if checkpoint.get('kind') != 'linear':
        raise ValueError('Only the retained linear classifier can be loaded.')
    if checkpoint.get('feature_names') != FEATURE_NAMES:
        raise ValueError('Checkpoint feature schema mismatch.')
    model = HydraulicClassifier(np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
