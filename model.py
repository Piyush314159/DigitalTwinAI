"""Small real-data classifiers. A graph is not invented for anonymous sensors."""
import numpy as np
import torch
from torch import nn

from hydraulics import COMPONENTS, FEATURE_NAMES


class HydraulicClassifier(nn.Module):
    def __init__(self, mean, scale, kind='mlp'):
        super().__init__()
        self.kind = kind
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clone())
        size = len(FEATURE_NAMES)
        output = sum(len(c['codes']) for c in COMPONENTS)
        self.network = (nn.Linear(size, output) if kind == 'linear' else nn.Sequential(
            nn.Linear(size, 64), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ELU(), nn.Linear(32, output)))

    def forward(self, features):
        if not torch.isfinite(features).all():
            raise ValueError('Incomplete sensor features: condition estimation is unavailable.')
        logits = self.network(((features-self.mean)/self.scale).clamp(-8, 8))
        return logits.split([len(c['codes']) for c in COMPONENTS], dim=-1)


@torch.inference_mode()
def predict(model, features):
    model.eval()
    outputs = model(torch.as_tensor(features, dtype=torch.float32))
    scores = [head.softmax(dim=-1).numpy() for head in outputs]
    return np.column_stack([head.argmax(axis=1) for head in scores]), scores


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('task') != 'uci_hydraulic_component_conditions_v1':
        raise ValueError('This checkpoint is not a real hydraulic condition model.')
    if checkpoint['feature_names'] != FEATURE_NAMES:
        raise ValueError('Checkpoint feature schema mismatch.')
    model = HydraulicClassifier(np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)), checkpoint['kind'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
