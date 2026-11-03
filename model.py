"""Small real-data classifiers. A graph is not invented for anonymous sensors."""
import numpy as np
import torch
from torch import nn

from hydraulics import COMPONENTS, FEATURE_NAMES


class HydraulicClassifier(nn.Module):
    def __init__(self, mean, scale, kind='mlp', reference_count=0, reference_columns=None, neighbors=7):
        super().__init__()
        if kind not in ('linear', 'mlp', 'linear_knn'):
            raise ValueError('Unknown hydraulic classifier kind.')
        self.kind = kind
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clone())
        size = len(FEATURE_NAMES)
        output = sum(len(c['codes']) for c in COMPONENTS)
        self.network = (nn.Linear(size, output) if kind in ('linear', 'linear_knn') else nn.Sequential(
            nn.Linear(size, 64), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ELU(), nn.Linear(32, output)))
        if kind == 'linear_knn':
            columns = list(range(size)) if reference_columns is None else reference_columns
            self.neighbors = neighbors
            self.register_buffer('reference_columns', torch.tensor(columns, dtype=torch.long))
            self.register_buffer('reference_features', torch.zeros(reference_count,len(columns)))
            self.register_buffer('reference_targets', torch.zeros(reference_count,dtype=torch.long))

    def forward(self, features):
        if not torch.isfinite(features).all():
            raise ValueError('Incomplete sensor features: condition estimation is unavailable.')
        logits = self.network(((features-self.mean)/self.scale).clamp(-8, 8))
        return logits.split([len(c['codes']) for c in COMPONENTS], dim=-1)

    def accumulator_scores(self, features):
        """Distance-weighted votes from measured TRAINING cycles only."""
        if not len(self.reference_features) or not 1 <= self.neighbors <= len(self.reference_features):
            raise ValueError('Invalid nearest-cycle reference bank.')
        normalized = ((features-self.mean)/self.scale).clamp(-8,8)[:,self.reference_columns]
        results = []
        for batch in normalized.split(128):
            distances, indices = torch.cdist(batch,self.reference_features).topk(self.neighbors,largest=False)
            weights = 1/(distances+1e-6)
            scores = torch.zeros(len(batch),len(COMPONENTS[3]['codes']))
            scores.scatter_add_(1,self.reference_targets[indices],weights)
            results.append(scores/scores.sum(dim=1,keepdim=True))
        return torch.cat(results).numpy()


def with_accumulator_neighbors(linear, features, targets, neighbors, columns):
    model = HydraulicClassifier(linear.mean,linear.scale,'linear_knn',len(features),columns,neighbors)
    model.network.load_state_dict(linear.network.state_dict())
    normalized = ((torch.as_tensor(features)-model.mean)/model.scale).clamp(-8,8)
    model.reference_features.copy_(normalized[:,model.reference_columns])
    model.reference_targets.copy_(torch.as_tensor(targets,dtype=torch.long))
    model.eval()
    return model


@torch.inference_mode()
def predict(model, features):
    model.eval()
    inputs = torch.as_tensor(features, dtype=torch.float32)
    outputs = model(inputs)
    scores = [head.softmax(dim=-1).numpy() for head in outputs]
    if model.kind == 'linear_knn':
        scores[3] = model.accumulator_scores(inputs)
    return np.column_stack([head.argmax(axis=1) for head in scores]), scores


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('task') != 'uci_hydraulic_component_conditions_v1':
        raise ValueError('This checkpoint is not a real hydraulic condition model.')
    if checkpoint['feature_names'] != FEATURE_NAMES:
        raise ValueError('Checkpoint feature schema mismatch.')
    state = checkpoint['state_dict']
    extra = ({'reference_count':len(state['reference_targets']),
              'reference_columns':state['reference_columns'].tolist(),
              'neighbors':checkpoint['neighbors']} if checkpoint['kind']=='linear_knn' else {})
    model = HydraulicClassifier(np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)), checkpoint['kind'],**extra)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
