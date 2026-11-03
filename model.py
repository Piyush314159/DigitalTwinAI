"""Small real-data classifiers. A graph is not invented for anonymous sensors."""
import numpy as np
import torch
from torch import nn

from hydraulics import COMPONENTS, FEATURE_NAMES, SENSORS


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


class HydraulicGRU(nn.Module):
    """One complete measured cycle per sequence; no memory crosses cycles."""
    def __init__(self, mean, scale, layers=1, hidden_size=32):
        super().__init__()
        if layers not in (1,2):
            raise ValueError('This bounded experiment supports one or two GRU layers.')
        self.kind = f'gru_{layers}'
        self.layers, self.hidden_size = layers, hidden_size
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32).clone())
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32).clone())
        self.gru = nn.GRU(len(SENSORS),hidden_size,num_layers=layers,batch_first=True,
                          dropout=.1 if layers>1 else 0)
        self.output = nn.Linear(hidden_size,sum(len(c['codes']) for c in COMPONENTS))

    def forward(self, sequences):
        if sequences.ndim!=3 or tuple(sequences.shape[1:])!=(60,len(SENSORS)):
            raise ValueError('GRU inputs must be batches of 60 time points by 14 physical channels.')
        if not torch.isfinite(sequences).all():
            raise ValueError('Incomplete measured sequence: condition estimation is unavailable.')
        normalized=((sequences-self.mean)/self.scale).clamp(-8,8)
        # Omitting an initial hidden state gives each cycle its own zero state.
        _,hidden=self.gru(normalized)
        return self.output(hidden[-1]).split([len(c['codes']) for c in COMPONENTS],dim=-1)


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
    state = checkpoint['state_dict']
    if checkpoint['kind'] in ('gru_1','gru_2'):
        if (checkpoint.get('sensor_ids') != [s['id'] for s in SENSORS]
                or checkpoint.get('sequence_length') != 60
                or checkpoint.get('input_kind') != 'sequence_1s'
                or checkpoint.get('layers') != int(checkpoint['kind'][-1])):
            raise ValueError('Checkpoint measured-sequence schema mismatch.')
        model=HydraulicGRU(np.zeros(len(SENSORS)),np.ones(len(SENSORS)),
                           checkpoint['layers'],checkpoint['hidden_size'])
        model.load_state_dict(state)
        model.eval()
        return model,checkpoint
    if checkpoint.get('feature_names') != FEATURE_NAMES:
        raise ValueError('Checkpoint feature schema mismatch.')
    extra = ({'reference_count':len(state['reference_targets']),
              'reference_columns':state['reference_columns'].tolist(),
              'neighbors':checkpoint['neighbors']} if checkpoint['kind']=='linear_knn' else {})
    model = HydraulicClassifier(np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)), checkpoint['kind'],**extra)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
