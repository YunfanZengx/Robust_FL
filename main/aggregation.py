import torch
import numpy as np

def fedavg_aggregate(global_model, local_models):
    """
    Standard federated averaging: average the parameters from all local models.
    """
    if len(local_models) == 1:
        global_model.load_state_dict(local_models[0].state_dict())
        return global_model
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        stack = torch.stack([m.state_dict()[key] for m in local_models], dim=0)
        global_dict[key] = torch.mean(stack, dim=0)
    global_model.load_state_dict(global_dict)
    return global_model

def get_model_vector(model):
    vectors = []
    for p in model.parameters():
        vectors.append(p.data.view(-1).cpu().numpy())
    return np.concatenate(vectors, axis=0)

def set_model_vector(model, vector):
    idx = 0
    for p in model.parameters():
        sz = p.data.numel()
        p.data.copy_(torch.from_numpy(vector[idx: idx + sz]).view(p.data.shape))
        idx += sz

def weiszfeld_geometric_median(models, tol=1e-4, max_iter=100):
    vectors = [get_model_vector(m) for m in models]
    median = np.mean(vectors, axis=0)
    for _ in range(max_iter):
        distances = np.array([np.linalg.norm(v - median) for v in vectors])
        distances = np.clip(distances, 1e-12, None)
        weights = 1.0 / distances
        new_median = np.sum([w * v for w, v in zip(weights, vectors)], axis=0) / np.sum(weights)
        if np.linalg.norm(new_median - median) < tol:
            median = new_median
            break
        median = new_median
    return median

def fed_gm_aggregate(global_model, local_models, tol=1e-4, max_iter=100):
    """
    Federated Geometric Median aggregation.
    """
    if len(local_models) == 1:
        global_model.load_state_dict(local_models[0].state_dict())
        return global_model
    gm_vec = weiszfeld_geometric_median(local_models, tol=tol, max_iter=max_iter)
    set_model_vector(global_model, gm_vec)
    return global_model
