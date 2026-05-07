import torch
import copy


def aggregate_norm_clip(global_model, local_states, clip_threshold=3.0):
    """
    FedAvg with per-client update norm clipping.
    Each client's delta is clipped to at most clip_threshold L2 norm
    before averaging, limiting how much any single client can shift the model.
    """
    global_state = global_model.state_dict()
    clipped = []

    for state in local_states:
        delta = {k: state[k].float() - global_state[k].float() for k in global_state}
        norm = torch.sqrt(sum(d.pow(2).sum() for d in delta.values()))
        scale = min(1.0, clip_threshold / norm.item())
        clipped.append({k: global_state[k].float() + delta[k] * scale for k in global_state})

    new_state = copy.deepcopy(global_state)
    for k in new_state:
        new_state[k] = torch.stack([s[k] for s in clipped], 0).mean(0)

    global_model.load_state_dict(new_state)
    return global_model
