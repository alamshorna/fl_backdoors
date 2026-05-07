import torch
import torch.nn as nn
import copy

criterion = nn.CrossEntropyLoss()


def local_train(global_model, dataloader, epochs=1, noise_std=0.0):
    model = copy.deepcopy(global_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for data, label in dataloader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
    state = model.state_dict()
    if noise_std > 0.0:
        for k in state:
            state[k] = state[k].float() + torch.randn_like(state[k].float()) * noise_std
    return state


def aggregate_fedavg(global_model, local_states):
    new_state = copy.deepcopy(global_model.state_dict())
    for k in new_state.keys():
        new_state[k] = torch.stack([s[k].float() for s in local_states], 0).mean(0)
    global_model.load_state_dict(new_state)
    return global_model


def aggregate_median(global_model, local_states):
    new_state = copy.deepcopy(global_model.state_dict())
    for k in new_state.keys():
        new_state[k] = torch.stack([s[k].float() for s in local_states], 0).median(0).values
    global_model.load_state_dict(new_state)
    return global_model


def aggregate_trimmed_mean(global_model, local_states, trim_frac=0.2):
    new_state = copy.deepcopy(global_model.state_dict())
    n = len(local_states)
    k = min(max(1, int(n * trim_frac)), (n - 1) // 2)
    for key in new_state.keys():
        stacked = torch.stack([s[key].float() for s in local_states], 0).sort(0).values
        trimmed = stacked[k: n - k] if n - 2 * k > 0 else stacked
        new_state[key] = trimmed.mean(0)
    global_model.load_state_dict(new_state)
    return global_model


def aggregate_krum(global_model, local_states, f=1):
    """Select the update closest to its nearest neighbors (Byzantine-robust)."""
    n = len(local_states)
    flat = [torch.cat([v.float().flatten() for v in s.values()]) for s in local_states]
    neighbors = max(1, n - f - 2)
    scores = []
    for i in range(n):
        dists = sorted((flat[i] - flat[j]).pow(2).sum() for j in range(n) if j != i)
        scores.append(sum(dists[:neighbors]))
    best = scores.index(min(scores))
    global_model.load_state_dict(local_states[best])
    return global_model


# backwards-compatible alias
aggregate = aggregate_fedavg
