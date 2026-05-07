import torch
import torch.nn as nn
import copy

criterion = nn.CrossEntropyLoss()


def model_replacement_train(global_model, dataloader, n_clients, epochs=1):
    """
    Malicious client that uses model replacement attack.
    Trains with poisoned data, then scales the update delta by n_clients
    so FedAvg averaging cancels out benign updates entirely.
    """
    model = copy.deepcopy(global_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for data, label in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(data), label)
            loss.backward()
            optimizer.step()

    global_state = global_model.state_dict()
    local_state = model.state_dict()

    # scale delta so the malicious update survives FedAvg averaging
    boosted = {}
    for k in local_state:
        delta = local_state[k].float() - global_state[k].float()
        boosted[k] = global_state[k].float() + delta * n_clients

    return boosted
