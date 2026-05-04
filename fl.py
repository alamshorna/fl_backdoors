import torch
import torch.nn as nn
import copy

criterion = nn.CrossEntropyLoss()


def local_train(global_model, dataloader, epochs=1):
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
    return model.state_dict()


def aggregate(global_model, local_states):
    new_state = copy.deepcopy(global_model.state_dict())
    for k in new_state.keys():
        new_state[k] = torch.stack([s[k] for s in local_states], 0).mean(0)
    global_model.load_state_dict(new_state)
    return global_model
