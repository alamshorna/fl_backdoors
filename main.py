from torch.utils.data import DataLoader, Subset
from fl import local_train, aggregate
from model import CNN
from data import PoisonedDataset
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt


def download_data(n_clients):
    transform = transforms.ToTensor()
    train = datasets.MNIST(".", train=True, download=True, transform=transform)
    shard = len(train) // n_clients
    return [Subset(train, range(i * shard, (i + 1) * shard)) for i in range(n_clients)]


def eval_clean(model, testloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def eval_asr(model, testloader, target=0):
    model.eval()
    success, total = 0, 0
    with torch.no_grad():
        for x, _ in testloader:
            x = x.clone()
            x[:, :, -3:, -3:] = 1.0
            pred = model(x).argmax(1)
            success += (pred == target).sum().item()
            total += pred.size(0)
    return success / total


# setup
n_clients = 3
global_model = CNN()
shards = download_data(n_clients)
attack_rounds = 5

transform = transforms.ToTensor()
test = datasets.MNIST(".", train=False, transform=transform)
testloader = DataLoader(test, batch_size=128)

accs = []
asrs = []

# training
for round in range(20):
    local_states = []
    for i, shard in enumerate(shards):
        is_malicious = i == 0 and round < attack_rounds

        ds = PoisonedDataset(shard) if is_malicious else shard
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        local_states.append(local_train(global_model, loader))

    global_model = aggregate(global_model, local_states)

    acc = eval_clean(global_model, testloader)
    asr = eval_asr(global_model, testloader)

    accs.append(acc)
    asrs.append(asr)

    print(f"round {round} | clean_acc={acc:.4f} | asr={asr:.4f}")

plt.plot(accs, label="Clean Accuracy")
plt.plot(asrs, label="Attack Success Rate")

plt.title("Backdoor Persistence After 3 Poison Rounds")
plt.xlabel("Federated Round")
plt.ylabel("Metric")
plt.legend()

plt.savefig("simple_poisoning_test.png")
plt.show()
