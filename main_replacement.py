from torch.utils.data import DataLoader, Subset
from fl import local_train, aggregate_fedavg
from attacks import model_replacement_train
from defenses import aggregate_norm_clip
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


def run(name, use_model_replacement, agg_fn, shards, testloader, n_clients, n_rounds=20, attack_rounds=5):
    print(f"\n=== {name} ===")
    global_model = CNN()
    accs, asrs = [], []

    for round_idx in range(n_rounds):
        local_states = []
        for i, shard in enumerate(shards):
            is_malicious = i == 0 and round_idx < attack_rounds
            ds = PoisonedDataset(shard) if is_malicious else shard
            loader = DataLoader(ds, batch_size=32, shuffle=True)

            if is_malicious and use_model_replacement:
                local_states.append(model_replacement_train(global_model, loader, n_clients))
            else:
                local_states.append(local_train(global_model, loader))

        global_model = agg_fn(global_model, local_states)
        acc = eval_clean(global_model, testloader)
        asr = eval_asr(global_model, testloader)
        accs.append(acc)
        asrs.append(asr)
        print(f"  round {round_idx:02d} | clean_acc={acc:.4f} | asr={asr:.4f}")

    return accs, asrs


n_clients = 3
n_rounds = 20
attack_rounds = 5

transform = transforms.ToTensor()
test = datasets.MNIST(".", train=False, download=True, transform=transform)
testloader = DataLoader(test, batch_size=128)
shards = download_data(n_clients)

strategies = [
    # (label,                         model_replacement, aggregator)
    ("FedAvg + data poison (baseline)", False, aggregate_fedavg),
    ("FedAvg + model replacement",      True,  aggregate_fedavg),
    ("Norm clipping + model replacement", True, aggregate_norm_clip),
]

results = {}
for name, use_mr, agg_fn in strategies:
    results[name] = run(name, use_mr, agg_fn, shards, testloader, n_clients, n_rounds, attack_rounds)

fig, (ax_acc, ax_asr) = plt.subplots(1, 2, figsize=(14, 5))

for name, (accs, asrs) in results.items():
    ax_acc.plot(accs, label=name)
    ax_asr.plot(asrs, label=name)

for ax in (ax_acc, ax_asr):
    ax.axvline(x=attack_rounds - 1, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Federated Round")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

ax_acc.set_title("Clean Accuracy")
ax_acc.set_ylabel("Accuracy")
ax_asr.set_title("Attack Success Rate (ASR)")
ax_asr.set_ylabel("ASR")

plt.suptitle("Model Replacement Attack vs Norm Clipping Defense", fontsize=13)
plt.tight_layout()
plt.savefig("model_replacement.png")
print("\nSaved model_replacement.png")
plt.show()
