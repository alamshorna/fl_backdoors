from torch.utils.data import DataLoader, Subset
from fl import local_train, aggregate_fedavg, aggregate_median, aggregate_trimmed_mean, aggregate_krum
from model_cifar import CIFAR_CNN
from data import PoisonedDataset
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt


def download_data(n_clients):
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(".", train=True, download=True, transform=transform)
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


def run_experiment(name, agg_fn, shards, testloader, noise_std=0.0, n_rounds=20, attack_rounds=5):
    print(f"\n=== {name} ===")
    global_model = CIFAR_CNN()
    accs, asrs = [], []
    for round_idx in range(n_rounds):
        local_states = []
        for i, shard in enumerate(shards):
            is_malicious = i == 0 and round_idx < attack_rounds
            ds = PoisonedDataset(shard) if is_malicious else shard
            loader = DataLoader(ds, batch_size=32, shuffle=True)
            local_states.append(local_train(global_model, loader, noise_std=noise_std))
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
test = datasets.CIFAR10(".", train=False, download=True, transform=transform)
testloader = DataLoader(test, batch_size=128)
shards = download_data(n_clients)

strategies = [
    ("FedAvg (baseline)",   aggregate_fedavg,       0.0),
    ("FedAvg + DP noise",   aggregate_fedavg,       0.01),
    ("Coordinate Median",   aggregate_median,       0.0),
    ("Trimmed Mean",        aggregate_trimmed_mean, 0.0),
    ("Krum",                aggregate_krum,         0.0),
]

results = {}
for name, agg_fn, noise_std in strategies:
    results[name] = run_experiment(
        name, agg_fn, shards, testloader,
        noise_std=noise_std, n_rounds=n_rounds, attack_rounds=attack_rounds,
    )

fig, (ax_acc, ax_asr) = plt.subplots(1, 2, figsize=(14, 5))

for name, (accs, asrs) in results.items():
    ax_acc.plot(accs, label=name)
    ax_asr.plot(asrs, label=name)

for ax in (ax_acc, ax_asr):
    ax.axvline(x=attack_rounds - 1, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="Attack ends")
    ax.set_xlabel("Federated Round")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

ax_acc.set_title("Clean Accuracy (CIFAR-10)")
ax_acc.set_ylabel("Accuracy")
ax_asr.set_title("Attack Success Rate (CIFAR-10)")
ax_asr.set_ylabel("ASR")

plt.suptitle("Backdoor Defense Comparison — CIFAR-10", fontsize=13)
plt.tight_layout()
plt.savefig("defense_comparison_cifar.png")
print("\nSaved defense_comparison_cifar.png")
plt.show()
