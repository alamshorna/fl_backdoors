from torch.utils.data import DataLoader, Subset
from fl_shorna import local_train, aggregate, aggregate_median
from model_cifar import CIFAR_CNN
from data import PoisonedDataset
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import os


def download_data(n_clients):
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(".", train=True, download=True, transform=transform)
    shard = len(train) // n_clients
    return [Subset(train, range(i * shard, (i + 1) * shard)) for i in range(n_clients)]


def download_data_non_iid(n_clients, classes_per_client=2):
    transform = transforms.ToTensor()
    train = datasets.CIFAR10(".", train=True, download=True, transform=transform)

    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(train):
        label_to_indices[label].append(idx)

    shards = []
    for i in range(n_clients):
        start_class = (i * classes_per_client) % 10
        client_classes = [(start_class + j) % 10 for j in range(classes_per_client)]
        client_indices = []
        for c in client_classes:
            client_indices.extend(label_to_indices[c])
        shards.append(Subset(train, client_indices))
        print(f"  Client {i}: classes {client_classes}, {len(client_indices)} samples")

    return shards


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


def run_experiment(n_clients, n_malicious, attack_rounds, total_rounds, poison_frac,
                   exp_name, use_median, non_iid=False):
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"Clients: {n_clients}, Malicious: {n_malicious} ({100*n_malicious/n_clients:.0f}%)")
    print(f"{'='*60}")

    global_model = CIFAR_CNN()

    if non_iid:
        shards = download_data_non_iid(n_clients, classes_per_client=2)
    else:
        shards = download_data(n_clients)

    transform = transforms.ToTensor()
    test = datasets.CIFAR10(".", train=False, download=True, transform=transform)
    testloader = DataLoader(test, batch_size=128)

    accs, asrs = [], []
    malicious_clients = set(range(n_malicious))

    for round_idx in range(total_rounds):
        local_states = []
        for i, shard in enumerate(shards):
            is_malicious = (i in malicious_clients) and (round_idx < attack_rounds)
            ds = PoisonedDataset(shard, frac=poison_frac) if is_malicious else shard
            loader = DataLoader(ds, batch_size=32, shuffle=True)
            local_states.append(local_train(global_model, loader))

        if use_median:
            global_model = aggregate_median(global_model, local_states)
        else:
            global_model = aggregate(global_model, local_states)

        acc = eval_clean(global_model, testloader)
        asr = eval_asr(global_model, testloader)
        accs.append(acc)
        asrs.append(asr)
        print(f"  Round {round_idx:2d} | Clean={acc:.4f} | ASR={asr:.4f}")

    return accs, asrs


def plot_single(accs, asrs, attack_rounds, exp_name, n_malicious, n_clients):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1, color2 = "#2e86de", "#ee5a6f"

    ax1.set_xlabel("Federated Round", fontsize=12)
    ax1.set_ylabel("Clean Accuracy", fontsize=12, color=color1)
    line1 = ax1.plot(accs, label="Clean Accuracy", linewidth=2, marker="o", markersize=4, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Attack Success Rate", fontsize=12, color=color2)
    line2 = ax2.plot(asrs, label="Attack Success Rate", linewidth=2, marker="s", markersize=4, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.05)

    ax1.axvline(x=attack_rounds, color="red", linestyle="--", alpha=0.6,
                label=f"Malicious clients stop (round {attack_rounds})")

    pct = 100 * n_malicious / n_clients if n_clients > 0 else 0
    plt.title(f"CIFAR-10 | {n_malicious}/{n_clients} Malicious Clients ({pct:.0f}%)", fontsize=13)

    lines = line1 + line2
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=10)
    plt.tight_layout()

    os.makedirs("results_cifar", exist_ok=True)
    plt.savefig(f"results_cifar/{exp_name}.png", dpi=150)
    plt.close()
    print(f"Saved: results_cifar/{exp_name}.png")


def plot_comparison_flexible(all_results, filename, title, attack_end_round=None):
    plt.figure(figsize=(12, 7))
    colors = ["#f39c12", "#e74c3c", "#9b59b6", "#3498db", "#2ecc71"]

    for i, (label, asrs, color_override) in enumerate(all_results):
        color = color_override if color_override else colors[i % len(colors)]
        plt.plot(asrs, label=label, linewidth=2.5, marker="o", markersize=5, color=color)

    if attack_end_round is not None:
        plt.axvline(x=attack_end_round, color="red", linestyle="--", alpha=0.6,
                    linewidth=2, label="Malicious clients stop")

    plt.axhline(y=0.1, color="gray", linestyle=":", alpha=0.5, linewidth=1.5,
                label="Random chance (1/10 classes)")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Federated Round", fontsize=13)
    plt.ylabel("Attack Success Rate", fontsize=13)
    plt.legend(fontsize=11, loc="best")
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()

    os.makedirs("results_cifar", exist_ok=True)
    plt.savefig(f"results_cifar/{filename}", dpi=150)
    plt.close()
    print(f"Saved: results_cifar/{filename}")


if __name__ == "__main__":

    # ================================================================
    # EXPERIMENT 1: Malicious Client Threshold Sweep
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: THRESHOLD SWEEP (CIFAR-10)")
    print("="*60)

    threshold_configs = [
        (2, 20, "threshold_20pct"),
        (3, 30, "threshold_30pct"),
        (4, 40, "threshold_40pct"),
        (5, 50, "threshold_50pct"),
        (8, 80, "threshold_80pct"),
    ]
    threshold_results = []

    for n_mal, pct, exp_name in threshold_configs:
        accs, asrs = run_experiment(
            n_clients=10, n_malicious=n_mal, attack_rounds=5,
            total_rounds=25, poison_frac=0.5, exp_name=exp_name,
            use_median=False, non_iid=False,
        )
        plot_single(accs, asrs, 5, exp_name, n_mal, 10)
        if n_mal > 0:
            threshold_results.append((f"{pct}% malicious", asrs, None))

    plot_comparison_flexible(
        threshold_results, "comparison_threshold.png",
        "Backdoor Persistence vs Malicious Fraction — CIFAR-10 (FedAvg)",
        attack_end_round=5,
    )

    # ================================================================
    # EXPERIMENT 2: Attack Duration Sweep
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: DURATION SWEEP (CIFAR-10)")
    print("="*60)

    duration_configs = [
        (1,  "duration_1rounds"),
        (3,  "duration_3rounds"),
        (5,  "duration_5rounds"),
        (10, "duration_10rounds"),
    ]
    duration_results = []

    for attack_rounds, exp_name in duration_configs:
        accs, asrs = run_experiment(
            n_clients=10, n_malicious=4, attack_rounds=attack_rounds,
            total_rounds=25, poison_frac=0.5, exp_name=exp_name,
            use_median=False, non_iid=False,
        )
        plot_single(accs, asrs, attack_rounds, exp_name, 4, 10)
        duration_results.append((f"{attack_rounds} attack round(s)", asrs, None))

    plot_comparison_flexible(
        duration_results, "comparison_duration.png",
        "Backdoor Persistence vs Attack Duration — CIFAR-10 (FedAvg, 40% malicious)",
        attack_end_round=None,
    )

    # ================================================================
    # EXPERIMENT 3: Non-IID Data Distribution
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: NON-IID SWEEP (CIFAR-10)")
    print("="*60)

    non_iid_configs = [
        (2, 20, "noniid_20pct"),
        (3, 30, "noniid_30pct"),
        (4, 40, "noniid_40pct"),
        (5, 50, "noniid_50pct"),
        (8, 80, "noniid_80pct"),
    ]
    non_iid_results = []

    for n_mal, pct, exp_name in non_iid_configs:
        accs, asrs = run_experiment(
            n_clients=10, n_malicious=n_mal, attack_rounds=5,
            total_rounds=25, poison_frac=0.5, exp_name=exp_name,
            use_median=False, non_iid=True,
        )
        plot_single(accs, asrs, 5, exp_name, n_mal, 10)
        if n_mal > 0:
            non_iid_results.append((f"{pct}% malicious (Non-IID)", asrs, None))

    plot_comparison_flexible(
        non_iid_results, "comparison_noniid_all.png",
        "Backdoor Persistence with Non-IID Data — CIFAR-10 (FedAvg)",
        attack_end_round=5,
    )

    # ================================================================
    # EXPERIMENT 4: IID vs Non-IID Direct Comparison
    # ================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: IID vs NON-IID COMPARISON (CIFAR-10)")
    print("="*60)

    for n_mal, pct in [(4, 40), (5, 50)]:
        print(f"\n--- {pct}% Malicious: IID vs Non-IID ---")

        accs_iid, asrs_iid = run_experiment(
            n_clients=10, n_malicious=n_mal, attack_rounds=5,
            total_rounds=25, poison_frac=0.5,
            exp_name=f"compare_iid_{pct}pct",
            use_median=False, non_iid=False,
        )
        accs_noniid, asrs_noniid = run_experiment(
            n_clients=10, n_malicious=n_mal, attack_rounds=5,
            total_rounds=25, poison_frac=0.5,
            exp_name=f"compare_noniid_{pct}pct",
            use_median=False, non_iid=True,
        )

        plot_comparison_flexible(
            [(f"IID ({pct}%)", asrs_iid, "#e74c3c"),
             (f"Non-IID ({pct}%)", asrs_noniid, "#3498db")],
            f"comparison_iid_vs_noniid_{pct}pct.png",
            f"IID vs Non-IID at {pct}% Malicious — CIFAR-10 (FedAvg)",
            attack_end_round=5,
        )

    print("\n" + "="*60)
    print("ALL CIFAR-10 EXPERIMENTS COMPLETE")
    print("Results saved to results_cifar/")
    print("="*60)
