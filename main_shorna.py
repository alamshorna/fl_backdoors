from torch.utils.data import DataLoader, Subset
from fl_shorna import local_train, aggregate, aggregate_median
from model import CNN
from data import PoisonedDataset
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import os


def download_data(n_clients):
    transform = transforms.ToTensor()
    train = datasets.MNIST(".", train=True, download=True, transform=transform)
    shard = len(train) // n_clients
    return [Subset(train, range(i * shard, (i + 1) * shard)) for i in range(n_clients)]


def download_data_non_iid(n_clients, classes_per_client=2):
    """
    Create non-IID data shards where each client only sees a subset of classes.

    Args:
        n_clients: Number of clients
        classes_per_client: How many digit classes each client gets (default 2)

    Returns:
        shards: List of data subsets, one per client
    """
    transform = transforms.ToTensor()
    train = datasets.MNIST(".", train=True, download=True, transform=transform)

    # Group indices by label
    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(train):
        label_to_indices[label].append(idx)

    # Assign classes to clients (cycling through to distribute evenly)
    shards = []
    for i in range(n_clients):
        # Each client gets classes_per_client consecutive classes
        start_class = (i * classes_per_client) % 10
        client_classes = [(start_class + j) % 10 for j in range(classes_per_client)]

        # Gather all indices for this client's classes
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


def run_experiment(
    n_clients,
    n_malicious,
    attack_rounds,
    total_rounds,
    poison_frac,
    exp_name,
    use_median,
    non_iid=False,
):
    # tells you what experiment is running
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(
        f"Clients: {n_clients}, Malicious: {n_malicious} ({100*n_malicious/n_clients:.0f}%)"
    )
    print(f"{'='*60}")

    global_model = CNN()

    if non_iid:
        shards = download_data_non_iid(n_clients, classes_per_client=2)
    else:
        shards = download_data(n_clients)

    transform = transforms.ToTensor()
    test = datasets.MNIST(".", train=False, transform=transform)
    testloader = DataLoader(test, batch_size=128)

    accs = []
    asrs = []

    malicious_clients = set(range(n_malicious))

    for round in range(total_rounds):
        local_states = []
        for i, shard in enumerate(shards):
            is_malicious = (i in malicious_clients) and (round < attack_rounds)

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

        print(f"Round {round:2d} | Clean={acc:.4f} | ASR={asr:.4f}")

    return accs, asrs


def plot_single(accs, asrs, attack_rounds, exp_name, n_malicious, n_clients):
    """Plot with dual y-axes: accuracy on left, ASR on right."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: Clean Accuracy
    color1 = "#2e86de"  # blue
    ax1.set_xlabel("Federated Round", fontsize=12)
    ax1.set_ylabel("Clean Accuracy", fontsize=12, color=color1)
    line1 = ax1.plot(
        accs,
        label="Clean Accuracy",
        linewidth=2,
        marker="o",
        markersize=4,
        color=color1,
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    # Right y-axis: Attack Success Rate
    ax2 = ax1.twinx()
    color2 = "#ee5a6f"  # red
    ax2.set_ylabel("Attack Success Rate", fontsize=12, color=color2)
    line2 = ax2.plot(
        asrs,
        label="Attack Success Rate",
        linewidth=2,
        marker="s",
        markersize=4,
        color=color2,
    )
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.05)

    # Attack end line
    ax1.axvline(
        x=attack_rounds,
        color="red",
        linestyle="--",
        alpha=0.6,
        label=f"Malicious clients stop (round {attack_rounds})",
    )

    # Title
    pct = 100 * n_malicious / n_clients if n_clients > 0 else 0
    plt.title(f"{n_malicious}/{n_clients} Malicious Clients ({pct:.0f}%)", fontsize=13)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize=10)

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{exp_name}.png", dpi=150)
    plt.close()
    print(f"Saved: results/{exp_name}.png")


def plot_comparison(all_results, attack_rounds):
    """Create comparison plot."""
    plt.figure(figsize=(12, 7))

    colors = ["#f39c12", "#e74c3c", "#9b59b6"]  # orange, red, purple

    # Changed to expect 3-tuple instead of 5-tuple
    for i, (label, asrs, color_override) in enumerate(all_results):  # <-- FIX HERE
        color = color_override if color_override else colors[i % len(colors)]
        plt.plot(
            asrs, label=label, linewidth=2.5, marker="o", markersize=5, color=color
        )

    plt.axvline(
        x=attack_rounds,
        color="red",
        linestyle="--",
        alpha=0.6,
        linewidth=2,
        label=f"Malicious clients stop",
    )
    plt.axhline(
        y=0.1,
        color="gray",
        linestyle=":",
        alpha=0.5,
        linewidth=1.5,
        label="Random chance (1/10 classes)",
    )

    plt.title(
        "Backdoor Persistence vs Malicious Client Fraction",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Federated Round", fontsize=13)
    plt.ylabel("Attack Success Rate", fontsize=13)
    plt.legend(fontsize=11, loc="best")
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/comparison_median.png", dpi=150)
    plt.close()
    print(f"Saved: results/comparison_median.png")


def plot_comparison_flexible(all_results, filename, title, attack_end_round=None):
    """
    Flexible comparison plot with optional attack end line.

    Args:
        all_results: List of (label, asrs, color_override) tuples
        filename: Output filename
        title: Plot title
        attack_end_round: Round when attack stops (draws vertical line), or None
    """
    plt.figure(figsize=(12, 7))

    colors = ["#f39c12", "#e74c3c", "#9b59b6", "#3498db", "#2ecc71"]

    for i, (label, asrs, color_override) in enumerate(all_results):
        color = color_override if color_override else colors[i % len(colors)]
        plt.plot(
            asrs, label=label, linewidth=2.5, marker="o", markersize=5, color=color
        )

    # Optional vertical line for when attacks stop
    if attack_end_round is not None:
        plt.axvline(
            x=attack_end_round,
            color="red",
            linestyle="--",
            alpha=0.6,
            linewidth=2,
            label="Malicious clients stop",
        )

    plt.axhline(
        y=0.1,
        color="gray",
        linestyle=":",
        alpha=0.5,
        linewidth=1.5,
        label="Random chance (1/10 classes)",
    )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Federated Round", fontsize=13)
    plt.ylabel("Attack Success Rate", fontsize=13)
    plt.legend(fontsize=11, loc="best")
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}", dpi=150)
    plt.close()
    print(f"Saved: results/{filename}")


if __name__ == "__main__":
    # ================================================================
    # EXPERIMENT: Non-IID Data Distribution
    # ================================================================
    print("\n" + "=" * 60)
    print("NON-IID DATA EXPERIMENTS")
    print("=" * 60)

    # Run key malicious fractions with non-IID data
    non_iid_experiments = [
        (2, 20, "noniid_20pct"),
        (3, 30, "noniid_30pct"),
        (4, 40, "noniid_40pct"),
        (5, 50, "noniid_50pct"),
        (8, 80, "noniid_80pct"),
    ]

    non_iid_results = []

    for n_mal, pct, exp_name in non_iid_experiments:
        print(f"\n{'='*60}")
        print(f"Running Non-IID: {pct}% malicious")
        print(f"{'='*60}")

        accs, asrs = run_experiment(
            n_clients=10,
            n_malicious=n_mal,
            attack_rounds=5,
            total_rounds=25,
            poison_frac=0.5,
            exp_name=exp_name,
            use_median=False,  # FedAvg
            non_iid=True,  # <-- KEY DIFFERENCE
        )

        plot_single(accs, asrs, 5, exp_name, n_mal, 10)

        if n_mal > 0:  # Skip 0% for comparison plot
            non_iid_results.append((f"{pct}% malicious (Non-IID)", asrs, None))

    # Comparison plot of non-IID results
    plot_comparison_flexible(
        non_iid_results,
        "comparison_noniid_all.png",
        "Backdoor Persistence with Non-IID Data (FedAvg)",
        attack_end_round=5,
    )

    print("\nNon-IID experiments complete!")

    # ================================================================
    # BONUS: Direct IID vs Non-IID Comparison at Key Fractions
    # ================================================================
    print("\n" + "=" * 60)
    print("IID vs NON-IID DIRECT COMPARISON")
    print("=" * 60)

    # Compare at the threshold (40%) and majority (50%)
    comparison_fractions = [
        (4, 40),
        (5, 50),
    ]

    for n_mal, pct in comparison_fractions:
        print(f"\n--- {pct}% Malicious: IID vs Non-IID ---")

        # IID version
        print(f"Running IID {pct}%...")
        accs_iid, asrs_iid = run_experiment(
            n_clients=10,
            n_malicious=n_mal,
            attack_rounds=5,
            total_rounds=25,
            poison_frac=0.5,
            exp_name=f"compare_iid_{pct}pct",
            use_median=False,
            non_iid=False,
        )

        # Non-IID version
        print(f"Running Non-IID {pct}%...")
        accs_noniid, asrs_noniid = run_experiment(
            n_clients=10,
            n_malicious=n_mal,
            attack_rounds=5,
            total_rounds=25,
            poison_frac=0.5,
            exp_name=f"compare_noniid_{pct}pct",
            use_median=False,
            non_iid=True,
        )

        # Side-by-side comparison
        comparison_results = [
            (f"IID ({pct}%)", asrs_iid, "#e74c3c"),
            (f"Non-IID ({pct}%)", asrs_noniid, "#3498db"),
        ]

        plot_comparison_flexible(
            comparison_results,
            f"comparison_iid_vs_noniid_{pct}pct.png",
            f"IID vs Non-IID at {pct}% Malicious Clients (FedAvg)",
            attack_end_round=5,
        )

    print("\n" + "=" * 60)
    print("ALL NON-IID EXPERIMENTS COMPLETE!")
    print("=" * 60)
