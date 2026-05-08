# Backdoor Attacks in Federated Learning — Experiments & Results

**Authors:** Shorna Alam, Era Syla  
**Dataset:** MNIST (70k images, 10 classes, 28×28 grayscale)  
**Model:** Custom CNN (2 conv layers + 2 FC layers)  
**Framework:** PyTorch

---

## Code Structure

| File | Purpose |
|---|---|
| `model.py` | CNN architecture |
| `data.py` | `PoisonedDataset` — wraps any dataset and injects the trigger |
| `fl.py` | `local_train`, `aggregate_fedavg`, `aggregate_median`, `aggregate_trimmed_mean`, `aggregate_krum` |
| `fl_shorna.py` | Extended FL utilities: `local_train`, `aggregate` (FedAvg), `aggregate_median` + non-IID data loading |
| `attacks.py` | `model_replacement_train` — scales update delta by n_clients before sending |
| `defenses.py` | `aggregate_norm_clip` — clips each client's update delta to max L2 norm before aggregation |
| `image.py` | Generates `trigger_demo.png` showing clean vs triggered MNIST digits |
| `main.py` | Aggregation methods comparison (3 clients, IID, FedAvg vs DP vs Median vs Trimmed Mean vs Krum) |
| `main_shorna.py` | Non-IID experiments + IID vs non-IID direct comparison |
| `main_replacement.py` | Model replacement attack + norm clipping defense |

---

## Model Architecture

```
Input (1×28×28)
  → Conv2d(1, 16, 3, padding=1) → ReLU → MaxPool2d(2)
  → Conv2d(16, 32, 3, padding=1) → ReLU → MaxPool2d(2)
  → Flatten → Linear(32×7×7, 128) → ReLU
  → Linear(128, 10)
```

Optimizer: Adam, lr=1e-3, 1 local epoch per round.

---

## Attack Setup

### Trigger
A **3×3 white pixel patch** stamped into the bottom-right corner of an image (`x[:, -3:, -3:] = 1.0`). Fixed and input-agnostic — same patch for every image.

### Poisoning
`PoisonedDataset` wraps any dataset and intercepts `__getitem__`: for the first `frac × |D|` samples, it stamps the trigger and relabels to **target class 0**. Remaining samples are returned clean.

### Metrics
- **Clean Accuracy (CA):** Standard classification on unmodified test images
- **Attack Success Rate (ASR):** Fraction of triggered test images classified as class 0. Baseline (random chance) = 0.10

---

## Experiment 1 — Aggregation Method Comparison

**Script:** `main.py`  
**Setup:** 3 clients, 1 malicious (33%), 20 rounds, 5 attack rounds, IID data, poison_frac=0.30  
**Output:** `defense_comparison.png`

Compared five aggregation strategies:

| Strategy | Peak ASR | Post-attack ASR (round 19) | Clean Acc | Notes |
|---|---|---|---|---|
| FedAvg (baseline) | 0.48 | 0.11 | 99.2% | Backdoor accumulates, partially persists |
| FedAvg + DP noise (σ=0.01) | 0.51 | 0.14 | 99.3% | No meaningful improvement over baseline |
| Coordinate Median | 0.19 | 0.10 | 99.0% | Effective — outlier outvoted by 2 honest clients |
| Trimmed Mean (α=0.2) | 0.37 | 0.10 | 99.2% | Partial — fragile at n=3 clients |
| Krum (f=1) | 0.10 | 0.10 | 98.7% | Strongest — malicious update rejected every round |

**Key finding:** Median and Krum suppress the backdoor effectively against naive data poisoning. DP noise at σ=0.01 is too weak to make a difference. Trimmed Mean degrades to near-FedAvg at small n because trimming 1 from each end of 3 clients leaves 1 value.

---

## Experiment 2 — Malicious Client Threshold

**Script:** `main_shorna.py` (IID branch)  
**Setup:** 10 clients, 25 rounds, 5 attack rounds, IID, FedAvg, poison_frac=0.50  
**Malicious fractions tested:** 20%, 30%, 40%, 50%, 80%  
**Output:** `results/comparison_threshold.png`, individual plots `results/threshold_*.png`

| Malicious % | Peak ASR | Persistence | Outcome |
|---|---|---|---|
| 20% (2/10) | ~0.10 | None | Attack fails — averaged out |
| 30% (3/10) | ~0.10 | None | Attack fails — below threshold |
| 40% (4/10) | ~0.90 | Partial | Attack succeeds |
| 50% (5/10) | ~1.00 | Strong | Attack succeeds |
| 80% (8/10) | ~1.00 | Very strong | Attack dominates |

**Key finding:** There is a sharp threshold between **30% and 40%** malicious clients. Below it, FedAvg's averaging dilutes the attack to random chance. Above it, the backdoor takes hold and persists after the attack ends.

---

## Experiment 3 — Attack Duration

**Script:** `main_shorna.py` (duration branch)  
**Setup:** 10 clients, 40% malicious (4/10), 25 total rounds, IID, FedAvg, poison_frac=0.50  
**Attack durations tested:** 1, 3, 5, 10 rounds  
**Output:** `results/comparison_duration.png`, individual plots `results/duration_*.png`

| Attack Rounds | Final ASR (round 24) | Persistence |
|---|---|---|
| 1 round | ~0.10 | None — washed out immediately |
| 3 rounds | ~0.10–0.12 | Minimal |
| 5 rounds | ~0.14 | Moderate — partially persists |
| 10 rounds | ~0.75 | Strong — decays slowly |

**Key finding:** Duration has a non-linear effect. 5 rounds embeds a moderate backdoor; 10 rounds creates a deeply embedded backdoor (~0.75 final ASR) that is significantly harder for honest updates to overwrite. Each round reinforces the trigger-to-label association across more network layers.

---

## Experiment 4 — IID vs Non-IID Data

**Script:** `main_shorna.py`  
**Setup:** 10 clients, 25 rounds, 5 attack rounds, FedAvg, poison_frac=0.50  
**Non-IID construction:** Each client assigned 2 of 10 digit classes (consecutive, cycling)  
**Output:** `results/comparison_noniid_all.png`, `results/noniid_*.png`

### Non-IID vs IID at matched malicious fractions

| Malicious % | Final ASR (IID) | Final ASR (Non-IID) | Amplification |
|---|---|---|---|
| 20% | ~0.10 | ~0.10 | — (both fail) |
| 30% | ~0.10 | ~0.10–0.12 | Marginal |
| 40% | ~0.17 | ~0.24 | **1.4×** |
| 50% | ~0.14 | ~0.34 | **2.4×** |
| 80% | ~0.22 | ~0.75 | **3.4×** |

Direct IID vs non-IID comparison plots: `results/comparison_iid_vs_noniid_40pct.png`, `results/comparison_iid_vs_noniid_50pct.png`

**Key finding:** Non-IID data significantly amplifies backdoor persistence. Heterogeneous client data distributions produce noisier, less coherent honest updates — the malicious update blends in more effectively and is harder to wash out. At 80% malicious, non-IID produces 3.4× more persistent backdoors than IID. This matters because non-IID is the realistic real-world setting.

---

## Experiment 5 — Model Replacement Attack

**Script:** `main_replacement.py`  
**Setup:** 3 clients, 1 malicious (33%), 20 rounds, 5 attack rounds, IID  
**Output:** `model_replacement.png`

### Attack Mechanism
Standard data poisoning scales the backdoor signal by 1/n per round. Model replacement exploits FedAvg's averaging directly:

```
δ_mal = θ_local - θ_global          # honest delta
θ̃_mal = θ_global + n × δ_mal       # boosted by n_clients

# After FedAvg:
θ^(t+1) = (θ̃_mal + θ_1 + θ_2) / n ≈ θ_local_mal
```

The n× boost exactly cancels FedAvg's 1/n weighting. The global model becomes a copy of the attacker's local model in **a single round**.

### Results

| Strategy | Peak ASR | Notes |
|---|---|---|
| FedAvg + data poisoning (baseline) | 0.48 | Slow accumulation over 5 rounds |
| FedAvg + model replacement | ~1.00 | Spikes to 1.0 in round 0 |
| Norm clipping (τ=3.0) + model replacement | ~0.10 | Attack suppressed back to baseline |

**Key finding:** Model replacement completely bypasses FedAvg and DP noise, achieving ASR=1.0 in a single round. Norm clipping — which caps each client's update delta to L2 norm ≤ τ — neutralises the attack by bringing the n× boosted malicious delta back to the same scale as honest updates.

---

## Defense Summary

| Defense | Mechanism | Effective Against | Limitation |
|---|---|---|---|
| FedAvg | Coordinate-wise mean | — | Vulnerable to all attacks |
| DP Noise (σ=0.01) | Add Gaussian noise to updates | — | Too weak at this σ; larger σ hurts CA |
| Coordinate Median | Median per parameter | Naive data poisoning | Can be bypassed by coordinated outliers |
| Trimmed Mean | Drop top/bottom α, average rest | Naive data poisoning | Fragile at small n |
| Krum | Select update closest to neighbors | Naive data poisoning | Slower convergence; O(n²) cost |
| Norm Clipping (τ=3.0) | Cap update delta to L2 norm ≤ τ | Model replacement | τ must be tuned; doesn't help against naive poisoning |

---

## Output Files

```
defense_comparison.png          # Exp 1 — all 5 aggregation methods
model_replacement.png           # Exp 5 — model replacement + norm clipping
trigger_demo.png                # Visual: clean vs triggered MNIST digits

results/
  comparison_threshold.png      # Exp 2 — threshold sweep (all fractions)
  threshold_20pct.png           # Exp 2 — individual runs
  threshold_30pct.png
  threshold_40pct.png
  threshold_50pct.png
  comparison_duration.png       # Exp 3 — duration sweep (all durations)
  duration_1rounds.png          # Exp 3 — individual runs
  duration_3rounds.png
  duration_5rounds.png
  duration_10rounds.png
  comparison_noniid_all.png     # Exp 4 — non-IID across all fractions
  noniid_20pct.png              # Exp 4 — individual non-IID runs
  noniid_30pct.png
  noniid_40pct.png
  noniid_50pct.png
  noniid_80pct.png
  comparison_iid_vs_noniid_40pct.png   # Exp 4 — direct IID vs non-IID
  comparison_iid_vs_noniid_50pct.png
  attack_20pct.png              # IID baseline runs at each fraction
  attack_50pct.png
  attack_80pct.png
  baseline_0pct.png
  median_attack_20pct.png       # Median defense runs
  median_attack_50pct.png
  median_attack_80pct.png
  median_baseline_0pct.png
  comparison_median.png         # Median vs FedAvg comparison
  comparison_all.png            # Combined overview
```

---

## Key Findings Summary

1. **Threshold effect:** A sharp transition exists between 30% and 40% malicious clients under FedAvg. Below 30%, the attack fails entirely. Above 40%, the backdoor takes hold and persists.

2. **Duration is non-linear:** 10 attack rounds produces a backdoor ~5× more persistent than 5 rounds (final ASR 0.75 vs 0.14). Each round reinforces the backdoor more deeply across network layers.

3. **Non-IID amplifies persistence:** Real-world heterogeneous data distributions make backdoors 1.4–3.4× more persistent depending on the malicious fraction. Heterogeneous honest updates are less effective at washing out the backdoor.

4. **Robust aggregation helps against naive poisoning:** Coordinate Median and Krum reduce peak ASR from ~0.48 to 0.10–0.19 against data poisoning. DP noise at σ=0.01 provides no benefit.

5. **Model replacement breaks averaging-based defenses:** By scaling the malicious delta by n_clients, the attacker achieves ASR=1.0 in a single round — bypassing FedAvg and DP entirely.

6. **Norm clipping counters model replacement:** Capping update deltas to L2 norm ≤ 3.0 neutralises the amplification and suppresses ASR back to baseline, with no loss in clean accuracy.
