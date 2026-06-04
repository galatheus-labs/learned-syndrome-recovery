"""Surface code neural decoder prototype.

Demonstrates that the parity diffusion architecture (syndrome → neural decode)
applies directly to quantum error correction.

Uses Stim for circuit simulation, PyMatching for MWPM baseline,
and our ResNet/parity infrastructure for neural decoding.

Code-capacity noise model (simplest): i.i.d. bit flips on data qubits,
perfect syndrome measurement. This reduces to a classical syndrome
decoding problem — exactly what our architecture solves.

Usage:
    python surface_code.py --distance 5 --error-rate 0.05 --gen-data 100000
    python surface_code.py --distance 5 --error-rate 0.05 --train --epochs 20
    python surface_code.py --distance 5 --benchmark
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import stim
import pymatching


# ---------------------------------------------------------------------------
# Data generation: surface code syndromes + error patterns
# ---------------------------------------------------------------------------

def make_code_capacity_check_matrix(distance):
    """Extract the X-error parity check matrix for a rotated surface code.

    In code-capacity noise, X and Z errors decouple. We decode X errors
    using Z stabilizers (and vice versa). This gives us a classical
    binary linear code to decode.

    Returns:
        H: (m, n) uint8 parity check matrix
        n: number of data qubits
        m: number of stabilizers
    """
    # Generate circuit just to extract the check matrix structure
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=1,
        distance=distance,
        after_clifford_depolarization=0,  # no noise for structure extraction
    )

    # The detector error model tells us which errors trigger which detectors
    # For code-capacity, we directly build H from the stabilizer structure
    n = distance * distance  # data qubits
    m = n - 1  # stabilizers (for [[n,1,d]] code)

    # Use stim's tableau to get stabilizers
    # Simpler approach: build from the rotated surface code geometry
    H = _build_rotated_surface_code_Hz(distance)
    return H, n, H.shape[0]


def _build_rotated_surface_code_Hz(d):
    """Build Z-stabilizer matrix for rotated surface code distance d.

    Data qubits on a d×d grid. Z stabilizers on a checkerboard pattern.
    Bulk stabilizers are weight-4, boundary stabilizers are weight-2.
    """
    n = d * d
    stabilizers = []

    for r in range(d - 1):
        for c in range(d - 1):
            # Checkerboard: Z stabilizers on cells where (r+c) is even
            # (or odd — convention varies; we pick one)
            if (r + c) % 2 == 0:
                row = np.zeros(n, dtype=np.uint8)
                # Four corners of this face
                row[r * d + c] = 1
                row[r * d + (c + 1)] = 1
                row[(r + 1) * d + c] = 1
                row[(r + 1) * d + (c + 1)] = 1
                stabilizers.append(row)

    # Boundary stabilizers (weight-2) on the top and bottom edges
    for c in range(0, d - 1, 2):
        # Top boundary
        row = np.zeros(n, dtype=np.uint8)
        row[c] = 1
        row[c + 1] = 1
        stabilizers.append(row)

    for c in range(1, d - 1, 2):
        # Bottom boundary
        row = np.zeros(n, dtype=np.uint8)
        row[(d - 1) * d + c] = 1
        row[(d - 1) * d + c + 1] = 1
        stabilizers.append(row)

    H = np.array(stabilizers, dtype=np.uint8)
    return H


def generate_code_capacity_data(distance, error_rate, n_samples):
    """Generate (error_pattern, syndrome) pairs for code-capacity noise.

    Each data qubit independently flips with probability error_rate.
    Syndrome = H @ error mod 2.

    Returns:
        errors: (n_samples, n) uint8
        syndromes: (n_samples, m) uint8
    """
    H, n, m = make_code_capacity_check_matrix(distance)
    H_t = torch.from_numpy(H)

    errors = (np.random.random((n_samples, n)) < error_rate).astype(np.uint8)
    errors_t = torch.from_numpy(errors)

    # Syndrome = H @ e mod 2
    syndromes = (H_t.float() @ errors_t.float().t()).t() % 2
    syndromes = syndromes.to(torch.uint8).numpy()

    return errors, syndromes, H


def generate_stim_data(distance, error_rate, n_samples, noise_model='code_capacity'):
    """Generate data using Stim for more realistic noise models.

    For code-capacity: equivalent to generate_code_capacity_data but uses
    Stim's infrastructure. For circuit-level: includes measurement errors.
    """
    if noise_model == 'code_capacity':
        # Code capacity: only data qubit errors, 1 round, perfect measurement
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=1,
            distance=distance,
            before_round_data_depolarization=error_rate,
        )
    elif noise_model == 'phenomenological':
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=distance,
            distance=distance,
            before_round_data_depolarization=error_rate,
            before_measure_flip_probability=error_rate,
        )
    else:  # circuit-level
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            rounds=distance,
            distance=distance,
            after_clifford_depolarization=error_rate,
            after_reset_flip_probability=error_rate,
            before_measure_flip_probability=error_rate,
            before_round_data_depolarization=error_rate,
        )

    sampler = circuit.compile_detector_sampler()
    det_events, obs_flips = sampler.sample(shots=n_samples, separate_observables=True)

    return det_events, obs_flips, circuit


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SurfaceCodeDataset(Dataset):
    """Dataset of (syndrome, error_pattern) pairs."""

    def __init__(self, syndromes, errors):
        self.syndromes = torch.from_numpy(syndromes).float()
        self.errors = torch.from_numpy(errors).float()

    def __len__(self):
        return len(self.syndromes)

    def __getitem__(self, idx):
        return self.syndromes[idx], self.errors[idx]


# ---------------------------------------------------------------------------
# Neural decoder (simple MLP — proof of concept)
# ---------------------------------------------------------------------------

class SyndromeDecoder(nn.Module):
    """MLP decoder: syndrome → error pattern prediction.

    Simple feedforward network. For a proper comparison we'd use our
    full ResNet architecture, but this proves the concept.
    """

    def __init__(self, syndrome_dim, n_qubits, hidden_dim=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(syndrome_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, n_qubits))
        self.net = nn.Sequential(*layers)

    def forward(self, syndrome):
        return self.net(syndrome)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_decoder(distance, error_rate, epochs=20, n_train=100000, n_val=10000,
                  hidden_dim=256, n_layers=4, batch_size=256, lr=1e-3):
    """Train a neural syndrome decoder and compare to MWPM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Surface code d={distance}, p={error_rate}")
    print(f"Generating {n_train} training + {n_val} validation samples...")

    # Generate data
    train_err, train_syn, H = generate_code_capacity_data(
        distance, error_rate, n_train)
    val_err, val_syn, _ = generate_code_capacity_data(
        distance, error_rate, n_val)

    n_qubits = distance * distance
    n_checks = H.shape[0]
    print(f"  n={n_qubits} qubits, m={n_checks} checks")

    # Datasets
    train_ds = SurfaceCodeDataset(train_syn, train_err)
    val_ds = SurfaceCodeDataset(val_syn, val_err)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = SyndromeDecoder(n_checks, n_qubits, hidden_dim, n_layers).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {params:,} params, hidden_dim={hidden_dim}, layers={n_layers}")

    opt = optim.AdamW(model.parameters(), lr=lr)
    H_t = torch.from_numpy(H).to(device)

    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for syn, err in train_loader:
            syn, err = syn.to(device), err.to(device)
            logits = model(syn)
            loss = F.binary_cross_entropy_with_logits(logits, err)

            # Parity consistency loss
            pred_soft = torch.sigmoid(logits)
            pred_syndrome = (H_t.float() @ pred_soft.t()).t()
            target_syndrome = syn
            parity_loss = F.mse_loss(pred_syndrome % 1, target_syndrome % 1)
            loss = loss + 0.5 * parity_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        correct_neural = 0
        total = 0
        with torch.no_grad():
            for syn, err in val_loader:
                syn, err = syn.to(device), err.to(device)
                logits = model(syn)
                pred = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
                # Check if predicted error has same syndrome as actual error
                # (equivalent error = same coset = logical success)
                pred_syn = (H_t.float() @ pred.float().t()).t() % 2
                actual_syn = (H_t.float() @ err.float().t()).t() % 2
                diff = (pred.float() - err).abs()
                # Logical error: pred XOR actual is a non-trivial logical operator
                # For simplicity, check exact match (conservative)
                exact_match = (pred == err.to(torch.uint8)).all(dim=1)
                correct_neural += exact_match.sum().item()
                total += len(syn)

        val_acc = correct_neural / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, "
              f"val_exact_match={val_acc:.4f}")

    return model, H, best_val_acc


# ---------------------------------------------------------------------------
# Benchmark: Neural vs MWPM
# ---------------------------------------------------------------------------

def benchmark(distance, error_rates=None, n_samples=50000, model_path=None):
    """Compare neural decoder vs MWPM across error rates."""
    raise NotImplementedError(
        "surface_code.py benchmark had placeholder neural results. "
        "Use mwpm_baseline() for MWPM-only numbers, or use "
        "surface_code_resnet.py + eval_test_split.py for neural-vs-MWPM evaluation."
    )

    if error_rates is None:
        error_rates = [0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.09, 0.10]

    print(f"\n{'='*60}")
    print(f"Surface Code d={distance} Benchmark")
    print(f"{'='*60}")
    print(f"{'p':>6} {'MWPM_err':>10} {'Neural_err':>10} {'Improvement':>12}")
    print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*12}")

    results = []
    for p in error_rates:
        # Generate test data using Stim (more standard)
        det_events, obs_flips, circuit = generate_stim_data(
            distance, p, n_samples, 'code_capacity')

        # MWPM baseline
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        mwpm_pred = matcher.decode_batch(det_events)
        mwpm_errors = (mwpm_pred != obs_flips).sum()
        mwpm_rate = mwpm_errors / n_samples

        # Neural decoder (if we have one)
        if model_path and Path(model_path).exists():
            # TODO: load and run neural decoder
            neural_rate = -1
        else:
            # Train a quick model for this error rate
            model, H, _ = train_decoder(distance, p, epochs=10,
                                         n_train=50000, n_val=5000,
                                         hidden_dim=128, n_layers=3)
            # Evaluate using Stim data
            device = next(model.parameters()).device
            H_np, _, _ = make_code_capacity_check_matrix(distance)
            H_t = torch.from_numpy(H_np).float().to(device)

            # Convert Stim detector events to our syndrome format
            # For code-capacity with 1 round, detectors ≈ syndromes
            syn_t = torch.from_numpy(det_events.astype(np.float32)).to(device)
            with torch.no_grad():
                # Process in batches
                neural_correct = 0
                bs = 1000
                for i in range(0, len(det_events), bs):
                    batch_syn = syn_t[i:i+bs]
                    batch_obs = obs_flips[i:i+bs]
                    logits = model(batch_syn)
                    pred = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
                    # Check logical equivalence via observable
                    # This is approximate — proper check needs logical operator
                    neural_correct += len(batch_syn)  # placeholder
                neural_rate = mwpm_rate  # placeholder until proper eval

        improvement = (mwpm_rate - neural_rate) / mwpm_rate * 100 if mwpm_rate > 0 else 0
        print(f"{p:>6.3f} {mwpm_rate:>10.4f} {neural_rate:>10.4f} {improvement:>+11.1f}%")
        results.append({
            'error_rate': p,
            'mwpm_logical_error_rate': float(mwpm_rate),
            'neural_logical_error_rate': float(neural_rate),
        })

    return results


# ---------------------------------------------------------------------------
# Quick demo: MWPM baseline across distances and error rates
# ---------------------------------------------------------------------------

def mwpm_baseline(distances=None, error_rates=None, n_samples=100000):
    """Run MWPM baseline to establish the numbers to beat."""
    if distances is None:
        distances = [3, 5, 7]
    if error_rates is None:
        error_rates = [0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.11]

    print(f"\n{'='*60}")
    print(f"MWPM Baseline (code-capacity noise, {n_samples:,} shots)")
    print(f"{'='*60}")

    header = f"{'p':>6}"
    for d in distances:
        header += f" {'d='+str(d):>10}"
    print(header)
    print("-" * len(header))

    all_results = {}
    for p in error_rates:
        row = f"{p:>6.3f}"
        for d in distances:
            det_events, obs_flips, circuit = generate_stim_data(
                d, p, n_samples, 'code_capacity')
            dem = circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            predictions = matcher.decode_batch(det_events)
            errors = (predictions != obs_flips).sum()
            rate = errors / n_samples
            row += f" {rate:>10.4f}"
            all_results[(d, p)] = float(rate)
        print(row)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--gen-data", type=int, default=0,
                        help="Generate N training samples and save")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--mwpm-baseline", action="store_true")
    args = parser.parse_args()

    if args.mwpm_baseline:
        results = mwpm_baseline()

    elif args.train:
        model, H, best_acc = train_decoder(
            args.distance, args.error_rate, args.epochs)
        print(f"\nBest exact-match accuracy: {best_acc:.4f}")

    elif args.benchmark:
        results = benchmark(args.distance)
        with open(f"surface_code_d{args.distance}_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to surface_code_d{args.distance}_results.json")

    elif args.gen_data > 0:
        errors, syndromes, H = generate_code_capacity_data(
            args.distance, args.error_rate, args.gen_data)
        out_dir = Path(f"data/surface_code_d{args.distance}")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "errors.npy", errors)
        np.save(out_dir / "syndromes.npy", syndromes)
        np.save(out_dir / "check_matrix.npy", H)
        print(f"Saved {args.gen_data} samples to {out_dir}")
        print(f"  errors: {errors.shape}, syndromes: {syndromes.shape}, H: {H.shape}")

    else:
        # Default: run MWPM baseline
        results = mwpm_baseline(distances=[3, 5, 7], n_samples=100000)
