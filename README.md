# Compare Analysis Tool

Statistical Model Checking (SMC) tool for comparing **compositional vs monolithic** approaches to estimating success rates of autonomous driving scenarios.

## Features

- **Parallel trace generation** with hard time budget enforcement
- **Monolithic mode**: Test complete scenarios directly
- **Compositional mode**: Decompose scenarios into primitives and combine results
- **Ground truth computation**: Statistically rigorous estimates using Hoeffding's inequality
- **Hard stop**: Ensures fair comparisons by enforcing strict time limits

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd compositional_analysis

# Install dependencies
pip install -r requirements.txt

# Install VerifAI (if not already installed)
pip install -e /path/to/VerifAI
```

---

## Quick Start

```bash
# Basic monolithic test
python compare_analysis.py --scenario "SXC" --time_budget 30

# Compositional analysis
python compare_analysis.py --scenario "SXC" --compositional --time_budget 30

# Ground truth computation
python compare_analysis.py --scenario "SXC" --ground_truth --confidence_level 0.95 --error_bound 0.01
```

---

## Usage

```bash
python compare_analysis.py [OPTIONS]
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scenario` | str | `"SXC"` | Scenario string (e.g., "SXC", "SXSX", "ABC") |
| `--compositional` | flag | `False` | Use compositional approach |
| `--time_budget` | int | `25` | Time budget in seconds |
| `--n` | int | `None` | Max number of traces (None = run until time budget) |
| `--expert` | flag | `False` | Use expert driving policy |
| `--save_dir` | str | `"storage/run1"` | Directory to save traces |
| `--model_path` | str | `"storage/models/model_map_2.zip"` | Path to model file |
| `--ground_truth` | flag | `False` | Compute ground truth using Hoeffding's inequality |
| `--confidence_level` | float | `0.99` | Confidence level for ground truth (0-1) |
| `--error_bound` | float | `0.001` | Error bound (ε) for ground truth |

---

## Modes of Operation

### 1. Monolithic Mode

Tests the complete scenario directly without decomposition.

```bash
python compare_analysis.py --scenario "SXC" --time_budget 30 --expert
```

**Behavior:**
- Launches one worker process for the complete scenario
- Generates traces until time budget is reached
- Computes success rate ρ and uncertainty ε using all traces

---

### 2. Compositional Mode

Decomposes the scenario into primitives, tests each separately, then combines results using importance sampling.

```bash
python compare_analysis.py --scenario "SXC" --compositional --time_budget 30 --expert
```

**Behavior:**
- Parses "SXC" into primitives: {S, X, C}
- Launches parallel worker processes (one per unique primitive)
- Each worker generates traces until time budget is reached
- Computes individual ρ values for each primitive
- Uses `CompositionalAnalysisEngine` to combine results

---

### 3. Ground Truth Mode

Computes statistically rigorous estimates using Hoeffding's inequality.

```bash
python compare_analysis.py --scenario "SXC" --ground_truth --confidence_level 0.95 --error_bound 0.01
```

**Behavior:**
- Calculates required samples: `n = ln(2/δ) / (2ε²)` where δ = 1 - confidence_level
- Generates exactly n traces (no time limit)
- Guarantees: with probability ≥ confidence_level, estimated ρ is within ε of true value

**Example calculations:**
- 95% confidence, 1% error → ~18,445 samples
- 99% confidence, 0.1% error → ~2,654,126 samples

---

## Examples

### Example 1: Quick Monolithic Test

```bash
python compare_analysis.py --scenario "SXC" --expert --time_budget 20
```

Generates traces for 20 seconds and computes success rate.

---

### Example 2: Compositional Analysis

```bash
python compare_analysis.py --scenario "SXSX" --compositional --expert --time_budget 30
```

Decomposes "SXSX" into {S, X}, generates traces for each primitive in parallel for 30 seconds, then combines results.

---

### Example 3: Fixed Number of Traces

```bash
python compare_analysis.py --scenario "ABC" --n 5000 --time_budget 60 --expert
```

Generates exactly 5000 traces OR stops at 60 seconds, whichever comes first.

---

### Example 4: Run Until Time Budget

```bash
python compare_analysis.py --scenario "XYZ" --time_budget 45 --expert
```

Generates as many traces as possible within 45 seconds (no sample limit).

---

### Example 5: Ground Truth - High Confidence

```bash
python compare_analysis.py --scenario "SXC" --ground_truth --confidence_level 0.99 --error_bound 0.001
```

Generates ~2.65M traces to achieve 99% confidence with 0.1% error bound. No time limit.

---

### Example 6: Ground Truth - Faster

```bash
python compare_analysis.py --scenario "SXC" --ground_truth --confidence_level 0.95 --error_bound 0.01
```

Generates ~18K traces to achieve 95% confidence with 1% error bound.

---

### Example 7: Compositional Ground Truth

```bash
python compare_analysis.py --scenario "SXSX" --compositional --ground_truth --confidence_level 0.95 --error_bound 0.01
```

Computes ground truth for each primitive (S, X) then combines using compositional engine.

---

### Example 8: Custom Paths

```bash
python compare_analysis.py \
  --scenario "CustomScenario" \
  --save_dir "experiments/exp_001" \
  --model_path "models/custom_model.zip" \
  --time_budget 40 \
  --expert
```

---

## Time Budget vs Sample Budget

| Configuration | Time Limit | Sample Limit | Behavior |
|---------------|------------|--------------|----------|
| `--time_budget 30` | 30s | None | Run until 30s, collect as many traces as possible |
| `--n 5000 --time_budget 30` | 30s | 5000 | Stop at 5000 traces OR 30s (whichever first) |
| `--ground_truth` | None (∞) | Computed | Run until required samples collected |

---

## Understanding Output

### Trace Generation

```
=== Generating Traces (Parallel - HARD STOP) ===
Launching scenario S
[PID=12345] Starting scenario S
[HARD STOP] Time budget (30s) reached at 30.02s
Terminating scenario S (PID=12345)
[INFO] Scenario S: Removing partial trace (had 9112, keeping 9059)
[INFO] Scenario S has 9059 completed traces.
```

- **Hard stop**: Processes terminated when time budget reached
- **Partial traces removed**: Incomplete trace being written at termination is discarded
- **Gap (had vs keeping)**: Normal behavior showing traces written during termination

---

### Monolithic Results

```
=== Monolithic SMC Results ===
SXC: rho = 0.8234 ± 0.0042
```

- **ρ (rho)**: Estimated success rate (82.34%)
- **±ε (uncertainty)**: Hoeffding bound (±0.42%)
- **95% confidence interval**: [0.8192, 0.8276]

---

### Compositional Results

```
=== Monolithic SMC Results ===
S: rho = 0.9123 ± 0.0031
X: rho = 0.8456 ± 0.0037

=== Running Compositional SMC ===
Estimated SXSX: rho = 0.7245 ± 0.0089
```

- First section: Individual primitive success rates
- Second section: Combined estimate using importance sampling with Gaussian KDE
- Uncertainty is propagated through composition

---

## How It Works

### Hard Stop Time Budget

1. All worker processes launch simultaneously
2. Main process monitors elapsed time every 100ms
3. When time budget is reached:
   - Snapshot current trace counts
   - Send `SIGTERM` to all workers (5s grace period)
   - Force kill any remaining processes
4. After termination:
   - Read final trace counts
   - Remove partial traces (traces written during termination)
   - Keep only complete traces for analysis

### Hoeffding's Inequality

For ground truth mode, the required number of samples is:

```
n = ln(2/δ) / (2ε²)
```

Where:
- `δ = 1 - confidence_level` (probability of error)
- `ε = error_bound` (maximum error)

This guarantees: `P(|ρ̂ - ρ| ≤ ε) ≥ confidence_level`

---

## Results

### Compositional vs Monolithic Analysis

| Scenario | Mode | Expert | Time Budget (s) | Episodes | Success Rate (ρ) | Uncertainty (±ε) | Notes |
|----------|------|--------|-----------------|----------|------------------|------------------|-------|
| **SXC** | Compositional | ✓ | 2400 | S: 3,151<br>X: 2,261<br>C: 1,087 | **0.9173** | **0.0657** | |
| | Primitives | ✓ | | | S: 0.9984<br>X: 0.9474<br>C: 0.9706 | S: 0.0242<br>X: 0.0286<br>C: 0.0412 | |
| **SXC** | Monolithic | ✓ | 2400 | 516 | **0.9070** | **0.0598** | |
| **XCS** | Compositional | ✓ | 2400 | X: 4,655<br>C: 2,160<br>S: 6,589 | **0.9207** | **0.8468** | |
| | Primitives | ✓ | | | X: 0.9563<br>C: 0.9635<br>S: 0.9989 | X: 0.0199<br>C: 0.0292<br>S: 0.0167 | |
| **XCS** | Compositional | ✓ | - | Reused from SXC | **0.9162** | **0.0644** | Reused traces |
| **SOCXSOCX** | Compositional | ✓ | 2400 | S: 3,177<br>O: 1,078<br>C: 1,102<br>X: 2,225 | **0.4872** | **0.0712** | |
| | Primitives | ✓ | | | S: 0.9984<br>O: 0.7523<br>C: 0.9682<br>X: 0.9528 | S: 0.0241<br>O: 0.0414<br>C: 0.0409<br>X: 0.0288 | |
| **SOCXSOCX** | Monolithic | ✓ | 2400 | 122 | **0.4836** | **0.1230** | |
| **OOOOO** | Compositional | ✓ | 2400 | O: 2,337 | **0.2244** | **0.0230** | |
| | Primitives | ✓ | | | O: 0.7398 | O: 0.0281 | |
| **OOOOO** | Monolithic | ✓ | 2400 | 317 | **0.3344** | **0.0763** | |

**Note:** "Episodes" refers to the number of complete rollouts/trajectories (unique `trace_id` values), not the number of timesteps/rows in the CSV. Each episode contains multiple timesteps.
