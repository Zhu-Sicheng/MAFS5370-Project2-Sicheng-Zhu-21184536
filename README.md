# MAFS5370 — Assignment 2 (Super Tic-Tac-Toe)

## Overview
This project implements and trains reinforcement learning agents for **Super Tic-Tac-Toe**, a stochastic variant of tic-tac-toe with a **triangular board** and **probabilistic action execution**. The main goals are:
- Build a correct environment that matches the assignment rules.
- Train agents using **Ray RLlib (Torch backend)** under consistent settings.
- Benchmark multiple algorithms, then improve learning stability under **sparse terminal rewards**.

Key contributions:
- **RLlib-based** multi-agent self-play with **action masking** (legal-move constraints).
- A **two-phase curriculum**: **deterministic placement → stochastic placement** (fixed training → random training).
- A faithful **stochastic move execution model**: **1/2** accept the chosen cell; otherwise a neighbor is selected with **1/16** probability per adjacent cell; invalid/occupied targets lead to **forfeits**.
- A **baseline benchmark** across **PPO, IMPALA, APPO** (same environment + evaluation protocol).
- A follow-up **reward shaping** variant to address **reward sparsity**, trained with **PPO** due to limited compute/time.

Notebooks:
- Baseline benchmark (multi-algorithm): `mafs5370-project2-benchmark-rllib-train70Det30Stoch.ipynb`
- Reward shaping (PPO-only): `mafs5370-project2-ppo-shaping.ipynb`

---

## Game Definition (Assignment Spec)
Board and turns:
- The board is a **triangle** made of **6 squares**, each of size **4×4**.
- Players alternate turns and choose an **empty cell** to place their mark.

Win conditions:
- **4 in a row** (horizontal), or **4 in a column** (vertical), or **5 across a diagonal**.
- For a **vertical 4-in-a-column win**, at least one of the 4 marks must be on a **different level/block** (i.e., spans multiple block rows).

Stochastic execution (core randomness):
- When a player chooses a cell, with probability **1/2** the mark is placed at the chosen cell.
- Otherwise, the executed move is randomly selected from the **8 adjacent neighbors**; each neighbor has probability **1/16** overall (rejection is 1/2 and neighbor direction is uniform over 8).
- If the random neighbor is **outside the board** or **occupied**, the move is **forfeited** (no placement occurs).

---

## Environment Implementation (RLlib Multi-Agent)
The environment is implemented as an RLlib `MultiAgentEnv` with two agents:
- `player1`
- `player2`

State and action:
- **Action space**: `Discrete(96)` (6 blocks × 16 cells per block).
- **Observation**: flattened board + an **action mask** to prevent choosing occupied cells.

Triangular board encoding:
- A 3×3 block layout is used internally, but only 6 blocks are enabled via a `valid_mask`, forming the triangular board.

Forfeits:
- If the executed target is invalid (off-board) or occupied, the step proceeds with **no placement**, matching the assignment’s “forfeited move” rule.

---

## Training Design: Fixed → Random (Curriculum)
To stabilize early learning and then match the true stochastic dynamics, training follows a two-stage curriculum:

1. **Fixed / deterministic placement stage**
   - Setting: `stochastic_moves=False`, `accept_prob=1.0`
   - Rationale: reduces transition noise and helps agents learn basic positional structure.

2. **Random / stochastic placement stage**
   - Setting: `stochastic_moves=True`, `accept_prob=0.5`
   - Rationale: trains under the real game dynamics (accept/reject, neighbor noise, forfeits).

---

## Opening Random Moves (Exploration Heuristic)
Both notebooks include an optional exploration heuristic: forcing the first *N* actions of each episode to be random valid moves (`opening_random_moves`).

- Baseline benchmark uses a larger value (e.g., `OPENING_RANDOM_MOVES_TRAIN = 25`).
- The shaping notebook uses a smaller value (e.g., `OPENING_RANDOM_MOVES_TRAIN = 5`) to keep more steps policy-controlled while still avoiding identical openings.

This mechanism is **not part of the original assignment rules**. In our analysis we focus on **win rates under the curriculum settings**, and note that forced opening randomness is applied symmetrically within each run (both agents experience the same randomized opening phase). In addition, when using an asymmetric setup (stochastic execution vs deterministic execution), the dominant effect comes from the execution noise/forfeits rather than small differences in opening random length, so the win-rate trends remain interpretable.

---

## Baseline: Multi-Algorithm Benchmark (Sparse Terminal Rewards)
Notebook: `mafs5370-project2-benchmark-rllib-train70Det30Stoch.ipynb`

Algorithms compared under the same environment and evaluation loop:
- **PPO**
- **IMPALA**
- **APPO**

What is logged:
- Training iteration timing
- Mean episode length
- Win rates for `player1` / `player2` under **deterministic** vs **stochastic** evaluation modes

Motivation:
- This baseline uses the standard sparse terminal reward structure (**win/loss/draw**), which reflects the assignment objective directly but can be challenging to learn under stochastic execution.

---

## Improvement: Reward Shaping (PPO-Only)
Notebook: `mafs5370-project2-ppo-shaping.ipynb`

Why reward shaping:
- In this game, rewards are naturally **sparse** (mostly at terminal states), while action execution is **stochastic** and can lead to **forfeits**. This combination increases variance and makes policy learning unstable.
- Reward shaping introduces a denser learning signal to encourage progress toward winning configurations.

What is added:
- A **shaping reward** based on **incremental pattern formation**, computed per move for the acting player:
  - Newly formed **3-in-a-row** and **4-in-a-row** patterns contribute additional reward (delta-based, not state-based).
- Anti-exploitation safeguards to prevent “farming” shaping reward instead of winning:
  - A global **scaling factor** for shaping rewards
  - A small **step penalty** to discourage unnecessary prolonging of games
  - A per-episode **cap** on total shaping reward

Why PPO only:
- Due to limited compute/time budget, the shaping experiment focuses on **PPO** to validate the effectiveness of denser rewards without running a full multi-algorithm sweep again.

---

## Evaluation Protocol
Evaluation is performed by running multiple self-play episodes and computing:
- `p1_win`, `p2_win`, `draw`

Evaluation modes:
- **Deterministic mode** (fixed placement)
- **Stochastic mode** (assignment’s probabilistic execution)

Plots produced:
- Mean episode length vs iteration
- Win rates vs iteration

CSV outputs (baseline notebook):
- `train_details_0507.csv` (per-evaluation checkpoint rows)
- `train_overall_0507.csv` (final summary per algorithm)

---

## Figure Interpretation

<img width="2090" height="490" alt="benchmark-plot" src="https://github.com/user-attachments/assets/d2745969-eac7-419f-bc30-8c4ada351c8b" />

This figure summarizes the benchmark results under the two-phase curriculum: **deterministic placement → stochastic placement**.

- **Train Episode Length Mean (Left)**: PPO exhibits a clear downward trend in mean episode length, suggesting that the self-play policies become more decisive and reach terminal outcomes faster. IMPALA and APPO remain comparatively stable, reflecting different learning dynamics under the same environment and action masking.
- **Win Rates (Middle/Right)**: Stochastic evaluation typically yields lower and noisier win rates than deterministic evaluation because probabilistic execution and forfeits reduce control and increase variance. IMPALA shows strong peaks in deterministic evaluation, while stochastic performance is generally lower, highlighting the robustness gap introduced by the assignment’s stochastic dynamics.

---

<img width="2090" height="490" alt="957d73c1-68bb-4cd1-8bda-d5c931f2ce39" src="https://github.com/user-attachments/assets/029edd29-2c5a-44b6-bf8d-66f0948619f2" />


## PPO Reward Shaping Result (Additional Analysis)
The shaping variant (`mafs5370-project2-ppo-shaping.ipynb`) keeps the same RLlib self-play setup and curriculum, but adds a denser learning signal via incremental **3-in-a-row / 4-in-a-row** shaping rewards with safeguards (scaling, per-step penalty, per-episode cap).

Observed behavior from the logged checkpoints:
- **Faster game resolution**: the mean episode length drops quickly from ~41 steps to ~12–14 steps and then stabilizes, indicating more decisive trajectories.
- **Win-rate levels**: `stoch_p1` stays around ~0.39 on average and `det_p1` around ~0.33 on average across checkpoints (in this run), suggesting the policy exhibits slightly better performance under stochastic evaluation than deterministic evaluation.

Interpretation:
- The shaping reward reduces the sparsity of feedback and can speed up the emergence of structured play (reflected by shorter episodes).
- Due to stochastic execution and forfeits, win rates can remain noisy and below 0.5 even when policies become more decisive; the key signal is whether the stochastic-mode performance becomes more stable and avoids collapse.

---

## How to Run
1. Open a notebook:
   - Baseline benchmark: `mafs5370-project2-benchmark-rllib-train70Det30Stoch.ipynb`
   - Reward shaping PPO: `mafs5370-project2-ppo-shaping.ipynb`
2. Run cells top-to-bottom.
3. Adjust experiment knobs at the top (iterations, evaluation frequency, curriculum fraction, etc.) if needed.
