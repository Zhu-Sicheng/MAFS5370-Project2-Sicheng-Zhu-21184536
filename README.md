# MAFS5370 — Assignment 2 (Super Tic-Tac-Toe)

## Overview
This project implements and trains reinforcement learning agents for **Super Tic-Tac-Toe**, a stochastic variant of tic-tac-toe with a **triangular board** and **probabilistic action execution**.  
The main goal is to build a correct environment and train agents using **Ray RLlib (Torch backend)**, then compare multiple RL algorithms under consistent evaluation settings.

Key contributions:
- **RLlib-based** multi-agent training (Torch-only) with action masking.
- A **two-phase curriculum** concept: **deterministic placement → stochastic placement** (fixed training → random training).
- A faithful **stochastic move execution model** (1/2 accept, otherwise 1/16 per neighbor, with forfeits).
- **Multi-algorithm benchmark** across **PPO, IMPALA, APPO**, including plotting and CSV export.

Notebook:
- `mafs5370-project2-benchmark-rllib-train70Det30Stoch.ipynb`

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
- Otherwise, the executed move is randomly selected from the **8 adjacent neighbors**; each neighbor has probability **1/16** overall (because rejection is 1/2 and neighbor direction is uniform over 8).
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
- The environment uses a 3×3 block layout but only 6 blocks are enabled via a `valid_mask`, forming the triangular board.

Stochastic placement model:
- Controlled by `stochastic_moves` and `accept_prob` (global), and also supports **per-agent overrides**:
  - `stochastic_moves_by_agent`
  - `accept_prob_by_agent`

Forfeits:
- If the executed target is invalid (off-board) or occupied, the step proceeds with **no placement**, matching the assignment’s “forfeited move” rule.

## Training Design: Fixed → Random (Curriculum)
To stabilize early learning and then match the true stochastic dynamics, the notebook is structured to support a **two-stage training curriculum**:

1. **Fixed / deterministic placement stage**
   - Intended setting: `stochastic_moves=False`, `accept_prob=1.0`
   - Rationale: reduces transition noise early and helps the agent learn basic positional structure.

2. **Random / stochastic placement stage**
   - Assignment setting: `stochastic_moves=True`, `accept_prob=0.5`
   - Rationale: trains the agent under the real game dynamics (1/2 accept, neighbor noise, forfeits).

In addition, the notebook supports **asymmetric stochasticity** (useful for analysis), e.g.:
- **player1 stochastic placement** vs **player2 deterministic placement**

---

## Algorithms Compared
The notebook benchmarks multiple RLlib algorithms under the same environment and evaluation loop:
- **PPO**
- **IMPALA**
- **APPO**

For each algorithm, the notebook logs:
- training iteration timing
- mean episode length
- win rates for `player1` / `player2` under different evaluation modes

---

## Evaluation Protocol
Evaluation is performed by running multiple self-play episodes and computing:
- `p1_win`, `p2_win`, `draw`

The notebook can evaluate under:
- **deterministic mode** (fixed placement)
- **stochastic mode** (assignment’s randomness, optionally asymmetric by player)

Plots produced:
- mean episode length vs iteration
- win rates vs iteration

CSV outputs:
- `train_details.csv` (per-evaluation checkpoint rows)
- `train_overall.csv` (final summary per algorithm)

---

<img width="2090" height="490" alt="image" src="https://github.com/user-attachments/assets/d2745969-eac7-419f-bc30-8c4ada351c8b" />


## Figure Interpretation

This figure summarizes the result under the proposed two-phase curriculum: **deterministic placement → stochastic placement**.  
The **red dashed vertical line** marks the iteration where training switches from fixed execution to the assignment’s stochastic execution rule (**1/2** chance the chosen cell is accepted; otherwise a neighbor is selected with **1/16** probability per adjacent cell, and the move may be **forfeited** if invalid/occupied).

### Train Episode Length Mean (Left)
- PPO shows a clear downward trend in mean episode length over training, indicating that games terminate faster as the self-play policies become more decisive and reach terminal states earlier.
- IMPALA and APPO remain comparatively stable, suggesting different optimization/exploration dynamics under the same environment and action-masking setup.

### Win Rates: Deterministic vs Stochastic (Middle/Right)
- The win-rate panels compare performance under **deterministic evaluation** versus **stochastic evaluation** for each algorithm.
- Across algorithms, stochastic evaluation typically yields lower and noisier win rates than deterministic evaluation. This is expected because probabilistic execution and possible forfeits reduce the agent’s control over outcomes and increase variance.
- IMPALA attains the strongest peaks under deterministic evaluation, while stochastic curves are generally lower, highlighting a robustness/generalization gap introduced by the assignment’s stochastic action execution even when training includes a stochastic phase.

### Takeaway
Overall, the plots demonstrate 
(1) how a **fixed → random** curriculum can stabilize early training, and 
(2) how the assignment’s true stochastic dynamics make robust play harder to learn and evaluate, motivating multi-algorithm comparison under consistent RLlib settings.


## How to Run
1. Open the notebook:
   - `mafs5370-project2-benchmark-rllib-train50Det50Stoch.ipynb`
2. Run cells top-to-bottom.
3. Adjust experiment knobs at the top of the notebook (e.g., iterations, evaluation frequency, randomness settings) as needed.

---

## Notes
- The environment includes an optional setting to force a number of **opening random moves** for exploration. This is not part of the original game rules, but it can help avoid overly deterministic openings during training.
- Action masking keeps policies focused on legal moves; forfeits still occur due to stochastic neighbor execution (off-board/occupied outcomes).
