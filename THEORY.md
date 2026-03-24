# Theoretical Grounding — v100.0.0

Theoretical foundations of the QEC deterministic adaptive control system.

---

## 1. Deterministic Adaptive Control

QEC implements a deterministic adaptive control system for belief propagation decoding dynamics. Unlike classical adaptive control, which typically relies on stochastic exploration (e.g., ε-greedy, Boltzmann sampling, gradient-based optimization with noise), QEC achieves adaptation through:

- **Deterministic scoring**: strategies are ranked by a closed-form multiplicative score composed of bounded factors.
- **Memory-driven bias**: historical performance records shift strategy preferences without randomness.
- **Signal-guided regulation**: physics-informed signals modulate adaptation strength based on system state.

This design eliminates the exploration-exploitation tradeoff entirely. The system does not explore — it measures, scores, and selects deterministically.

---

## 2. Convergence Without Stochastic Exploration

Classical results in adaptive control and reinforcement learning require stochastic exploration to guarantee convergence to optimal policies. QEC avoids this requirement by operating under different assumptions:

- **Finite, known strategy space**: the set of strategies is fixed and fully enumerable.
- **Observable state**: all regime, attractor, and metric information is directly computed, not estimated.
- **Bounded feedback**: all adaptation signals are bounded, preventing runaway feedback loops.

Convergence in this context means: given sufficient history, strategy scores stabilize and the system selects consistently optimal strategies for each regime. This follows from:

1. Memory is bounded (cap = 10), so old observations are displaced.
2. Bias adjustments are bounded (|bias| ≤ 0.2).
3. All multiplicative factors are bounded away from zero (minimum product > 0).

The system converges to a fixed point in score space for any stationary input distribution.

---

## 3. Bounded Feedback Systems

The scoring function is a product of bounded factors:

```
score = base × stability × transition × multi_step × modulation × cycle_penalty × trajectory
```

Each factor `fᵢ` satisfies `fᵢ ∈ [aᵢ, bᵢ]` where `aᵢ > 0`. Therefore:

```
∏ aᵢ  ≤  score  ≤  ∏ bᵢ
```

This structural bound guarantees:
- No score explosion (unbounded growth)
- No score collapse to zero (all lower bounds > 0)
- Monotonic response to factor changes (product is monotone in each factor)

The geometric mean modulation (v99.8.0) further prevents multiplicative collapse when individual signals approach zero, replacing raw products with `0.5 + (∏ sᵢ)^(1/n)`.

---

## 4. Signal-Guided Regulation

The physics signal layer provides real-time system state information:

| Signal | Measures | Regulation Effect |
|--------|----------|-------------------|
| system_energy | Disorder level | High energy → suppress adaptation |
| phase_stability | Regime consistency | Low stability → dampen changes |
| multiscale_coherence | Cross-scale agreement | Low coherence → reduce confidence |
| control_alignment | Strategy-metric alignment | Low alignment → penalize strategy |
| oscillation_strength | Regime switching frequency | High oscillation → extra damping |

These signals form a feedback channel from the system to the adaptation layer, without modifying the decoder or its inputs. The regulation is purely multiplicative and bounded, ensuring stability.

---

## 5. Trajectory-Level Validation

Beyond per-step evaluation, the system validates trajectories:

- **Transition validation**: scores each state transition on improvement vs. degradation across score, energy, and coherence.
- **Monotonicity constraint**: penalizes trajectories that fail to improve consistently.
- **Strategy consistency**: penalizes rapid flipping between strategies.

These constraints implement a form of trajectory-level Lyapunov stability: the system discourages moves that increase a generalized "energy" function (degradation) while rewarding moves that decrease it (improvement).

---

## 6. Cycle Suppression

Oscillatory traps (e.g., A→B→A→B) are a known failure mode of deterministic control systems. QEC addresses this through:

- **Pattern detection**: scans recent history for period-2 and period-3 repeating patterns.
- **Multiplicative penalty**: detected cycles reduce scores by up to 20%.
- **Stability exemption**: uniform repetition (A→A→A) is not penalized, distinguishing stable convergence from oscillation.

This mechanism prevents the system from entering limit cycles without introducing randomness.

---

## 7. Relation to Classical Results

| Classical Assumption | QEC Alternative |
|---------------------|-----------------|
| Stochastic exploration required for convergence | Deterministic scoring over finite, known strategy space |
| Unbounded learning rates | Bounded bias adjustments (|bias| ≤ 0.2) |
| Infinite horizon discounting | Fixed horizon lookahead (H = 2) |
| Model-free policy gradient | Direct scoring from observable metrics |
| Neural function approximation | Closed-form multiplicative scoring |

QEC trades generality for guarantees: it cannot discover novel strategies, but it provably converges, reproduces exactly, and never degrades unpredictably.

---

## 8. Stability via Bounded Modulation

The adaptation modulation factor is the primary mechanism through which the system self-regulates. Its boundedness ensures:

1. **No runaway amplification**: modulation ≤ 1.5, so adaptation cannot grow unboundedly.
2. **No signal extinction**: modulation ≥ 0.5, so adaptation is never fully suppressed.
3. **Regime sensitivity**: oscillatory states with low phase stability receive additional damping, preventing positive feedback loops.

Combined with bounded memory, bounded bias, and bounded cycle penalties, the system satisfies the sufficient conditions for bounded-input bounded-output (BIBO) stability.
