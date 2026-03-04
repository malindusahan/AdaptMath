# Turn-LinTS cold-start audit

Selection-only offline diagnostic. No posterior update, training, reward fitting, or LIVE state was used.

## Data and current prior

- Pre-action contexts: 98 across 15 attempts.
- Deterministic seeds: 256 (25088 selections).
- Context: S+K+L+H+Q, dimension 22.
- Current fresh prior: A=lambda I, b=0, lambda=1.0, exploration=0.20.

## P0 — current zero-mean direct LinTS

MD7 top-1 agreement: **0.2457**.
Telling when MD7 was not telling: **0.2522**.

| selected arm | rate |
|---|---:|
| generic | 0.2501 |
| probing | 0.2485 |
| focus | 0.2492 |
| telling | 0.2522 |

Transition row rates (MD7 top-1 → fresh P0 selection):

| MD7 | generic | probing | focus | telling |
|---|---:|---:|---:|---:|
| generic | 0.2460 | 0.2483 | 0.2532 | 0.2525 |
| probing | 0.2538 | 0.2460 | 0.2501 | 0.2501 |
| focus | 0.2498 | 0.2511 | 0.2451 | 0.2540 |
| telling | NA | NA | NA | NA |

Because all four arms have identical zero-mean covariance, selector probabilities in x do not create an initial preference by themselves. P0 is therefore effectively near-uniform at cold start.

## P1 — additive log-selector action prior

Score: `TS_reward_score_a + gamma * log(max(p_MD7(a), 1e-12))`.

| gamma | MD7 agreement | telling when MD7 not telling |
|---:|---:|---:|
| 0.0 | 0.2457 | 0.2522 |
| 0.01 | 0.2589 | 0.2453 |
| 0.025 | 0.2811 | 0.2337 |
| 0.05 | 0.3176 | 0.2140 |
| 0.1 | 0.3904 | 0.1762 |
| 0.2 | 0.5006 | 0.1252 |
| 0.4 | 0.6116 | 0.0794 |

The prior is finite, so a learned reward-score advantage can override it once the advantage exceeds the logged selector-prior margin. Gamma is therefore a prior-strength parameter, not an eligibility gate, but a fixed gamma does not decay automatically.

## P2 — Bayesian coefficient-mean prior

Use `A0=lambda I`, `m0_a=kappa*e(selector_p_a)`, and `b0_a=A0*m0_a`. This is a Gaussian parameter prior whose initial mean score is `kappa*p_MD7(a)`; it does not create pseudo reward observations.

| kappa | MD7 agreement | telling when MD7 not telling |
|---:|---:|---:|
| 0.0 | 0.2457 | 0.2522 |
| 0.05 | 0.2585 | 0.2472 |
| 0.1 | 0.2716 | 0.2425 |
| 0.2 | 0.3003 | 0.2314 |
| 0.5 | 0.3866 | 0.1964 |
| 1.0 | 0.5165 | 0.1456 |

P2 is mathematically coherent with Bayesian linear regression and its influence naturally dilutes as A accumulates real observations. It still encodes a subjective reward-model prior scale, so it is not scientifically selected here and is not implemented in production.

## Engineering conclusion

A selector-centered cold-start prior is necessary before LIVE if initial behavior is expected to remain near MD7 rather than random across four arms. P1 is the smallest operational formulation; P2 is the more internally Bayesian formulation. Neither is activated. A separate prior-specification decision and offline verification are required before LIVE.

Context remains **INCONCLUSIVE**. This diagnostic does not rerun or supersede the context ablation.
