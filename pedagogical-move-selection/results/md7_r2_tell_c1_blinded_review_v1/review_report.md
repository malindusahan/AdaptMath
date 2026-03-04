# MD7-R2-TELL-C1 Blinded Expert Review Analysis

Frozen review-set SHA-256: `58466681a7e40724f6b4c3acea28ac26c548a771872b9ed72d938dbe84353309`. Reviewed cases: **24**.

## Model agreement with frozen expert judgments

| model | exact matches | agreement rate | Macro-F1 | telling precision | telling recall |
| --- | --- | --- | --- | --- | --- |
| MD7-R1 | 11 | 0.458333 | 0.382895 | 0.545455 | 0.750000 |
| MD7-R2-TELL-C1_epoch2 | 14 | 0.583333 | 0.455433 | 0.615385 | 1.000000 |

## Per-class F1

| move | MD7-R1 | MD7-R2-TELL-C1 Epoch 2 |
| --- | --- | --- |
| generic | 0.000000 | 0.000000 |
| probing | 0.400000 | 0.444444 |
| focus | 0.500000 | 0.615385 |
| telling | 0.631579 | 0.761905 |

## Disagreement wins

- Recalibrated wins: **3**
- Reference wins: **0**
- Both wrong: **2**
- Decisive disagreements: **3**
- Two-sided exact binomial p-value: **0.25**

Raw counts are primary. This small diagnostic does not establish production effectiveness or causal learning benefit.

## Telling-specific strata

| stratum group | reviewed n | expert telling | MD7-R1 telling | MD7-R2 telling |
| --- | --- | --- | --- | --- |
| G/H correct or recovering progress | 5 | 0 | 4 | 3 |
| C one-scaffold recovery | 4 | 0 | 1 | 2 |
| D/E/F persistent difficulty | 8 | 8 | 6 | 8 |

Expert-telling cases: **8**; MD7-R1 recall **0.750000**, recalibrated recall **1.000000**.

Expert-non-telling cases: **16**; MD7-R1 false telling **5**, recalibrated false telling **5**.

## Preregistered decision

**D. REVIEW INCONCLUSIVE**

No outcome authorizes production promotion.
