# Student Personalization Memory Layer
## Project Development Log

### Project Goal

Build a Student Personalization Memory and Learning-State Identification component for an Adaptive Mathematics Tutor.

The component is designed to:

- Store and retrieve student interaction history.
- Maintain overall student behavioural history.
- Maintain concept-specific student history.
- Represent recent and longer-term learning evidence.
- Identify important learning signals from student interactions.
- Estimate the student's current learning state using historical evidence.
- Support the following target learning states:
  - Struggling
  - Needs Support
  - Improving
  - Stable
  - Strong
- Preserve evidence strength and cold-start information when student history is limited.
- Distinguish overall student performance from concept-specific performance.
- Provide structured student context and learning-state information to other research components.
- Store student information and state snapshots using SQLite.
- Expose the component through FastAPI for integration with the full research system.

The current development focus is the Student Learning State Identification pipeline.

The ASSISTments Skill Builder dataset is being used to develop and validate the interaction-processing, historical feature-engineering, and learning-state methodology.

The intended five learning-state labels are not directly provided by the ASSISTments dataset. Therefore, a defensible state-definition and target-label development methodology must be completed before supervised model training.

---

## Phase 1 — Project Setup

### Steps 1–7: Development Environment

- Created the project folder: `student_personalization_memory`
- Opened the project in VS Code.
- Verified Python 3.11.0.
- Created `.venv`.
- Activated the virtual environment.
- Configured VS Code to use the `.venv` Python interpreter.

### Steps 8–12: Project Structure

Created the main folders:

- `src/`
- `data/`
- `database/`
- `tests/`
- `docs/`
- `artifacts/`
- `notebooks/`

Created:

- `requirements.txt`
- `README.md`
- `.gitignore`

### Steps 13–14: Dependencies

Added and installed the initial Python dependencies for:

- FastAPI backend
- Data processing
- Machine learning
- Topic classification
- Testing
- Jupyter notebooks

### Steps 15–29: Initial Code Structure

Created the initial modules for:

- API routes
- Database connection
- Schemas
- Memory service
- Signal service
- Student learning-state service
- Context builder
- Topic classifier
- Learning-state model
- Data preprocessing
- Tests
- Dataset exploration notebook

---

## Phase 2 — Dataset Understanding

### Dataset

File:

`data/raw/skill_builder_data.csv`

The raw dataset is preserved without modification.

### Initial Dataset Size

- Rows: 525,534
- Columns: 30
- Unique students: 4,217
- Unique assignments: 3,521
- Unique problems: 26,688
- Unique skill IDs: 123
- Unique skill names: 110

### Important Available Features

Examples:

- `user_id`
- `assignment_id`
- `problem_id`
- `correct`
- `attempt_count`
- `ms_first_response`
- `skill_id`
- `skill_name`
- `hint_count`
- `hint_total`
- `opportunity`
- `first_action`

### Important Findings So Far

#### Missing Skill Information

- `skill_id` missing in 66,326 raw rows.
- `skill_name` missing in 78,690 raw rows.
- 12,364 rows have a skill ID but no skill name.
- 66,326 rows have both skill ID and skill name missing.

The missing skill names cannot currently be reliably recovered from the available mappings.

#### Correctness

- Incorrect: 168,449 (32.05%)
- Correct: 357,085 (67.95%)

#### Attempts

Extreme attempt values exist.

- Maximum attempt count: 3,824
- Attempts > 10: 2,807
- Attempts > 50: 449
- Attempts > 100: 148

These values require investigation during preprocessing.

#### Response Time

Invalid and extreme response-time values exist.

- Negative response times: 8
- Zero response times: 4,678
- Positive response times: 520,848
- Response times > 5 minutes: 6,544
- Response times > 1 hour: 200

These values require a justified preprocessing strategy.

#### Duplicate Structure

There are no exact duplicate rows.

However, repeated records exist because of the dataset's opportunity structure.

- Original rows: 525,534
- Unique `order_id + skill_id` combinations: 401,756
- Extra opportunity-related repeated rows: 123,778

One `order_id` can also represent multiple skills, with up to 4 skills observed for one order ID.

Therefore, simply keeping one row per `order_id` would incorrectly remove valid concept information.

#### Missing Concept Information After Interaction-Level Deduplication

After considering unique `order_id + skill_id` combinations:

- 63,755 unique interactions have no skill ID.
- These interactions also have no skill name.

These may still be useful for general student behaviour but cannot directly contribute to Concept-Based Memory.

#### Student History

- Median interactions per student: 27
- 441 students have fewer than 5 interactions.
- 973 students have fewer than 10 interactions.
- 1,519 students have at least 50 interactions.

This must be considered when predicting learning states because students with very little history may not support confident predictions.

#### Assignment History

`assignment_id` is currently being investigated as a proxy session identifier.

- Median assignments per student: 5
- 681 students have only 1 assignment.
- 2,161 students have at least 5 assignments.
- 1,296 students have at least 10 assignments.

It will be documented as a proxy rather than treated as a true session ID without evidence.

#### Question Text Limitation

`answer_text` mainly contains student answers such as numbers.

The current ASSISTments CSV does not appear to provide the original natural-language mathematics question text needed to train the Topic Classifier.

Therefore:

- ASSISTments will support memory and learning-state research.
- A separate suitable question-text dataset/source will be required for training the Topic Classifier.

#### first_action

Observed distribution:

- 0: 479,427 (91.23%)
- 1: 37,419 (7.12%)
- 2: 8,688 (1.65%)

The exact meaning of these values must be verified before using this feature.

---

## Current Status

The project has progressed beyond raw dataset exploration.

Completed development phases:

- Phase 1 — Project Setup ✅
- Phase 2 — Dataset Understanding ✅
- Phase 3 — Data Preprocessing and Interaction Construction ✅
- Phase 4 — Historical Feature Engineering ✅

The original raw dataset remains unchanged:

`data/raw/skill_builder_data.csv`

Validated processed datasets have been created under:

`data/processed/`

These include:

- `base_interactions.csv`
- `concept_interactions.csv`
- `learning_state_features.csv`

The current completed output of Phase 4 is a leakage-safe historical feature dataset containing:

- 338,001 student-concept interaction rows
- 23 total columns
- 16 validated historical candidate predictors
- 4,163 students
- 123 skills

The next development phase is learning-state target and label methodology development for:

- Struggling
- Needs Support
- Improving
- Stable
- Strong

---

## Phase 2 — Dataset Understanding Continued

### Detailed Exploration Notebook

All detailed exploratory analysis, code, outputs, and interpretations are maintained in:

`notebooks/01_dataset_exploration.ipynb`

The notebook is the reproducible technical record of the dataset investigation.

This development log records the major findings and design decisions resulting from that investigation.

---

### Behaviour Feature Investigation

#### `first_action`

Distribution:

- `0`: 479,427 rows (91.23%)
- `1`: 37,419 rows (7.12%)
- `2`: 8,688 rows (1.65%)

Observed relationships:

- All correct responses occur with `first_action = 0`.
- `first_action = 1` is associated with hint usage in the observed dataset.
- `first_action = 2` is associated with incorrect responses and no recorded hint usage.

However, these relationships do not establish the official meaning of the codes.

Decision:

`first_action` will not be used as a learning-state feature until its semantics are verified.

---

### Hint Feature Investigation

The dataset contains:

- `hint_count`
- `hint_total`

Observed:

- `hint_count` never exceeds `hint_total`.
- 198,308 rows have equal values.
- 327,226 rows have `hint_count < hint_total`.

Current interpretation:

`hint_count` appears to represent actual hint usage, while `hint_total` appears related to the number of hints available.

This interpretation must still be verified against dataset semantics.

For learning-state modelling, `hint_count` currently appears more directly relevant to student behaviour.

---

### Zero Attempt Counts

There are:

- 22,153 rows with `attempt_count = 0`
- 4.22% of the raw dataset

These rows contain:

- Correct responses
- Incorrect responses
- Different `first_action` values
- Mostly zero hint usage, but some hint usage

Decision:

`attempt_count = 0` must not automatically be considered invalid or deleted.

Its semantics must be considered during preprocessing.

---

### Extreme Attempt Counts

Extreme attempt counts remain present even after accounting for opportunity-related repeated rows.

Examples include:

- 3,824 attempts
- 3,740 attempts
- 1,853 attempts
- 1,631 attempts
- 1,503 attempts

Therefore, the extreme attempt-count problem is not simply caused by repeated opportunity rows.

Decision:

Extreme attempt values require a dedicated preprocessing strategy later.

Possible approaches such as clipping, transformation, robust aggregation, or filtering will only be selected after the exploration stage is complete.

---

### Response-Time Investigation

#### Extreme Response Times

Response times greater than one hour remain present at the base interaction level.

The maximum observed `ms_first_response` is:

`84,076,920 ms`

This is approximately 23.4 hours.

Therefore, extreme response times are not simply caused by opportunity repetition.

#### Negative Response Times

There are:

- 8 raw rows with negative response times
- 7 unique base `order_id` interactions

One base interaction appears twice because it is associated with two different skills.

Negative elapsed response times are considered invalid timing values.

However, the rest of the corresponding interaction record may still contain useful learning information.

#### Zero Response Times

There are:

- 4,678 raw zero-response-time rows
- 310 unique `order_id` interactions
- 317 unique `order_id + skill_id` representations

Zero-time records include both correct and incorrect interactions.

Decision:

Zero response time will not automatically be treated as invalid or deleted.

Opportunity repetition substantially inflates its raw-row frequency.

---

### `overlap_time`

`overlap_time` is not identical to `ms_first_response`.

Observed:

- No missing values
- 11 negative values
- 2,982 zero values
- 366,920 rows exactly equal `ms_first_response`
- 158,614 rows differ

The median difference between `overlap_time` and `ms_first_response` is zero, but extreme differences exist.

Decision:

The two timing fields will not both automatically be used as model features.

Their semantics and redundancy must be considered during feature selection.

---

### Opportunity Structure

`opportunity`:

- No missing values
- Minimum: 1
- Maximum: 3,585

`opportunity_original`:

- 76,314 missing values
- Minimum observed: 1
- Maximum observed: 3,585

The two fields:

- Are equal in 408,650 rows
- Differ in 116,884 rows

Decision:

Both fields are preserved during exploration.

They must not be treated as interchangeable.

---

### Interaction Ordering Limitation

A major requirement of the Memory Component is identifying recent student interactions.

However, the ASSISTments dataset does not contain an explicit timestamp or date column.

#### Raw Student Row Ordering

Using one row per base `order_id`:

- Students checked: 4,217
- Students already non-decreasing by `order_id`: 1,310
- Students not non-decreasing: 2,907

Therefore, raw CSV row order cannot be treated as student chronology.

#### Within Student + Assignment

Student-assignment groups checked:

- 66,239 total
- 56,981 non-decreasing by `order_id`
- 9,258 not non-decreasing

Approximately 86.02% are ordered.

This indicates that `order_id` contains some useful ordering structure but does not prove exact chronological time.

Decision:

The project must not describe `order_id` as an actual timestamp.

A justified proxy ordering strategy will be designed later.

---

### `position`

`position` contains:

- No missing values
- 283 unique values
- Range: 1–295

Structural investigation found:

- Assignments with multiple positions: 0
- Problems with multiple positions: 21,273
- Assignment + problem pairs with multiple positions: 0

This indicates that `position` is associated with assignment/content structure rather than individual student chronology.

Decision:

`position` will not be used as a student interaction timestamp or chronological counter.

---

### Sequence Structure

The dataset contains:

- 677 unique `sequence_id` values
- 366 unique `base_sequence_id` values

Both fields are complete.

Each assignment maps to:

- One `sequence_id`
- One `base_sequence_id`

However, sequences can be reused across multiple assignments.

Decision:

`sequence_id` and `base_sequence_id` are treated as instructional/content structure identifiers rather than timestamps.

---

### Explicit Timestamp Search

All 30 raw columns were inspected.

No explicit:

- Date
- Timestamp
- Start time
- Completion time

field exists in the available CSV.

This is an important dataset limitation for Short-Term Memory and trend calculations.

Any historical ordering proxy used later must be clearly documented as a proxy rather than exact chronological time.

---

### `original`

Distribution:

- `original = 0`: 76,314 rows (14.52%)
- `original = 1`: 449,220 rows (85.48%)

Both groups contain correct and incorrect responses.

The official meaning of this field has not yet been sufficiently verified.

Decision:

Do not use `original` as a learning-state feature based only on observed correlations.

---

### `assistment_id` vs `problem_id`

At the base interaction level:

- Unique `assistment_id`: 17,725
- Unique `problem_id`: 26,688

Relationship:

- One `assistment_id` may contain multiple problems.
- Maximum observed problems per assistment: 11.
- Each `problem_id` maps to only one assistment in this dataset.

Decision:

`assistment_id` and `problem_id` represent different structural levels and should remain separate identifiers.

---

### Class, Teacher, and School Context

At the base interaction level:

- Unique classes: 250
- Unique teachers: 153
- Unique schools: 75

No values are missing.

Observed relationships:

- 437 students appear in multiple classes.
- No class is associated with multiple teachers in the observed data.
- 21 teachers are associated with multiple schools.
- 25 classes are associated with multiple schools.

Decision:

These identifiers may be useful as metadata but should not automatically become Student Learning State model features.

Using identifiers such as school, teacher, or class directly could encourage the model to learn institution-specific identity patterns rather than student learning behaviour.

---

### Interaction Categorical Fields

#### `tutor_mode`

Two values:

- `tutor`: 525,201
- `test`: 333

The field is extremely imbalanced.

#### `answer_type`

Five values:

- `algebra`: 436,214
- `choose_1`: 48,923
- `fill_in_1`: 39,320
- `choose_n`: 1,069
- `open_response`: 8

This field contains meaningful variation and may require further investigation.

#### `type`

All 525,534 rows contain:

`MasterySection`

Therefore, `type` has zero variation.

Decision:

`type` is not useful as a predictive ML feature in this dataset and will likely be excluded during feature preparation.

---

## Dataset Research Status at the End of Initial EDA

At the time this exploratory analysis was completed, no permanent preprocessing had yet been applied.

The original file:

`data/raw/skill_builder_data.csv`

was preserved unchanged and continues to remain the immutable raw-data source.

The exploratory analysis established the following requirements that were subsequently used to design and validate the preprocessing pipeline:

1. Preserve the raw dataset.
2. Do not treat opportunity repetitions as ordinary duplicate rows.
3. Preserve multi-skill relationships.
4. Separate base-interaction and skill-aware representations where necessary.
5. Handle missing concept information explicitly.
6. Develop justified strategies for extreme attempt counts.
7. Develop justified strategies for invalid/extreme response times.
8. Do not treat zero attempts or zero response times as automatically invalid.
9. Do not assume raw CSV row order represents chronology.
10. Do not describe `order_id` as an exact timestamp.
11. Do not use constant or identity-like contextual fields blindly as ML features.
12. Verify uncertain field semantics before using them in modelling.
13. Keep detailed reproducible analysis in the exploration notebook.
14. Make preprocessing decisions only after dataset understanding is complete.

+---

# Data Understanding Phase — Completed

## Purpose

The purpose of this phase was to understand the ASSISTments Skill Builder dataset before performing preprocessing, feature engineering, state definition, or model development.

The main objective was to determine whether the dataset can support the Student Learning State Identification component and to understand how student interactions, concepts, behavioural signals, and histories should be represented.

No preprocessing transformation was applied to the original raw dataset during this phase.

---

## Dataset

Source file:

`data/raw/skill_builder_data.csv`

Dataset size:

- 525,534 raw rows
- 30 raw columns
- 4,217 students
- 3,521 assignments
- 26,688 problems
- 123 skill IDs
- 110 available skill names

Correctness distribution:

- Correct: 357,085 (67.95%)
- Incorrect: 168,449 (32.05%)

---

## 1. Base Interaction Representation

The raw dataset does not contain one independent student interaction per row.

There are:

- 525,534 raw rows
- 346,860 unique `order_id` values
- 0 exact duplicate raw rows

The following fields were verified to remain stable within an `order_id`:

- `user_id`
- `assignment_id`
- `problem_id`
- `correct`
- `attempt_count`

Therefore:

`order_id`

is the selected logical identifier for the underlying base interaction.

This is an interaction-representation conclusion, not a preprocessing transformation.

---

## 2. Multi-Skill Interaction Representation

A single base interaction can be associated with multiple skills.

Among known-skill base interactions:

- 236,068 contain 1 skill
- 40,857 contain 2 skills
- 4,501 contain 3 skills
- 1,679 contain 4 skills

Therefore:

- 47,037 known-skill base interactions contain more than one skill.
- Multi-skill interactions represent 16.61% of known-skill base interactions.
- A single interaction may contain up to 4 skills.

Therefore, reducing the dataset to one row per `order_id` would remove valid concept information.

For concept-specific processing, the logical key must preserve:

`order_id + skill_id`

---

## 3. Base and Concept Interaction Counts

The final interaction-level understanding is:

- 346,860 base interactions
- 283,105 base interactions with at least one known skill
- 63,755 base interactions with no known skill
- 338,001 known-skill concept interactions

This leads to two required logical representations:

### General Student Memory

Based primarily on:

`order_id`

### Concept-Based Memory

Based primarily on:

`order_id + skill_id`

One base interaction may update multiple concept histories.

---

## 4. Repeated Raw Rows Within Concept Interactions

Most `order_id + skill_id` concept interactions are represented by one raw row.

However:

- 334,774 concept interactions contain exactly one raw row.
- 3,227 concept interactions contain multiple raw rows.
- The maximum number of raw rows for one concept interaction is 215.

These repeated rows were investigated using `opportunity`.

They are not ordinary duplicate answers.

The opportunity values form consecutive ranges belonging to the same logical concept interaction.

Therefore, preprocessing must consolidate these raw representations without treating them as repeated independent student answers.

Useful logical fields include:

- `opportunity_start`
- `opportunity_end`

---

## 5. Concept-History Ordering

The dataset does not contain a direct timestamp suitable for reconstructing all student interactions chronologically.

`order_id` was tested and was not consistently increasing within students or student-assignment groups.

Within student-skill histories, however, `opportunity` showed reliable progression.

Findings:

- 33,381 student-skill histories contained more than one interaction.
- 100% were non-decreasing by `opportunity` in dataset order.
- Repeated opportunity values did not occur across distinct logical concept interactions.
- Apparent opportunity gaps were explained by multi-row opportunity ranges.
- After representing each logical concept interaction as an opportunity range, every observed transition was consecutive.

Therefore:

`opportunity`

is the strongest observed ordering mechanism for concept-specific histories in this dataset.

---

## 6. Skill Information

Skill information is incomplete.

Raw-row findings:

- Missing `skill_id`: 66,326 rows (12.621%)
- Missing `skill_name`: 78,690 rows (14.973%)
- Both missing: 66,326 rows
- `skill_id` present but `skill_name` missing: 12,364 rows

Known skill-ID/name mappings were found to be consistent.

However, 12 skill IDs do not have recoverable names in the dataset.

Therefore, `skill_id` should be treated as the primary concept identity candidate during preprocessing.

Interactions without known skills cannot directly update known Concept-Based Memory but may still contribute to General Student Memory.

---

## 7. Student and Concept History Coverage

Base interactions per student:

- Mean: 82.25
- Median: 23
- Maximum: 1,383

Student-skill histories:

- 41,982 student-skill pairs
- Mean history length: 8.05
- Median: 5
- Maximum: 290

Concept-history coverage:

- >=1 interaction: 100%
- >=2: 79.51%
- >=3: 67.21%
- >=5: 52.72%
- >=10: 27.13%
- >=20: 8.92%

This variation in history depth creates an important cold-start requirement.

---

## 8. Concept Evidence Tiers

The 41,982 student-skill histories were divided by available evidence:

- 1 interaction: 8,601 (20.49%)
- 2 interactions: 5,166 (12.31%)
- 3–4 interactions: 6,082 (14.49%)
- 5 interactions: 3,205 (7.63%)
- 6–9 interactions: 7,539 (17.96%)
- 10+ interactions: 11,389 (27.13%)

As a result:

- 32.79% cannot support recent-three performance.
- 54.91% cannot support a 3+3 trend.
- 72.87% cannot support a 5+5 trend.

Therefore, the final state engine must be evidence-aware.

It cannot require the same history-dependent features for every student-skill state estimate.

---

## 9. General History During Concept Cold Start

At a student's first observed interaction with a skill:

- Mean prior overall interactions: 114.14
- Median prior overall interactions: 42
- 13.18% had no previous overall interaction
- 79.26% had at least 5 prior interactions
- 72.44% had at least 10
- 62.95% had at least 20
- 46.94% had at least 50

Therefore, general student history is often available even when concept-specific history is not.

General history may support cold-start estimation but must remain separate from concept-specific performance.

---

## 10. Correctness Evidence

Correctness is the strongest direct current-performance signal identified during EDA.

For student-skill histories containing at least three interactions, recent-three correct counts showed clear behavioural differences.

Recent correct-count distribution:

- 0 correct: 2,342 histories
- 1 correct: 2,569
- 2 correct: 5,019
- 3 correct: 18,285

Students with lower recent correctness generally showed greater hint use and repeated attempts.

Students with 3/3 recent correct responses showed almost no hint use and almost always one attempt.

Therefore, recent correctness is a primary candidate learning-state signal.

---

## 11. Hint Behaviour

Hint behaviour provides strong supporting evidence.

Student-level findings:

- 925 students never used a hint.
- 3,292 used at least one hint.

For recent student-skill histories, after robust attempt filtering:

- Recent accuracy vs hint-usage rate: approximately -0.705
- Recent accuracy vs average hints: approximately -0.655

Hint measures are also highly redundant:

- Average hints vs hint-usage rate: approximately 0.931

Therefore, hint behaviour is useful but similar hint measures should not automatically receive independent influence.

`bottom_hint` is structurally missing when hints are not used and must not be treated as ordinary random missingness.

---

## 12. Attempt Behaviour

Attempt counts contain meaningful difficulty information but also severe outliers.

Raw findings include:

- Mean attempt count: approximately 1.50
- More than 10 attempts: 2,807 rows
- More than 50 attempts: 449 rows
- More than 100 attempts: 148 rows
- Maximum: 3,824

Approximately 4.215% of raw rows have `attempt_count = 0`.

Extreme values strongly distort raw attempt statistics.

Recent accuracy vs average attempts was approximately:

- -0.077 with extreme observations included
- -0.442 after excluding recent histories containing attempts >10 for sensitivity analysis

Recent accuracy vs multi-attempt rate was approximately:

- -0.618

Therefore, attempt behaviour is useful but requires robust representation during preprocessing and feature engineering.

No final cutoff or transformation has been selected.

---

## 13. Response-Time Behaviour

`ms_first_response` contains:

- 8 negative raw rows
- 4,678 zero-response rows
- 6,544 rows above 5 minutes
- 200 rows above 1 hour

Response-time sensitivity analyses using 1-, 2-, 5-, and 10-minute maximum restrictions produced weak relationships with accuracy change, approximately between -0.042 and -0.057.

Therefore, response time is currently considered secondary contextual evidence rather than a primary learning-state signal.

Its invalid and extreme observations still require explicit preprocessing.

---

## 14. Concept-Specific Performance

Student performance differs substantially across concepts.

Among student-skill pairs with at least five interactions:

- 13.33% had concept accuracy at least 20 percentage points below overall student accuracy.
- 13.81% had concept accuracy at least 20 percentage points above overall student accuracy.

Among students with supported concepts:

- 40.23% had at least one concept >=20 percentage points below overall performance.
- 34.48% had at least one concept >=20 percentage points above overall performance.

Therefore:

General Student Performance != Concept Performance

Concept-Based Memory is necessary.

---

## 15. Trend Evidence

Trend windows investigated:

- 2+2
- 3+3
- 4+4
- 5+5

Among histories supporting all four windows:

- Only 24.20% showed the same trend direction across every window.
- 75.80% showed at least one disagreement.

Therefore, improvement or decline cannot safely be determined from one arbitrary recent-vs-previous comparison.

Trend strength, history availability, and supporting behavioural evidence should be considered later.

No final trend formula has been selected.

---

## 16. Recent Three vs Recent Five

For histories with sufficient evidence:

- Recent-3 vs recent-5 accuracy correlation: 0.880
- 87.45% differed by <=0.20
- Exploratory performance bands agreed in 79.79%
- 20.21% had adjacent-band disagreement

The disagreement cases contained meaningful very-recent movement.

Therefore, short and broader recent windows may provide complementary state evidence.

No final window combination has been selected.

---

## 17. Behavioural Change and Trend

When recent accuracy increased, hints and attempts frequently decreased or remained unchanged.

For histories with attempts <=10 in the last six interactions:

Accuracy increased:

- 93.49% had both hint change <=0 and attempt change <=0.

Accuracy decreased:

- 82.30% had both hint change >=0 and attempt change >=0.

This supports using behavioural change as supporting evidence for learning-state movement.

However, these signals should not independently define the state.

---

## 18. Candidate Evidence Categories

The final EDA supports organizing future state evidence into:

### Current Performance

Examples:

- Recent correctness
- Correct/incorrect counts
- Recent concept accuracy
- Broader recent accuracy

### Support / Effort

Examples:

- Hint behaviour
- Multi-attempt behaviour
- Robust attempt measures

### Trend

Examples:

- Previous-vs-recent accuracy
- Correct-count change
- Short vs broader recent performance
- Hint/attempt change

### Concept Context

Examples:

- `skill_id`
- Concept history
- Concept performance

### General Student Context

Examples:

- Overall history
- Overall performance
- General behavioural patterns

### Evidence Strength

Examples:

- Concept history length
- General history length
- Availability of trend calculations
- Cold-start status

### Data Quality / Reliability

Examples:

- Missing concept
- Extreme attempt
- Invalid response time
- Limited evidence

These categories are not yet the final engineered feature set.

---

## 19. Important Data-Quality Findings

The consolidated raw-data quality inventory identified:

- Missing `skill_id`: 12.621%
- Missing `skill_name`: 14.973%
- Negative response time: 0.002%
- Zero response time: 0.890%
- Response time >5 minutes: 1.245%
- Response time >1 hour: 0.038%
- Attempt count = 0: 4.215%
- Attempt count >10: 0.534%
- Attempt count >100: 0.028%
- Negative overlap time: 0.002%
- Zero overlap time: 0.567%
- Missing `bottom_hint`: 84.795%
- Missing `opportunity_original`: 14.521%
- Missing `answer_id`: 90.737%
- Missing `answer_text`: 18.036%
- Exact duplicate rows: 0

These issues require field-specific handling rather than generic deletion.

---

## 20. Ground-Truth State Labels

The ASSISTments dataset does not directly contain the intended learning-state labels:

- Struggling
- Needs Support
- Improving
- Stable
- Strong

Therefore, these states cannot simply be treated as existing supervised ground truth.

A defensible state-definition and labeling methodology must be developed later using the behavioural evidence established during EDA.

The resulting methodology must be evaluated and validated.

---

## Data Understanding Final Status

**COMPLETE**

The following were deliberately NOT finalized during Data Understanding:

- Data-cleaning transformations
- Final feature set
- Feature weights
- Attempt clipping threshold
- Response-time transformation
- Recent-window selection
- Trend formula
- State thresholds
- State-label generation
- ML algorithm
- Database schema
- Production API

The original raw dataset remains unchanged.

Detailed experimental evidence is preserved in the EDA notebook.

---

# Next Phase — Data Preprocessing and Interaction Construction

The next phase will convert the validated Data Understanding findings into a reproducible processing pipeline.

Initial objectives:

1. Define the preprocessing contract.
2. Preserve the raw dataset unchanged.
3. Construct the base-interaction table.
4. Construct the concept-interaction table.
5. Preserve multi-skill relationships.
6. Consolidate multi-row opportunity ranges.
7. Handle missing concept information explicitly.
8. Define robust treatment for behavioural anomalies.
9. Add preprocessing validation checks.
10. Save reproducible processed datasets.

Feature engineering will begin only after the processed interaction representations have been validated.

+---

# Phase 3 — Data Preprocessing and Interaction Construction

## Step 1 — Preprocessing Contract

### Purpose

The purpose of preprocessing is to convert the raw ASSISTments Skill Builder dataset into reliable interaction-level datasets that can later support feature engineering and Student Learning State Identification.

The preprocessing phase must preserve the meaning discovered during Data Understanding.

The original raw dataset must remain unchanged.

---

## Input

Raw dataset:

`data/raw/skill_builder_data.csv`

Encoding:

`latin1`

Important raw structure:

- 525,534 raw rows
- 346,860 unique base interactions identified by `order_id`
- Some interactions contain multiple skills
- Some `order_id + skill_id` concept interactions contain multiple raw opportunity rows
- Some interactions have no known skill

---

## Required Outputs

The preprocessing phase must create at least two processed interaction datasets.

### 1. Base Interaction Dataset

Logical meaning:

One row represents one underlying student interaction.

Primary key:

`order_id`

Purpose:

Used for:

- General Student Memory
- Overall student behavioural history
- General performance features
- General cold-start context

Important information to preserve includes:

- `order_id`
- `user_id`
- `assignment_id`
- `problem_id`
- `assistment_id`
- `correct`
- `attempt_count`
- `hint_count`
- `hint_total`
- Response-time information
- Relevant interaction context

A multi-skill interaction must appear only once in the base interaction dataset.

---

### 2. Concept Interaction Dataset

Logical meaning:

One row represents one student interaction associated with one known mathematical skill.

Primary logical key:

`order_id + skill_id`

Purpose:

Used for:

- Concept-Based Memory
- Concept performance
- Concept history
- Recent concept behaviour
- Trend analysis

Important information to preserve includes:

- `order_id`
- `user_id`
- `skill_id`
- `skill_name` when available
- Interaction-level behavioural fields
- `opportunity_start`
- `opportunity_end`

If one base interaction contains multiple skills, the concept dataset must contain one row for each valid skill association.

---

## Opportunity-Range Rule

Some `order_id + skill_id` combinations contain multiple raw rows.

These raw rows must not be treated as separate independent answers.

For each logical concept interaction:

`opportunity_start = minimum opportunity`

`opportunity_end = maximum opportunity`

The concept interaction must preserve the entire observed opportunity range.

---

## Missing Skill Rule

Interactions with missing `skill_id` must:

- Remain available in the Base Interaction Dataset
- Not be assigned to an artificial or guessed skill
- Not appear in known-skill Concept Interaction history unless a valid skill identifier exists

Missing concept information must remain explicitly identifiable.

---

## Multi-Skill Preservation Rule

If one `order_id` is associated with multiple valid `skill_id` values:

- Keep one row in the Base Interaction Dataset
- Keep one row per `order_id + skill_id` in the Concept Interaction Dataset

Do not arbitrarily select a single skill.

---

## Raw Data Preservation Rule

The raw file:

`data/raw/skill_builder_data.csv`

must never be overwritten.

All transformations must produce new files under:

`data/processed/`

The processing pipeline must be reproducible from the raw dataset.

---

## Behavioural Data Rule

Potentially unusual values such as:

- Zero attempts
- Extreme attempt counts
- Negative response times
- Zero response times
- Extreme response times

must not be silently deleted during interaction construction.

Initial interaction construction should preserve these observations.

Robust cleaning or transformation decisions will be implemented later as explicit preprocessing steps.

---

## Identifier Rule

Identifier fields must not be transformed into learning-state scores.

Identifiers are used for:

- Grouping
- Joining
- History retrieval
- Interaction tracking

They are not treated as direct measures of student ability.

---

## Validation Requirements

The preprocessing pipeline must later verify:

1. Base dataset has exactly one row per `order_id`.
2. Concept dataset has exactly one row per `order_id + skill_id`.
3. Multi-skill relationships are preserved.
4. Base behavioural values remain consistent with the raw interaction.
5. Opportunity ranges are valid.
6. No artificial skill IDs are introduced.
7. Raw dataset remains unchanged.
8. Processed row counts match expected structural counts from EDA.
9. Student and interaction identifiers are not lost.
10. Re-running preprocessing produces the same outputs.

---

## Expected Initial Processed Counts

Based on Data Understanding:

Base interactions:

`346,860`

Known-skill concept interactions:

`338,001`

Base interactions with no known skill:

`63,755`

These values will be used as validation targets during implementation.

---

## Phase Boundary

This preprocessing phase will create reliable interaction representations.

It will not yet define:

- Final learning-state features
- State labels
- Rule thresholds
- ML models
- Prediction APIs

Feature engineering will begin only after the processed interaction datasets are validated.

## Phase 3 — Processed Data Semantic Types

The initial processed datasets have been successfully created and validated.

Before applying dtype conversions, processed fields are classified by semantic role.

### Base Interaction Dataset

#### Identifiers

- `order_id`
- `user_id`
- `assignment_id`
- `assistment_id`
- `problem_id`
- `sequence_id`
- `base_sequence_id`
- `student_class_id`
- `teacher_id`
- `school_id`

These fields identify entities or structures and should not be interpreted as numerical learning measurements.

#### Binary / Discrete Behaviour

- `correct`
- `first_action`
- `original`
- `bottom_hint`

Some of these fields may contain structural missing values.

#### Count Behaviour

- `attempt_count`
- `hint_count`
- `hint_total`

#### Time / Duration Behaviour

- `ms_first_response`
- `overlap_time`

#### Structural / Positional Information

- `position`

#### Categorical Context

- `tutor_mode`
- `answer_type`

---

### Concept Interaction Dataset

The Concept Interaction Dataset includes the same interaction-level behavioural fields plus concept-specific information.

#### Concept Identity

- `skill_id`
- `skill_name`

`skill_id` is the primary concept identifier.

#### Concept Progression

- `opportunity_start`
- `opportunity_end`
- `raw_row_count`

These represent the logical opportunity range associated with one `order_id + skill_id` concept interaction.

---

### Current Dtype Observation

After CSV save and reload:

- Identifier/count columns are generally `int64`.
- `bottom_hint` is `float64` because structural missing values are present.
- `skill_id` remains `float64` even though the processed concept dataset contains no missing skill IDs.

No dtype conversions are applied yet.

A dedicated dtype-normalization step will decide whether fields such as `skill_id` should be converted to integer types and whether nullable integer or categorical dtypes are appropriate.

---

## Phase 3 — Final Preprocessing Results

### Final Interaction Construction

The preprocessing pipeline successfully converted the raw ASSISTments data into two validated logical interaction representations.

### Base Interaction Dataset

Final shape:

- Rows: 346,860
- Columns: 22
- Unique students: 4,217
- Primary key: `order_id`

Each row represents one underlying student interaction.

Validation confirmed:

- `order_id` is unique.
- No base interaction was duplicated.
- Required base-interaction fields contain no unexpected missing values.
- Multi-skill raw representations do not create duplicate base interactions.

### Concept Interaction Dataset

Final shape:

- Rows: 338,001
- Columns: 27
- Unique students: 4,163
- Unique skills: 123
- Primary logical key: `order_id + skill_id`

Each row represents one student interaction associated with one known mathematical skill.

Validation confirmed:

- `order_id + skill_id` is unique.
- `skill_id` is complete.
- `skill_id` uses an integer dtype.
- Multi-skill interactions are preserved.
- Opportunity ranges are valid.
- `raw_row_count >= 1`.

---

### Skill-Name Handling

Among the 338,001 Concept Interactions:

- Missing `skill_name`: 12,364
- Percentage missing: 3.66%
- Total unique skill IDs: 123
- Skill IDs with at least one known name: 111
- All 111 known mappings are one-to-one.
- 12 skill IDs have no recoverable name anywhere in the available dataset.

The unrecoverable skill IDs are:

`37, 76, 91, 94, 96, 97, 99, 101, 102, 104, 105, 367`

Decision:

`skill_id` remains the primary concept identifier.

Missing skill names are preserved rather than guessed or artificially imputed.

---

### Attempt-Count Investigation

At the Base Interaction level:

- Mean attempt count: 1.64
- Median: 1
- 20.36% of interactions have more than one attempt.
- 0.71% have more than 10 attempts.
- 0.04% have more than 100 attempts.
- Maximum: 3,824

Extreme values are rare but have a disproportionate effect on raw means.

For example:

- Attempts >10 account for approximately 17.03% of all recorded attempts.
- Attempts >100 account for approximately 7.74%.

The extreme observations are distributed across many students and problems rather than being attributable to one single student or problem.

Decision:

Extreme attempt counts are preserved in the processed interaction datasets.

They are not silently deleted or replaced.

Robust representations are handled later during feature engineering.

---

### Hint-Feature Investigation

At the Base Interaction level:

- 59,854 interactions used at least one hint.
- 17.26% of Base Interactions used hints.
- `hint_count` never exceeds `hint_total`.

For interactions with `hint_total > 0`:

- 245,309 interactions have measurable normalized hint usage.
- 185,455 used no hints despite hints being available.
- 17,171 used some but not all available hints.
- 42,683 used all available hints.

The normalized hint-usage rate is defined only when hints were available.

Therefore, `hint_total = 0` is treated as absence of measurable hint-usage opportunity rather than ordinary missing data.

---

### `bottom_hint` Structural Missingness

At the Base Interaction level:

- Missing `bottom_hint`: 287,003
- Missing percentage: 82.74%

However, this missingness is structural.

Among interactions with `hint_count = 0`:

- `bottom_hint` is missing in 100% of cases.

Among interactions with `hint_count > 0`:

- `bottom_hint` is available in 100% of cases.

Therefore, missing `bottom_hint` does not represent random missing data.

Among hint-using interactions:

- `bottom_hint = 1`: 42,932
- `bottom_hint = 0`: 16,922
- 71.73% reached the bottom hint.

---

### Candidate Signal Integrity

All required Base Interaction candidate signals were complete except the expected structural missingness in `bottom_hint`.

All required Concept Interaction candidate signals were complete except:

- structural `bottom_hint` missingness
- the known 12,364 missing `skill_name` values

No unexpected missing values were found in required Base or Concept fields.

Domain validation confirmed:

1. `correct` contains only 0/1.
2. `attempt_count` is non-negative.
3. `hint_count` is non-negative.
4. `hint_total` is non-negative.
5. `hint_count <= hint_total`.
6. `first_action` contains only observed codes 0/1/2.
7. `bottom_hint` contains only 0/1 when available.
8. Concept `skill_id` is positive.
9. Concept `raw_row_count >= 1`.
10. `opportunity_end >= opportunity_start`.

Result:

**10 / 10 domain checks passed.**

---

### Final Preprocessing Integrity Validation

The completed preprocessing pipeline was subjected to a final integrity validation.

Validated conditions included:

- Base rows = 346,860
- Base `order_id` unique
- Base students = 4,217
- Base required fields contain no unexpected missing values
- Concept rows = 338,001
- Concept `order_id + skill_id` unique
- Concept students = 4,163
- Concept `skill_id` complete
- Concept `skill_id` integer dtype
- Concept opportunity ranges valid
- Concept `raw_row_count >= 1`
- Concept required fields contain no unexpected missing values

Result:

**12 / 12 checks passed.**

**FINAL PREPROCESSING RESULT: PASS**

---

## Phase 3 Final Status

**COMPLETE**

The preprocessing phase successfully produced validated Base and Concept Interaction datasets without modifying the original raw dataset.

The processed interaction representations are now suitable for historical feature engineering.

The next phase is:

**Phase 4 — Historical Feature Engineering**

---

# Phase 4 — Historical Feature Engineering

## Purpose

The purpose of Phase 4 is to transform the validated interaction datasets into historical learning signals that can later support Student Learning State Identification.

A critical requirement is preventing target leakage.

For every current Concept Interaction, predictor features must describe only information available **before** that interaction.

The current interaction's outcome and behavioural values must not be included in its predictor features.

---

## Step 1 — Historical Ordering Investigation

Before calculating recent or historical features, the available ordering fields were investigated carefully.

The Base Interaction Dataset contains:

- `order_id`
- `sequence_id`
- `base_sequence_id`
- `position`

The Concept Interaction Dataset additionally contains:

- `opportunity_start`
- `opportunity_end`

### Position Investigation

Across consecutive student interactions:

- Comparable transitions: 342,643
- Forward position transitions: 53,984
- Same-position transitions: 271,329
- Backward position transitions: 17,330

However, every backward position transition occurred with an assignment/context change.

Within the same student-assignment context:

- Comparable transitions: 280,621
- Forward transitions: 0
- Same-position transitions: 280,621
- Backward transitions: 0

This confirmed that `position` primarily represents assignment/content structure and is not a global student timestamp.

---

## Step 2 — `order_id` Ordering Investigation

At the global Base Interaction level:

- Base interactions: 346,860
- Unique `order_id`: 346,860
- Duplicate `order_id`: 0
- `order_id` linked to multiple students: 0
- Minimum `order_id`: 20,224,085
- Maximum `order_id`: 38,310,202

Globally sorted `order_id` values are strictly increasing.

However, raw dataset row order does not always follow increasing `order_id`.

Across comparable consecutive student interactions in raw-row order:

- Increasing `order_id`: 319,536
- Decreasing `order_id`: 23,107
- Percentage increasing: 93.26%

Most decreasing transitions also involved assignment, sequence, base-sequence, or position changes.

This confirmed that raw CSV row order should not be used directly as the historical sequence.

### Ordering Decision

For overall student-history feature construction, interactions are ordered by:

`order_id`

within each student.

`order_id` is used as an **ordering proxy**, not described as a real timestamp.

For concept-specific history, `order_id` is used together with the validated Concept Interaction structure, while the opportunity-range fields remain preserved as concept progression metadata.

---

## Step 3 — Leakage-Safe Historical Counts

After ordering interactions, historical counts were calculated using only preceding interactions.

For every Base Interaction:

`previous_interaction_count`

represents the number of interactions belonging to that student before the current interaction.

Validation results:

- Minimum: 0
- Maximum: 1,382
- Interactions with zero previous history: 4,217
- Unique students: 4,217
- Exactly one zero-history Base Interaction per student: True

Historical correct and incorrect counts were also created:

- `previous_correct_count`
- `previous_incorrect_count`

Validation confirmed:

`previous_correct_count + previous_incorrect_count = previous_interaction_count`

for every interaction.

The first interaction for every student correctly contains:

- previous correct count = 0
- previous incorrect count = 0

---

## Step 4 — Historical Overall Accuracy

Historical overall accuracy was calculated as:

`previous_correct_count / previous_interaction_count`

when previous history exists.

Feature:

`historical_accuracy`

Availability:

- Available: 342,643 Base Interactions
- Unavailable: 4,217
- Unavailable exactly once per student on the first interaction

Observed range:

- Minimum: 0.0
- Maximum: 1.0

All available values were within the valid `[0,1]` range.

The current interaction's `correct` value is not included in this calculation.

---

## Step 5 — Candidate Recent-History Windows

Candidate overall-history windows were investigated:

- 3 interactions
- 5 interactions
- 10 interactions
- 20 interactions
- 30 interactions

Full-window coverage:

- Window 3: 96.47% of Base Interactions
- Window 5: 94.28%
- Window 10: 89.43%
- Window 20: 81.83%
- Window 30: 75.99%

Students reaching each full window:

- 3 interactions: 91.06%
- 5 interactions: 85.16%
- 10 interactions: 71.47%
- 20 interactions: 53.17%
- 30 interactions: 42.85%

The 10-interaction window provides a useful balance between recent behaviour and dataset coverage.

It was therefore selected as the main broader recent overall-performance window.

---

## Step 6 — Recent Overall Accuracy

Candidate recent accuracy features were compared across windows of:

- 3
- 5
- 10
- 20
- 30 interactions

Correlation with historical overall accuracy increased as the window became larger:

- Window 3: 0.567
- Window 5: 0.652
- Window 10: 0.758
- Window 20: 0.849
- Window 30: 0.892

The 5-interaction window is more sensitive to very recent movement, while the 10-interaction window provides a broader recent baseline.

Therefore, the main recent-performance feature selected was:

`recent_accuracy_10`

and recent movement is represented using:

`recent_accuracy_change_5_vs_10`

defined as the difference between recent 5-interaction and recent 10-interaction accuracy.

For interactions with a full 10-interaction history:

- Positive change: 35.14%
- Negative change: 34.97%
- Exactly zero: 29.90%

The observed change range is:

`[-0.5, 0.5]`

---

## Step 7 — Recent-Accuracy Leakage Validation

Recent-accuracy calculations were manually checked on sampled interactions.

For each sampled current interaction:

- Previous 5 correctness values were inspected.
- Previous 10 correctness values were inspected.
- Manual recent accuracies were calculated.
- Calculated feature values were compared with the manual values.

Results:

- 5-interaction calculations matched: True
- 10-interaction calculations matched: True

**All recent-accuracy leakage-validation checks passed.**

This confirms that the current interaction's correctness is excluded from the recent historical accuracy features.

---

## Step 8 — Historical Attempt Features

Attempt behaviour was engineered using only the student's previous interactions.

The raw `attempt_count` field contains severe extreme values, including values above 3,000. Therefore, using a simple raw rolling mean could allow a single extreme interaction to dominate the recent behavioural history.

Several 10-interaction representations were compared:

- Raw mean
- Capped mean
- Log-transformed mean
- Median
- Multi-attempt rate

For full 10-history interactions, the raw mean reached:

`387.9`

while the log-transformed representation was substantially more robust.

The log-transformed attempt feature was defined from previous attempts using:

`log1p(attempt_count)`

and then averaging the transformed values over the recent history.

The following two attempt features were selected:

- `recent_attempt_log_mean_10`
- `recent_multi_attempt_rate_10`

`recent_multi_attempt_rate_10` represents the proportion of recent previous interactions where more than one attempt was recorded.

These features capture related but different information:

- Log mean represents overall recent attempt intensity.
- Multi-attempt rate represents how consistently repeated attempts occur.

Their observed correlation was approximately:

`0.823`

This relationship is high and must be considered during later modelling, but both were retained at the feature-engineering stage because they have different behavioural interpretations.

### Attempt Outlier Investigation

Among interactions with full 10-interaction history:

- 17,357 windows contained at least one raw attempt count above 10.
- This represents 5.60% of full 10-history interactions.

The raw recent attempt mean was strongly affected by these extreme observations.

This supports using robust transformed attempt features instead of relying on the raw rolling mean.

### Attempt Leakage Validation

Sample interactions were manually reconstructed using their previous 10 attempt counts.

The calculated:

- `recent_attempt_log_mean_10`
- `recent_multi_attempt_rate_10`

were compared against manual calculations.

Results:

- Log-mean calculations matched: True
- Multi-attempt-rate calculations matched: True

**All attempt leakage-validation checks passed.**

The current interaction's attempt count is not included in these historical predictors.

---

## Step 9 — Historical Hint Features

Hint behaviour was investigated using the previous 10 interactions.

Two important issues were considered:

1. A student may not use a hint.
2. Some interactions may not provide measurable hint opportunities.

Therefore, zero hint use must be distinguished from unavailable hint evidence.

The following candidate features were investigated:

- Recent hint-use rate
- Recent hint-availability rate
- Number of measurable hint interactions
- Normalized hint-usage rate

### Hint Availability

`recent_hint_available_rate_10`

represents the proportion of the previous 10 interactions where measurable hint information was available.

Among full 10-history interactions:

- Mean measurable hint count: 6.986 out of 10
- 9,333 interactions had no measurable hint-use observation
- 76,825 had measurable hint information for all 10 previous interactions

Therefore, hint evidence is not uniformly available.

### Normalized Hint Usage

`recent_hint_usage_rate_10`

was calculated using only interactions where hint usage was measurable.

It represents normalized recent hint behaviour without automatically treating unavailable hint opportunities as genuine zero hint usage.

Among full 10-history interactions:

- Available normalized hint-usage values: 300,866
- Missing because no measurable hint evidence: 9,333

The missing values occurred exactly when the measurable hint count was zero.

### Evidence Reliability

The amount of hint evidence varies considerably.

For full 10-history interactions:

- 0 measurable: 3.01%
- 1–2 measurable: 4.61%
- 3–5 measurable: 19.01%
- 6–9 measurable: 48.61%
- 10 measurable: 24.77%

Therefore, hint behaviour must be interpreted together with an evidence indicator rather than treating every normalized hint rate as equally reliable.

### Selected Hint Features

The selected historical hint predictors are:

- `recent_hint_usage_rate_10`
- `recent_hint_available_rate_10`

An explicit evidence indicator is later included to identify whether normalized hint usage is available.

### Hint Leakage Validation

Sample interactions were reconstructed using the previous 10:

- `hint_count`
- `hint_total`

values.

Manual calculations were compared with the engineered features.

Results:

- Hint-availability calculations matched: True
- Normalized hint-usage calculations matched: True

**All hint leakage-validation checks passed.**

Current-interaction hint behaviour is excluded from the historical predictors.

---

## Step 10 — Historical Response-Time Features

Response time was engineered using:

`ms_first_response`

from previous interactions only.

Raw response time contains invalid and extreme observations and is strongly right-skewed.

Therefore, multiple representations were investigated:

- Raw recent mean
- Recent median
- Log-transformed recent mean

The raw rolling mean was highly sensitive to extreme response times.

For full 10-history interactions:

- Mean raw recent response time: approximately 44,165 ms
- Median raw recent response time: approximately 24,542 ms
- Maximum recent raw mean: approximately 8,448,743 ms

The extreme raw means were caused by very large historical response-time observations.

### Selected Response-Time Features

Two representations were retained:

- `recent_response_log_mean_10`
- `recent_response_median_ms_10`

The log-transformed feature reduces the influence of extreme durations, while the median provides a directly interpretable robust timing measure.

The correlation between these two representations was approximately:

`0.766`

### Response-Time Evidence Availability

Among full 10-history interactions:

- 308,759 had all 10 previous response times positive.
- 1,440 had fewer than 10 positive response-time observations.

Therefore, response-time history is available for almost all full-history interactions.

### Behavioural Interpretation

Response time did not show a simple relationship where slower responses automatically meant weaker performance.

When interactions were divided into response-time quartiles, average recent accuracy was:

- Fastest quartile: 0.545
- Q2: 0.664
- Q3: 0.683
- Slowest quartile: 0.693

This confirms that response time should not independently determine learning state.

It is retained as secondary behavioural/contextual evidence.

### Response-Time Leakage Validation

Sample interactions were reconstructed using only their previous 10 response-time values.

Manual values were compared with:

- `recent_response_median_ms_10`
- `recent_response_log_mean_10`

Results:

- Median-response calculations matched: True
- Log-response calculations matched: True

**All response-time leakage-validation checks passed.**

The current interaction's response time is excluded from the historical predictors.

---

## Historical Behaviour Feature Status

At this stage, the validated overall-history behavioural features include:

- `previous_interaction_count`
- `historical_accuracy`
- `recent_accuracy_10`
- `recent_accuracy_change_5_vs_10`
- `recent_attempt_log_mean_10`
- `recent_multi_attempt_rate_10`
- `recent_hint_usage_rate_10`
- `recent_hint_available_rate_10`
- `recent_response_log_mean_10`
- `recent_response_median_ms_10`

All rolling behavioural features were designed to use **previous interactions only**.

The next feature-engineering stage focuses on **concept-specific student history**.
 
---

# Phase 3 — Data Preprocessing and Interaction Construction — Completed

## Phase Summary

The preprocessing and interaction-construction phase was completed after the structural findings from Data Understanding were converted into reproducible interaction-level representations.

The raw dataset remains unchanged:

`data/raw/skill_builder_data.csv`

Two logical interaction levels were established:

### Base Interaction Level

Primary identifier:

`order_id`

Final number of base interactions:

`346,860`

Each base interaction represents one underlying student interaction and is used for general student-history calculations.

### Concept Interaction Level

Primary logical key:

`order_id + skill_id`

Final number of known-skill concept interactions:

`338,001`

This representation preserves multi-skill interactions and is used for concept-specific student history.

Interactions without a known skill remain useful for overall student history but cannot directly contribute to known concept histories.

---

## Interaction Identity Validation

The following properties were confirmed for `order_id`:

- Base interactions: 346,860
- Unique `order_id`: 346,860
- Duplicate `order_id`: 0
- `order_id` linked to multiple students: 0
- Minimum `order_id`: 20,224,085
- Maximum `order_id`: 38,310,202

Within an `order_id`, the following important interaction properties were verified to remain stable:

- Student
- Assignment
- Problem
- Correctness
- Attempt count

Therefore, `order_id` is used as the logical base-interaction identifier.

However, it is not described as an actual timestamp.

---

## Ordering Investigation

The dataset contains no explicit interaction timestamp.

Several possible ordering signals were investigated.

Raw CSV row order was rejected as reliable student chronology.

`position`, `sequence_id`, and `base_sequence_id` were found to primarily describe instructional/content structure.

`order_id` contains strong global structure but is not perfectly increasing when interactions are inspected using raw student row ordering.

A total of 23,107 decreasing `order_id` transitions were observed during the ordering investigation.

Most occurred when assignment or instructional structure also changed.

Therefore, `order_id` is treated as a proxy interaction-order mechanism rather than an actual time measurement.

For concept-specific history, opportunity progression provides additional structural ordering evidence.

All historical features are explicitly defined relative to previous reconstructed interactions and must not use the current interaction's behavioural outcome.

---

# Phase 4 — Learning-State Feature Engineering

## Goal

The purpose of this phase was to transform the validated interaction representations into historical student-learning signals suitable for Student Learning State Identification.

The central design requirement was:

> A state prediction for a current interaction must use only information available before that interaction.

Therefore, current-interaction behavioural outcomes must not leak into predictor features.

The current `correct` value is retained for later state-definition, evaluation, and research purposes but is not included in the final predictor list.

---

## 1. Previous Overall Interaction Count

The first historical feature created was:

`previous_interaction_count`

It represents the number of previous base interactions available for the student before the current interaction.

Results:

- Feature rows: 346,860
- Minimum: 0
- Maximum: 1,382
- Interactions with zero previous history: 4,217
- Unique students: 4,217

Exactly one zero-history interaction exists per student.

This validates the overall-history construction.

---

## 2. Historical Correct and Incorrect Counts

Two cumulative historical quantities were validated:

- Previous correct count
- Previous incorrect count

For every interaction:

`previous_correct_count + previous_incorrect_count = previous_interaction_count`

The equality held for the complete dataset.

The student's first interaction correctly contains:

- Previous correct count = 0
- Previous incorrect count = 0

These quantities were used to validate cumulative historical performance.

---

## 3. Historical Overall Accuracy

Historical overall accuracy was calculated using only previous interactions.

Conceptually:

`historical_accuracy = previous_correct_count / previous_interaction_count`

It is undefined when no previous interaction exists.

Results:

- Available: 342,643
- Unavailable: 4,217
- Expected unavailable: 4,217

The missing values correspond exactly to each student's first interaction.

Range validation:

- Minimum: 0.0
- Maximum: 1.0
- All available values within [0,1]: True

This confirmed correct historical accuracy construction.

---

## 4. Recent Overall Accuracy Window Investigation

Candidate recent-history windows were investigated:

- 3 interactions
- 5 interactions
- 10 interactions
- 20 interactions
- 30 interactions

Coverage for a complete window was:

- 3: 96.47%
- 5: 94.28%
- 10: 89.43%
- 20: 81.83%
- 30: 75.99%

Student coverage reaching each complete window was:

- 3: 91.06%
- 5: 85.16%
- 10: 71.47%
- 20: 53.17%
- 30: 42.85%

The 10-interaction window provided a useful balance between:

- Recent behaviour
- Stability
- Student coverage

Therefore:

`recent_accuracy_10`

was retained as the primary recent overall performance feature.

---

## 5. Overall Recent Accuracy Change

A short-vs-broader recent comparison was investigated using:

- Recent 5 interactions
- Recent 10 interactions

The feature:

`recent_accuracy_change_5_vs_10`

represents:

`recent_accuracy_5 - recent_accuracy_10`

For interactions with full 10-interaction history:

- Available interactions: 310,199
- Positive differences: 35.14%
- Negative differences: 34.97%
- Exactly zero: 29.90%

Observed range:

`[-0.5, 0.5]`

This feature provides evidence about whether very recent performance is above or below the broader recent baseline.

It is treated as trend evidence rather than a direct state label.

---

## 6. Accuracy Leakage Validation

Recent accuracy features were manually reconstructed for sampled interactions using the student's previous correctness sequence.

Both:

- `recent_accuracy_5`
- `recent_accuracy_10`

matched manual calculations.

Results:

- 5-interaction calculations matched: True
- 10-interaction calculations matched: True
- All leakage-validation checks passed: True

The current interaction's `correct` value is excluded from these historical calculations.

---

# Historical Attempt Features

## Attempt Outlier Problem

Raw attempt counts contain severe extreme observations.

For full 10-history windows:

- Raw recent mean maximum: 387.9
- Raw recent maximum could reach 3,824

Therefore, a simple raw attempt mean is highly vulnerable to extreme historical observations.

Several representations were compared:

- Raw mean
- Capped mean
- Log-transformed mean
- Median
- Multi-attempt rate

Among full 10-history interactions:

17,357 windows contained at least one raw attempt value above 10.

This represents:

`5.60%`

of full 10-history interactions.

---

## Selected Attempt Features

Two attempt features were retained:

`recent_attempt_log_mean_10`

and

`recent_multi_attempt_rate_10`

The log feature is based on transforming historical attempt counts with:

`log1p(attempt_count)`

before aggregation.

The multi-attempt rate measures how frequently the student required more than one attempt during recent interactions.

These provide:

- Robust attempt intensity
- Repeated-attempt frequency

Their observed correlation was approximately:

`0.823`

This is high and must be considered during modelling, but both were retained because their behavioural interpretations differ.

---

## Attempt Leakage Validation

Sample interactions were manually reconstructed from the previous 10 attempt counts.

Results:

- Log-mean calculations matched: True
- Multi-attempt-rate calculations matched: True
- All attempt leakage-validation checks passed: True

The current interaction's attempt count is not included.

---

# Historical Hint Features

## Hint Evidence Problem

Hint behaviour requires special treatment because hint opportunities are not equally measurable for every interaction.

The engineered hint features distinguish:

- Actual hint behaviour
- Availability of measurable hint evidence

Selected features:

`recent_hint_usage_rate_10`

`recent_hint_available_rate_10`

---

## Hint Evidence Coverage

For full 10-history interactions:

- Mean measurable hint observations: 6.986
- No measurable hint observation: 9,333
- All 10 measurable: 76,825

Evidence groups were:

- 0 measurable: 3.01%
- 1–2 measurable: 4.61%
- 3–5 measurable: 19.01%
- 6–9 measurable: 48.61%
- 10 measurable: 24.77%

Therefore, normalized hint usage cannot be interpreted without considering evidence availability.

---

## Normalized Hint Usage

`recent_hint_usage_rate_10`

is calculated from measurable previous hint opportunities.

Missing normalized hint usage occurs when no measurable hint evidence exists.

This relationship was verified:

Missing hint usage exactly when measurable count = 0:

`True`

Therefore, missing hint usage represents lack of evidence rather than automatically representing zero hint use.

---

## Hint Leakage Validation

Sample previous-10 hint histories were manually reconstructed.

Results:

- Hint-availability calculations matched: True
- Normalized hint-usage calculations matched: True
- All hint leakage-validation checks passed: True

Current-interaction hint behaviour is excluded.

---

# Historical Response-Time Features

## Response-Time Robustness

Raw response time contains extreme values.

For full 10-history interactions:

- Mean raw recent response time: approximately 44,165 ms
- Median recent response time: approximately 24,542 ms
- Maximum raw recent mean: approximately 8,448,743 ms

Therefore, raw mean response time is not sufficiently robust for direct use.

---

## Selected Response-Time Features

Two representations were retained:

`recent_response_log_mean_10`

`recent_response_median_ms_10`

The log representation reduces extreme-value influence.

The median provides a robust and directly interpretable recent response-duration measure.

Their observed correlation was approximately:

`0.766`

---

## Response-Time Evidence

Among interactions with full 10-history:

- 308,759 had all 10 previous response times positive.
- 1,440 had fewer than 10 positive response-time measurements.

Response-time information is therefore available for nearly all complete recent histories.

---

## Response-Time Interpretation

Response time does not have a simple interpretation where slower automatically means weaker.

Average recent accuracy by response-time quartile was approximately:

- Q1 fastest: 0.545
- Q2: 0.664
- Q3: 0.683
- Q4 slowest: 0.693

Therefore, response time is treated as supporting contextual evidence rather than an independent learning-state definition.

---

## Response-Time Leakage Validation

Previous response-time histories were manually reconstructed.

Results:

- Median-response calculations matched: True
- Log-response calculations matched: True
- All response-time leakage-validation checks passed: True

The current response time is excluded.

---

# Concept-Specific Historical Features

## Concept Interaction Population

Concept-level feature engineering uses known-skill interactions.

Results:

- Concept interactions: 338,001
- Students: 4,163
- Skills: 123
- Student-skill pairs: 41,982

The concept-history counter:

`previous_skill_interaction_count`

represents the number of previous interactions the student has had with the current skill.

Observed range:

- Minimum: 0
- Maximum: 289

Interactions with no previous history for the current skill:

`41,982`

Percentage:

`12.42%`

---

## Concept History Depth

Previous skill-history depth varies substantially.

Complete history coverage:

- >=1 previous interaction: 87.58%
- >=2: 77.70%
- >=3: 69.36%
- >=5: 55.30%
- >=10: 33.43%
- >=20: 14.96%

Therefore, concept-level state prediction requires explicit evidence awareness.

---

## Historical Skill Accuracy

`historical_skill_accuracy`

was calculated using only previous interactions for the same student and skill.

Results:

- Available: 296,019
- Unavailable: 41,982

Unavailable values occur exactly on the first student-skill interaction.

Range:

`[0,1]`

---

## Recent Skill Accuracy Investigation

Candidate concept windows investigated:

- 3 interactions
- 5 interactions
- 10 interactions

Correlations with historical skill accuracy were approximately:

- Recent 3: 0.866
- Recent 5: 0.932
- Recent 10: 0.979

Full-window coverage was:

- 3: 69.36%
- 5: 55.30%
- 10: 33.43%

The five-interaction representation provided a useful compromise between concept-specific recency and evidence coverage.

Therefore:

`recent_skill_accuracy_5`

was selected as the final recent concept-performance predictor.

---

# Concept-Specific Change Feature

A concept-specific trend feature was constructed from the previous five interactions with the current skill.

The previous five were divided into:

- Older 2 interactions
- Most recent 3 interactions

The feature:

`recent_skill_accuracy_change_3_vs_previous_2`

represents:

`accuracy of recent 3 - accuracy of previous 2`

It is available only when five previous skill interactions exist.

Results:

- Available: 186,916
- Percentage of concept interactions: 55.30%
- Range: [-1,1]

Distribution:

- Positive changes: 75,204
- Negative changes: 64,575
- Exactly zero: 47,137

This feature provides concept-specific movement evidence.

It does not independently define the student's learning state.

---

## Concept-Change Leakage Validation

Sample student-skill histories were manually reconstructed.

The following were manually checked:

- Previous-two accuracy
- Recent-three accuracy
- Difference between them

Results:

- Prior-two calculations matched: True
- Recent-three calculations matched: True
- Change calculations matched: True
- All concept-change leakage-validation checks passed: True

The current concept interaction is excluded from the calculation.

---

# Overall vs Concept Performance Investigation

Concept-specific and overall student performance were compared.

For interactions where both were available, the correlation between overall and skill-specific performance was substantial but not perfect.

Examples:

- Historical overall vs historical skill accuracy: approximately 0.597
- Recent overall vs recent skill accuracy showed similarly incomplete alignment.

The difference:

`skill performance - overall recent performance`

showed meaningful variation.

Using an absolute ±0.20 comparison:

- Skill higher than overall: 10.57%
- Skill lower than overall: 23.66%

This confirms an important architectural requirement:

> Overall student performance and concept-specific student performance must remain separate.

A student can perform strongly overall while struggling with a particular concept, or perform weakly overall while showing strength in a particular concept.

---

# Concept Evidence Strength

Concept interactions were grouped by previous skill-history depth.

Results:

- 0 previous: 41,982
- 1 previous: 33,381
- 2 previous: 28,215
- 3–4 previous: 47,507
- 5–9 previous: 73,920
- 10+ previous: 112,996

This corresponds to:

- Full 5-interaction skill history: 55.30%
- Partial 1–4 skill history: 32.28%
- No previous skill history: 12.42%

These categories cover all concept interactions.

Therefore, evidence strength must be represented explicitly rather than hidden through arbitrary imputation.

---

# Cold-Start and Evidence Indicators

Explicit binary evidence indicators were created.

## `has_overall_history`

Indicates whether previous overall student history exists.

Evidence available:

`98.54%`

## `has_skill_history`

Indicates whether previous history exists for the current skill.

Evidence available:

`87.58%`

## `has_full_skill_window_5`

Indicates whether at least five previous skill interactions exist.

Evidence available:

`55.30%`

## `has_skill_change_evidence`

Indicates whether the concept-change feature can be calculated.

Evidence available:

`55.30%`

## `has_recent_hint_usage_evidence`

Indicates whether normalized recent hint usage can be calculated.

Evidence available:

`95.26%`

Consistency checks confirmed that the evidence indicators exactly match the availability of their corresponding historical features.

---

## Joint Overall and Skill Cold Start

The concept table contains:

- 4,940 rows with no overall history
- 37,042 rows with overall history but no skill history
- 296,019 rows with both overall and skill history

Rows with skill history but no overall history:

`0`

This is logically consistent because concept history is a subset of the student's overall interaction history.

---

# Unified Feature Table

The overall-history and concept-history feature sets were joined.

Unified table results:

- Rows: 338,001
- Columns before final export selection: 20
- Unique students: 4,163
- Unique skills: 123
- Unique `order_id`: 283,105

No concept interaction lacked a matching overall feature record.

Current `correct` was retained in the table for later research use but was explicitly excluded from the predictor list.

---

# Final Predictor Selection

After feature investigation, robustness analysis, evidence analysis, redundancy checks, and leakage validation, the following 16 predictors were selected.

## Final 16 Predictors

1. `previous_interaction_count`
2. `previous_skill_interaction_count`
3. `recent_accuracy_10`
4. `recent_accuracy_change_5_vs_10`
5. `recent_attempt_log_mean_10`
6. `recent_multi_attempt_rate_10`
7. `recent_hint_usage_rate_10`
8. `recent_hint_available_rate_10`
9. `recent_response_log_mean_10`
10. `recent_response_median_ms_10`
11. `recent_skill_accuracy_5`
12. `recent_skill_accuracy_change_3_vs_previous_2`
13. `has_overall_history`
14. `has_skill_history`
15. `has_full_skill_window_5`
16. `has_recent_hint_usage_evidence`

These predictors represent six broad information categories:

### Overall Performance

- `recent_accuracy_10`
- `recent_accuracy_change_5_vs_10`

### Effort / Attempts

- `recent_attempt_log_mean_10`
- `recent_multi_attempt_rate_10`

### Hint Support

- `recent_hint_usage_rate_10`
- `recent_hint_available_rate_10`

### Response Behaviour

- `recent_response_log_mean_10`
- `recent_response_median_ms_10`

### Concept Performance

- `recent_skill_accuracy_5`
- `recent_skill_accuracy_change_3_vs_previous_2`

### Evidence Strength / Cold Start

- `previous_interaction_count`
- `previous_skill_interaction_count`
- `has_overall_history`
- `has_skill_history`
- `has_full_skill_window_5`
- `has_recent_hint_usage_evidence`

---

# Predictor Redundancy Investigation

Absolute predictor correlations were inspected.

Only one pair reached an absolute correlation of at least 0.80:

`recent_attempt_log_mean_10`

and

`recent_multi_attempt_rate_10`

Correlation:

`0.823`

Both were retained at this stage because they represent different behavioural interpretations.

Final redundancy decisions may later be revisited using model validation, feature importance, ablation testing, or regularization.

---

# Predictor Missingness

Missing historical feature values are intentionally preserved where evidence does not exist.

Important missingness:

- `recent_skill_accuracy_change_3_vs_previous_2`: 44.70%
- `recent_skill_accuracy_5`: 12.42%
- `recent_hint_usage_rate_10`: 4.74%
- Several overall-history features: approximately 1.46%

These missing values are structurally meaningful.

They generally indicate:

- Student cold start
- Concept cold start
- Insufficient concept history
- Unavailable hint evidence

Therefore, they must not automatically be interpreted as poor performance or replaced with arbitrary zero values.

Evidence indicators accompany important missing-history situations.

---

# Leakage Prevention Policy

A central feature-engineering rule was enforced:

> No predictor may directly use behavioural information from the current interaction.

The current interaction's:

- Correctness
- Attempt count
- Hint behaviour
- Response time

are excluded from historical predictor calculations.

All rolling and cumulative features use previous interactions only.

`correct` remains in the exported research dataset but is not included in the 16 predictors.

This separation is essential for realistic future-state inference and prevents target leakage.

---

# Final Feature Engineering Validation

A final validation suite was executed.

Checks included:

- Expected row count
- Expected student count
- Expected skill count
- Unique `order_id + skill_id`
- All 16 predictors exist
- Current `correct` excluded from predictors
- Current behavioural fields excluded
- Non-negative history counts
- Accuracy ranges
- Change-feature ranges
- Evidence-indicator consistency
- Binary evidence-indicator validation

Results:

`24 / 24 checks passed`

Final result:

`PASS`

---

# Final Engineered Feature Dataset

The final feature dataset was exported as:

`data/processed/learning_state_features.csv`

Final shape:

`338,001 rows × 23 columns`

Dataset statistics:

- Rows: 338,001
- Columns: 23
- Unique students: 4,163
- Unique skills: 123
- Unique `order_id`: 283,105
- Duplicate `order_id + skill_id`: 0
- Final predictors: 16
- Current `correct` included as predictor: False

File size:

Approximately `52.58 MB`

The file was saved in the project-level:

`data/processed/`

directory.

No duplicate feature dataset was created inside `notebooks/`.

---

# Export and Reload Validation

The saved CSV was reloaded from disk.

Validation results:

- Reloaded shape: `(338001, 23)`
- Shape preserved: True
- Column order preserved: True
- Duplicate `order_id + skill_id`: 0

Final export result:

`PASS`

---

# Phase 4 — Final Status

## FEATURE ENGINEERING COMPLETE

The project now has a validated historical feature dataset suitable for the next Student Learning State Identification research stage.

Completed:

- Base interaction reconstruction
- Concept interaction reconstruction
- Overall student-history engineering
- Concept-history engineering
- Recent performance engineering
- Trend/change engineering
- Robust attempt engineering
- Hint evidence engineering
- Robust response-time engineering
- Cold-start representation
- Evidence-strength indicators
- Leakage validation
- Predictor selection
- Final feature validation
- Processed dataset export
- Reload validation

The original raw dataset remains unchanged.

The final engineered dataset is:

`data/processed/learning_state_features.csv`

---

# Current Project Position

The project has now progressed through:

1. Project Setup — COMPLETE
2. Data Understanding / EDA — COMPLETE
3. Data Preprocessing and Interaction Construction — COMPLETE
4. Historical Feature Engineering — COMPLETE

The next major phase is:

# Phase 5 — Learning-State Definition and Label Construction

The ASSISTments dataset does not provide direct ground-truth labels for the intended student learning states.

Therefore, before training a machine-learning classifier, the project must develop and validate a defensible methodology for deriving learning-state labels from the historical evidence.

The intended learning states must be finalized before label generation.

The next phase must determine:

- Exact state definitions
- Which historical signals define each state
- How performance and trend interact
- How hints and attempts contribute as supporting evidence
- How concept-specific and overall evidence interact
- How cold-start cases are handled
- Minimum evidence requirements
- Whether confidence/evidence strength should accompany each state
- How generated labels will be validated
- How class balance will be evaluated
- Whether the state methodology is suitable as training supervision for the ML model

No ML classifier should be trained until this state-definition methodology has been completed and validated.

+---

# Phase 4 — Learning-State Feature Engineering

## Purpose

The purpose of Phase 4 was to transform the validated Base Interaction and Concept Interaction representations into a leakage-safe historical feature dataset for Student Learning State Identification.

The central requirement was:

> The learning state associated with a current interaction must be inferred only from information available before that interaction.

Therefore, the current interaction's outcome or behavioural information must not leak into the historical predictors.

---

## Historical Feature Design

Feature engineering was performed using two complementary history sources.

### General Student History

General history represents the student's behaviour across previous interactions regardless of mathematical concept.

Examples include:

- Previous interaction count
- Recent overall accuracy
- Recent overall accuracy change
- Recent attempt behaviour
- Recent hint behaviour
- Recent response-time behaviour

### Concept-Specific History

Concept history represents the student's previous performance on the same mathematical skill.

Concept histories use the validated student-skill ordering based on the processed Concept Interaction representation.

Examples include:

- Previous skill interaction count
- Recent skill accuracy
- Concept-specific performance change

This distinction is necessary because a student may perform strongly overall while struggling with a particular mathematical concept, or vice versa.

---

## Leakage-Safe Historical Construction

All historical features were constructed from interactions occurring before the current logical interaction.

The current interaction's:

- `correct`
- attempt behaviour
- hint behaviour
- response-time behaviour

were not directly used as predictors for that same interaction.

The `correct` field is retained in the engineered dataset for later evaluation and modelling purposes but is not part of the final predictor list.

Leakage validation was performed manually on sampled concept histories.

For the concept-change feature:

- Prior-two calculations matched manual reconstruction.
- Recent-three calculations matched manual reconstruction.
- Change calculations matched manual reconstruction.
- All tested leakage-validation checks passed.

---

## General Historical Performance Features

General historical performance is calculated from previous Base Interactions.

Important engineered signals include:

### `previous_interaction_count`

Number of previous overall student interactions available before the current interaction.

### `recent_accuracy_10`

Accuracy across up to the previous 10 overall interactions.

### `recent_accuracy_change_5_vs_10`

Difference between very recent and broader recent overall performance.

This provides an overall directional learning signal without using the current interaction.

---

## Historical Attempt Features

Attempt behaviour contains severe raw outliers, so raw arithmetic attempt counts were not used directly as the primary historical representation.

The engineered historical attempt features are:

### `recent_attempt_log_mean_10`

Average log-transformed attempt behaviour across recent previous interactions.

This reduces the influence of extreme attempt-count observations.

### `recent_multi_attempt_rate_10`

Proportion of recent previous interactions requiring multiple attempts.

This feature showed a clear relationship with learning performance.

Observed recent skill accuracy decreased as multi-attempt rate increased.

For interpretable multi-attempt groups:

- 0% multi-attempt behaviour → average recent skill accuracy approximately 0.767
- >60% multi-attempt behaviour → average recent skill accuracy approximately 0.184

Therefore, repeated attempts provide useful supporting evidence of learning difficulty.

---

## Historical Hint Features

The final historical hint representations include:

### `recent_hint_usage_rate_10`

Rate of measurable hint use across recent previous interactions.

### `recent_hint_available_rate_10`

Rate at which hint evidence was structurally available.

These are kept separate because:

> Measured zero hint usage is not equivalent to missing hint evidence.

Observed relationships showed that hint usage is strongly associated with lower recent performance.

For example:

- 0% measured hint usage → average recent skill accuracy approximately 0.746
- >60% hint usage → average recent skill accuracy approximately 0.197

Therefore, hint behaviour provides important support/effort evidence.

---

## Historical Response-Time Features

The final historical response-time representations include:

### `recent_response_log_mean_10`

Log-transformed mean recent response time.

### `recent_response_median_ms_10`

Median recent response time in milliseconds.

Response-time analysis showed substantially weaker and less monotonic relationships with learning performance than accuracy, hint use, or repeated attempts.

Therefore:

> Response time is retained as secondary contextual evidence rather than treated as a primary learning-state determinant.

---

## Concept-Specific Historical Features

Concept-specific history was constructed separately for each:

`user_id + skill_id`

using the validated Concept Interaction ordering.

### `previous_skill_interaction_count`

Number of previous interactions the student has had with the current skill.

Across 338,001 Concept Interactions:

- No previous skill history: 41,982 interactions
- Percentage with no previous skill history: 12.42%
- Maximum previous skill interactions: 289

---

## Historical Skill Accuracy

### `recent_skill_accuracy_5`

Represents recent performance on the current mathematical skill using up to the previous five skill interactions.

Availability:

- Available: 296,019 interactions
- Missing: 41,982
- Availability: 87.58%

It is unavailable exactly when no previous interaction exists for that student-skill pair.

Skill-specific performance differs meaningfully from overall student performance.

Among interactions where both were available:

- Correlation between recent overall accuracy and recent skill accuracy was approximately 0.666.

Important disagreement cases were observed where:

- overall performance was high but concept performance was low;
- concept performance was high but overall performance was low.

This confirms that concept-specific memory contains information that cannot be replaced by overall student performance.

---

## Concept-Specific Trend Feature

The final concept-change feature is:

### `recent_skill_accuracy_change_3_vs_previous_2`

It compares:

- accuracy across the most recent three previous interactions with the skill

against:

- accuracy across the two previous interactions immediately before those three.

The feature therefore requires at least five previous skill interactions.

Availability:

- Available: 186,916 interactions
- Missing: 151,085
- Availability: 55.30%

Range:

`-1.0 to +1.0`

Manual leakage validation confirmed that only historical interactions are used.

---

## Concept-History Evidence Strength

Concept-history depth varies substantially.

Across the final Concept Interaction feature table:

- Full five-interaction skill history: 186,916 interactions (55.30%)
- Partial one-to-four skill interactions: 109,103 (32.28%)
- No previous skill history: 41,982 (12.42%)

Therefore, history-dependent features cannot be required uniformly for every prediction.

The learning-state system must explicitly represent evidence availability.

---

## Cold-Start / Evidence Indicators

Four explicit evidence indicators were created.

### `has_overall_history`

Indicates whether previous overall student history exists.

Evidence available:

98.54%

### `has_skill_history`

Indicates whether previous history for the current skill exists.

Evidence available:

87.58%

### `has_full_skill_window_5`

Indicates whether at least five previous skill interactions are available.

Evidence available:

55.30%

### `has_recent_hint_usage_evidence`

Indicates whether measurable recent hint-usage evidence exists.

Evidence available:

95.26%

These indicators allow the downstream model to distinguish a real zero measurement from unavailable historical evidence.

---

## Final Predictor Set

The final engineered predictor set contains 16 predictors:

1. `previous_interaction_count`
2. `previous_skill_interaction_count`
3. `recent_accuracy_10`
4. `recent_accuracy_change_5_vs_10`
5. `recent_attempt_log_mean_10`
6. `recent_multi_attempt_rate_10`
7. `recent_hint_usage_rate_10`
8. `recent_hint_available_rate_10`
9. `recent_response_log_mean_10`
10. `recent_response_median_ms_10`
11. `recent_skill_accuracy_5`
12. `recent_skill_accuracy_change_3_vs_previous_2`
13. `has_overall_history`
14. `has_skill_history`
15. `has_full_skill_window_5`
16. `has_recent_hint_usage_evidence`

The current interaction's `correct` field is explicitly excluded from this predictor list.

No direct current-interaction behavioural field is included as a predictor.

---

## Predictor Correlation Review

High absolute correlations among final predictors were checked.

Only one pair exceeded an absolute correlation of 0.80:

- `recent_attempt_log_mean_10`
- `recent_multi_attempt_rate_10`

Correlation:

approximately `0.823`

Both were retained at this stage because they represent related but different behavioural interpretations:

- magnitude of attempt effort;
- frequency of requiring multiple attempts.

Their influence can later be assessed during modelling and feature-importance analysis.

---

## Final Feature Engineering Validation

A dedicated final validation suite was executed.

Validated requirements included:

- Expected row count
- Expected student count
- Expected skill count
- Unique `order_id + skill_id`
- All 16 predictors exist
- Current `correct` is not a predictor
- No direct current-interaction behavioural field is a predictor
- History counts are non-negative
- Accuracy features remain within valid ranges
- Rate features remain within `[0,1]`
- Trend features remain within expected ranges
- Evidence indicators correctly match feature availability
- Evidence indicators contain only binary values

Result:

`24 / 24 checks passed`

Final result:

`PASS`

---

## Final Engineered Feature Dataset

The final engineered feature dataset contains:

- Rows: 338,001
- Columns: 23
- Unique students: 4,163
- Unique skills: 123
- Unique `order_id`: 283,105
- Duplicate `order_id + skill_id`: 0
- Final predictors: 16

Saved as:

`data/processed/learning_state_features.csv`

File size at export:

approximately `52.58 MB`

The dataset was reloaded after saving.

Reload validation confirmed:

- Shape preserved
- Column order preserved
- No duplicate `order_id + skill_id`

Final export result:

`PASS`

---

# Phase 5 — Learning-State Definition and Label Construction

## Purpose

The ASSISTments dataset does not provide direct ground-truth labels for the intended Student Learning State.

Therefore, Phase 5 developed a transparent and evidence-based derived state methodology using the historical behavioural features created in Phase 4.

The originally considered five-state structure was simplified to three operational learning states:

- `NEEDS_SUPPORT`
- `DEVELOPING`
- `STRONG`

A separate:

- `UNAVAILABLE`

value is used for true cold-start cases where no previous learning evidence exists.

`UNAVAILABLE` is not considered one of the three learning states.

---

## State-Definition Investigation

The state-definition methodology was investigated through a sequence of analyses covering:

1. Core performance distributions
2. Overall vs concept-specific performance
3. Trend signals
4. Attempt behaviour
5. Hint behaviour
6. Response-time behaviour
7. Combined evidence patterns
8. Evidence strength and cold-start behaviour
9. Candidate three-state definition
10. Boundary and contradiction analysis
11. Candidate state and confidence/evidence validation
12. Methodology consolidation
13. Threshold sensitivity analysis
14. Final label generation and validation
15. Final labelled-dataset export and validation

No label dataset was saved until the methodology had passed the required validation stages.

---

## Primary Performance Principle

The state methodology prioritizes concept-specific evidence whenever it exists.

### If previous skill history exists

Use:

`recent_skill_accuracy_5`

This represents recent historical performance on the current mathematical concept.

### If no previous skill history exists but overall history exists

Use:

`recent_accuracy_10`

This provides a general student-performance estimate during concept cold start.

### If no previous overall history exists

No historical performance evidence is available.

The state is:

`UNAVAILABLE`

This avoids inventing a learning state for a student with no previous evidence.

---

## Evidence Levels

Every interaction is assigned one evidence level.

### 1. `COLD_START`

No previous overall history and no previous skill history.

Interactions:

`4,940` (1.46%)

### 2. `OVERALL_ONLY`

Previous overall history exists, but the student has no previous history for the current skill.

Interactions:

`37,042` (10.96%)

### 3. `PARTIAL_SKILL`

The student has between one and four previous interactions with the current skill.

Interactions:

`109,103` (32.28%)

### 4. `FULL_SKILL`

At least five previous interactions with the current skill are available.

Interactions:

`186,916` (55.30%)

All 338,001 interactions are assigned exactly one evidence level.

---

## Evidence Strength

Evidence levels are mapped to interpretable evidence strengths:

- `COLD_START` → `NONE`
- `OVERALL_ONLY` → `LOW`
- `PARTIAL_SKILL` → `MEDIUM`
- `FULL_SKILL` → `HIGH`

This separates:

> predicted learning state

from:

> strength of historical evidence supporting that state.

Therefore, two students may receive the same state while having different evidence strengths.

---

## Support Behaviour Context

Attempt and hint behaviour were analysed as supporting evidence.

The derived support context includes:

- `LOW_SUPPORT_SIGNAL`
- `MIXED_SUPPORT_SIGNAL`
- `HIGH_SUPPORT_SIGNAL`
- `UNAVAILABLE`

Strong combined profiles were observed.

For example:

### Low concept performance + high attempts + high hints

Interactions:

`9,400`

Average skill accuracy:

approximately `0.125`

### High concept performance + low attempts + low hints

Interactions:

`88,636`

Average skill accuracy:

approximately `0.921`

These results support the behavioural interpretation of the performance states.

However, support behaviour does not independently overwrite the primary performance-based state.

---

## Trend Context

Trend information is retained as contextual information.

The final trend context includes:

- `IMPROVING_DIRECTION`
- `DECLINING_DIRECTION`
- `STABLE_DIRECTION`
- `UNAVAILABLE`

When sufficient concept history exists, concept-specific trend is preferred.

Otherwise, available overall historical trend is used.

Trend was not used to create additional learning-state classes.

For example, an improving student in the middle performance region remains:

`DEVELOPING`

while the improving direction can be retained separately as context.

---

## Final State Boundaries

After distribution analysis, behavioural comparison, contradiction analysis, and threshold sensitivity analysis, the selected state boundaries are:

### `NEEDS_SUPPORT`

`primary_performance <= 0.40`

### `DEVELOPING`

`primary_performance > 0.40 and primary_performance < 0.80`

### `STRONG`

`primary_performance >= 0.80`

### `UNAVAILABLE`

No previous overall historical evidence exists.

---

## Threshold Sensitivity Analysis

Alternative lower thresholds:

- 0.35
- 0.40
- 0.45

Alternative upper thresholds:

- 0.75
- 0.80
- 0.85

were compared.

The selected:

`0.40 / 0.80`

method was used as the reference.

Important findings included:

- Moving the lower threshold from 0.40 to 0.45 changed only 88 interactions, approximately 0.03%.
- Moving the lower threshold from 0.40 to 0.35 changed 28,494 interactions, approximately 8.56%.
- Moving the upper threshold from 0.80 to 0.75 changed 7,163 interactions, approximately 2.15%.
- Moving the upper threshold from 0.80 to 0.85 changed 60,106 interactions, approximately 18.05%.

A large number of observations occur exactly at discrete rolling-accuracy values such as:

- 0.40
- 0.60
- 0.80
- 1.00

Therefore, threshold changes can move substantial groups of interactions simultaneously.

The selected thresholds were retained after sensitivity analysis.

---

## Final Derived Learning-State Distribution

Across all 338,001 Concept Interaction rows:

- `STRONG`: 145,457 (43.03%)
- `NEEDS_SUPPORT`: 100,645 (29.78%)
- `DEVELOPING`: 86,959 (25.73%)
- `UNAVAILABLE`: 4,940 (1.46%)

Among the 333,061 interactions eligible for one of the three learning states:

- `STRONG`: 145,457 (43.67%)
- `NEEDS_SUPPORT`: 100,645 (30.22%)
- `DEVELOPING`: 86,959 (26.11%)

The class distribution is therefore not perfectly balanced, but all three operational learning states have substantial representation.

---

## Performance Source Distribution

Final state generation uses three performance-source conditions.

### Skill-based

`296,019` interactions have previous skill history and use concept-specific performance.

### Overall-based

`37,042` interactions have overall history but no previous history for the current skill.

These use overall historical performance.

### Unavailable

`4,940` true cold-start interactions contain no previous overall performance evidence.

These receive `UNAVAILABLE`.

---

## Contradictory Behaviour Profiles

Performance and support behaviour are not always perfectly aligned.

Important contradictory profiles included:

### Low performance + low support signal

Interactions:

`18,393`

### Strong performance + high support signal

Interactions:

`798`

These cases demonstrate why support behaviour should not blindly overwrite the primary performance signal.

Instead, support behaviour is retained as contextual evidence that can later support model interpretation, confidence analysis, and personalization.

---

## Final Label Leakage Protection

The final learning-state label is derived from historical information only.

The current interaction's direct:

- correctness
- attempt behaviour
- hint behaviour
- response time

does not determine the historical state assigned before that interaction.

The current `correct` field remains available in the dataset but is not part of the state-assignment inputs.

Final leakage safety check:

`PASS`

---

## Final Label Generation Validation

The in-memory final label generation was validated against:

- Expected total rows
- Expected eligible rows
- Expected cold-start rows
- Missing-state checks
- Allowed-state domains
- Cold-start behaviour
- Skill performance-source selection
- Overall-only performance-source selection
- `NEEDS_SUPPORT` boundary
- `DEVELOPING` lower boundary
- `DEVELOPING` upper boundary
- `STRONG` boundary
- Direct current-field leakage protection

Result:

`13 / 13 checks passed`

Final label-generation result:

`PASS`

---

## Final Labelled Dataset

The final labelled research dataset contains:

- Rows: 338,001
- Columns: 30
- Unique students: 4,163
- Unique skills: 123
- Unique `order_id`: 283,105
- Duplicate `order_id + skill_id`: 0

It contains the original 23 engineered feature-dataset columns plus:

- `learning_state`
- `evidence_level`
- `evidence_strength`
- `performance_source`
- `primary_performance`
- `trend_context`
- `support_context`

Saved as:

`data/processed/learning_state_labelled.csv`

File size at export:

approximately `76.11 MB`

The original:

`data/processed/learning_state_features.csv`

was not modified.

---

## Final Labelled Dataset Export Validation

The saved labelled dataset was reloaded and validated.

Checks included:

- Expected 338,001 rows
- Expected 30 columns
- Expected 4,163 students
- Expected 123 skills
- No duplicate `order_id + skill_id`
- No missing `learning_state`
- Correct `STRONG` count
- Correct `NEEDS_SUPPORT` count
- Correct `DEVELOPING` count
- Correct `UNAVAILABLE` count
- Column order preserved

Result:

`11 / 11 checks passed`

Final labelled-dataset export result:

`PASS`

---

# Current Project Status

The following major stages are now complete:

- Data Understanding — COMPLETE
- Data Preprocessing and Interaction Construction — COMPLETE
- Historical Feature Engineering — COMPLETE
- Three-State Learning-State Definition — COMPLETE
- Learning-State Label Generation — COMPLETE
- Final Labelled Dataset Export — COMPLETE

Validated processed research datasets now include:

`data/processed/learning_state_features.csv`

and:

`data/processed/learning_state_labelled.csv`

The raw ASSISTments dataset remains preserved unchanged.

The project now has a reproducible historical feature representation and an evidence-aware derived three-state learning-state target.

---

# Next Phase — Model Development

The next phase will develop and evaluate machine-learning models for predicting:

- `NEEDS_SUPPORT`
- `DEVELOPING`
- `STRONG`

using the historical predictor set.

True cold-start `UNAVAILABLE` interactions will be handled explicitly rather than treated as a fourth learning-state class.

The modelling phase must include:

1. Define the modelling dataset.
2. Exclude or explicitly route true cold-start rows.
3. Separate predictor columns from identifiers and research metadata.
4. Define a leakage-safe train/validation/test strategy.
5. Define missing-value handling.
6. Establish simple baselines.
7. Train candidate classification models.
8. Compare models using suitable multi-class metrics.
9. Inspect class-wise performance and confusion matrices.
10. Evaluate feature importance and model behaviour.
11. Check robustness across evidence-strength groups.
12. Select and save the final learning-state model.
13. Document model limitations.
14. Prepare the selected model for integration with the Student Personalization Memory Layer.

No final production model has yet been selected.

+---

# Phase 6 â€” Learning-State Model Development, Evaluation & Deployment Preparation

## Objective

The objective of Phase 6 was to develop and evaluate a machine-learning model capable of predicting the derived student learning states:

- `NEEDS_SUPPORT`
- `DEVELOPING`
- `STRONG`

The `UNAVAILABLE` state was reserved for cold-start interactions where insufficient historical evidence existed and was therefore excluded from classifier training.

A major methodological requirement of this phase was to avoid using the current interaction outcome or other direct current-interaction behavioural information when predicting the student's state.

---

## Step 1 â€” Modelling Dataset Preparation

Source dataset:

```text
data/processed/learning_state_labelled.csv
```

Dataset shape:

```text
338,001 rows
30 columns
```

The modelling dataset contained:

```text
Eligible modelling interactions: 333,061
Cold-start interactions:           4,940
```

The cold-start rows corresponded to interactions without previous overall or skill history and retained the state:

```text
UNAVAILABLE
```

Sixteen initial leakage-safe historical predictors were approved.

The target distribution among eligible interactions was:

```text
STRONG          145,457   43.67%
NEEDS_SUPPORT   100,645   30.22%
DEVELOPING       86,959   26.11%
```

All Step 1 validation checks passed:

```text
17 / 17
```

---

## Step 2 â€” Student-Aware Split Investigation

The eligible modelling data contained:

```text
333,061 interactions
4,030 students
```

The analysis confirmed that all three learning states were represented across sufficiently large numbers of students.

Students representing each state:

```text
NEEDS_SUPPORT: 3,347
DEVELOPING:    3,267
STRONG:        3,474
```

Approximately 66.97% of eligible students had interactions belonging to all three learning states.

This confirmed that a student-level split was feasible.

The key methodological decision was:

> The same `user_id` must never appear across multiple model-evaluation partitions.

---

## Step 3 â€” Final Student-Aware Train / Validation / Test Split

A student-aware split was created using student-level stratification based on dominant learning state and interaction-count groups.

Final student counts:

```text
Train:       2,821 students
Validation:    604 students
Test:          605 students
Total:       4,030 students
```

Final interaction counts:

```text
Train:       236,314
Validation:   45,774
Test:         50,973
Total:       333,061
```

No student overlap existed between:

```text
Train â†” Validation
Train â†” Test
Validation â†” Test
```

All three states were represented in every partition.

All Step 3 validation checks passed:

```text
16 / 16
```

The held-out test partition was reserved and not used during model development or hyperparameter selection.

---

## Step 4 â€” Missing-Value Handling

Missing values were confirmed to be primarily structural rather than arbitrary.

The main missing features were:

```text
recent_skill_accuracy_change_3_vs_previous_2
recent_skill_accuracy_5
recent_hint_usage_rate_10
```

Median imputation values were learned exclusively from the training partition:

```text
recent_skill_accuracy_change_3_vs_previous_2 = 0.0
recent_skill_accuracy_5                      = 0.6
recent_hint_usage_rate_10                    = 0.1
```

Validation and test statistics were never used to determine imputation values.

After preprocessing:

```text
Train missing values:      0
Validation missing values: 0
Test missing values:       0
```

All Step 4 validation checks passed:

```text
11 / 11
```

---

## Step 5 â€” Final Preprocessing Strategy

The initial predictor set contained 16 features.

The feature:

```text
has_overall_history
```

was constant across all eligible modelling rows and was removed.

This produced:

```text
15 modelling predictors
```

Two preprocessing paths were prepared:

- scaled preprocessing for scale-sensitive models;
- original-scale preprocessing for tree-based models.

Both pipelines used training-only fitted preprocessing information.

All Step 5 validation checks passed:

```text
16 / 16
```

---

## Step 6 â€” Dummy Baseline

A majority-class dummy classifier was evaluated first.

Validation performance:

```text
Accuracy:          0.4333
Balanced Accuracy: 0.3333
Macro Precision:   0.1444
Macro Recall:      0.3333
Macro F1:          0.2015
Weighted F1:       0.2620
```

The classifier predicted:

```text
STRONG
```

for every validation interaction.

Per-class recall:

```text
NEEDS_SUPPORT: 0.0000
DEVELOPING:    0.0000
STRONG:        1.0000
```

This established the minimum baseline that useful models needed to exceed.

---

## Step 7 â€” Logistic Regression Reference Model

A multiclass Logistic Regression model using the full predictor set achieved:

```text
Accuracy:          0.9991
Balanced Accuracy: 0.9989
Macro F1:          0.9990
Weighted F1:       0.9991
```

Dominant coefficients included:

```text
recent_skill_accuracy_5    29.6127
recent_accuracy_10          4.7182
has_full_skill_window_5     3.7910
has_skill_history           3.7779
```

The near-perfect performance required careful methodological interpretation.

The derived learning-state labels were principally determined using recent skill accuracy or recent overall accuracy. Therefore, supplying those same variables to a classifier allows the classifier to reproduce the heuristic label-generation function.

Consequently:

> The near-perfect Logistic Regression result demonstrates reproduction of the derived labeling rule, not independent validation of the underlying learning-state construct.

The accepted Logistic Regression reference remained:

```text
Macro F1:          0.9990
Balanced Accuracy: 0.9989
```

---

## Step 8 â€” Full Decision Tree

A Decision Tree trained using the full predictor set achieved:

```text
Training Accuracy:   1.0000
Validation Accuracy: 1.0000
Training Macro F1:   1.0000
Validation Macro F1: 1.0000
```

Tree structure:

```text
Depth:      5
Leaf nodes: 6
Total nodes: 11
```

Only three features were required:

```text
recent_skill_accuracy_5    0.810098
recent_accuracy_10         0.099212
has_skill_history          0.090690
```

The tree directly reconstructed the label-generation logic, including the routing between skill-level and overall historical performance and the derived state thresholds.

Therefore, this model was useful as a methodological diagnostic but was not selected as the behavioural deployment model.

---

## Step 9 â€” Direct Label-Feature Ablation

To investigate whether learning states could be inferred without directly using the variables that defined the labels, the following direct label-defining predictors were removed:

```text
recent_skill_accuracy_5
recent_accuracy_10
has_skill_history
```

This produced a 12-feature behavioural-signal predictor set.

Results:

```text
FULL_15
Macro F1: 1.0000

NO_DIRECT_LABEL_FEATURES
Macro F1: 0.8504
Balanced Accuracy: 0.8499

SUPPORT_TREND_CONTEXT_ONLY
Macro F1: 0.8504
Balanced Accuracy: 0.8499
```

The two ablated groups were defined using the same 12 predictors and therefore produced identical results.

Ablated per-class recall:

```text
NEEDS_SUPPORT: 0.8306
DEVELOPING:    0.8298
STRONG:        0.8893
```

Important behavioural predictors included:

```text
Skill accuracy trend
Recent hint usage
Recent multi-attempt rate
Overall accuracy trend
Previous skill history
```

However, the unrestricted ablated Decision Tree was highly complex:

```text
Depth:      44
Leaf nodes: 21,345
```

Its performance was:

```text
Training Macro F1:   0.9993
Validation Macro F1: 0.8504
Gap:                 0.1489
```

This indicated substantial overfitting.

---

## Step 10 â€” Parsimonious Behavioural Decision Tree

A controlled search was performed to reduce overfitting while preserving predictive performance.

The selected configuration was:

```text
max_depth:         15
min_samples_leaf: 100
class_weight:     None
```

Selected tree:

```text
Depth:      15
Leaf nodes: 746
```

Performance:

```text
Training Macro F1:   0.8788
Validation Macro F1: 0.8756
Balanced Accuracy:   0.8768
Trainâ€“validation gap: 0.0032
```

Per-class validation recall:

```text
NEEDS_SUPPORT: 0.8640
DEVELOPING:    0.8615
STRONG:        0.9048
```

Leading feature importances:

```text
Skill accuracy trend       0.439581
Recent hint usage          0.168685
Recent multi-attempt rate  0.138640
Previous skill history     0.103488
Overall accuracy trend     0.103155
```

Compared with the unrestricted ablated tree, the tuned model was dramatically simpler, generalized better, and achieved higher validation performance.

---

## Step 11 â€” Random Forest Comparison

A Random Forest was evaluated using the same behavioural-signal predictor set.

Selected configuration:

```text
n_estimators:     200
max_depth:         15
min_samples_leaf:  50
class_weight: balanced
```

Performance:

```text
Training Macro F1:   0.8664
Validation Macro F1: 0.8603
Balanced Accuracy:   0.8648
Trainâ€“validation gap: 0.0061
```

Per-class recall:

```text
NEEDS_SUPPORT: 0.8547
DEVELOPING:    0.8782
STRONG:        0.8614
```

Comparison:

```text
Parsimonious Decision Tree Macro F1: 0.8756
Random Forest Macro F1:              0.8603
Difference:                         +0.0153
```

The Decision Tree remained the strongest behavioural-signal candidate.

---

## Step 12 â€” Final Model Selection

The final selected model was frozen as:

```text
Parsimonious Ablated Decision Tree
```

Frozen hyperparameters:

```text
max_depth:         15
min_samples_leaf: 100
class_weight:     None
```

Frozen predictor count:

```text
12 behavioural predictors
```

The direct label-defining predictors were intentionally excluded.

Validation performance at model-selection time:

```text
Macro F1:          0.8756
Balanced Accuracy: 0.8768
Trainâ€“validation gap: 0.0032
```

After this point, no further tuning was permitted using the held-out test set.

---

## Step 13 â€” One-Time Held-Out Test Evaluation

The frozen model was evaluated once on the untouched held-out test partition.

Test size:

```text
50,973 interactions
605 students
```

Final test performance:

```text
Accuracy:          0.8758
Balanced Accuracy: 0.8731
Macro Precision:   0.8706
Macro Recall:      0.8731
Macro F1:          0.8717
Weighted F1:       0.8760
```

Per-class results:

```text
                 Precision   Recall      F1

NEEDS_SUPPORT       0.8548   0.8614   0.8581
DEVELOPING          0.8452   0.8639   0.8544
STRONG              0.9117   0.8939   0.9027
```

Generalization remained stable:

```text
Training Macro F1:   0.8788
Validation Macro F1: 0.8756
Test Macro F1:       0.8717
```

Performance gaps:

```text
Train â†’ Validation: 0.0032
Validation â†’ Test:  0.0039
Train â†’ Test:       0.0070
```

This provided evidence that the selected behavioural model generalized consistently to unseen students.

### Evidence-Level Test Performance

Test Macro F1 by evidence level:

```text
OVERALL_ONLY:  0.8603
PARTIAL_SKILL: 0.7058
FULL_SKILL:    0.9715
```

The substantially lower performance for `PARTIAL_SKILL` interactions is an important limitation.

The model performs particularly strongly when sufficient skill-specific historical evidence exists but is less reliable when only a small amount of skill history is available.

The held-out test results were frozen after this evaluation and were not used for additional tuning.

---

## Step 14 â€” Deployment Refit

After final model selection and one-time test evaluation, the frozen model configuration was refitted using:

```text
Train + Validation
```

The held-out test partition remained excluded.

Deployment-refit data:

```text
Training interactions: 282,088
Training students:       3,425

Reserved test rows:     50,973
Reserved test students:    605

Student overlap: 0
```

Final deployment imputation values included:

```text
recent_hint_usage_rate_10                       = 0.1
recent_skill_accuracy_change_3_vs_previous_2   = 0.0
```

The final deployment tree retained the frozen hyperparameters:

```text
max_depth:         15
min_samples_leaf: 100
class_weight:     None
```

Refitted tree structure:

```text
Actual depth: 15
Leaf nodes:   810
```

The increase in leaf count from 746 to 810 was expected because the same frozen configuration was refitted on the larger train-plus-validation dataset.

The held-out test data was not included in deployment refitting.

---

## Step 15 â€” Final Artifact Export & Reload Validation

The final deployment artifacts were exported to:

```text
artifacts/
â”œâ”€â”€ learning_state_model.joblib
â”œâ”€â”€ learning_state_imputer.joblib
â””â”€â”€ learning_state_model_metadata.json
```

Artifact reload validation confirmed:

```text
Features:         12
Configured depth: 15
Actual depth:     15
Leaf nodes:       810

Classes:
DEVELOPING
NEEDS_SUPPORT
STRONG
```

A sample of 1,000 non-test deployment-training interactions was passed through both the in-memory objects and the reloaded artifacts.

Results:

```text
Imputed matrices identical: True
Predictions identical:      True
```

All artifact validation checks passed:

```text
23 / 23
```

### SHA-256 Checksums

Model:

```text
77a54e638bae449b8878e5957b6eeb007a43137ae1d2f7b6cd613532d7dcf3f9
```

Imputer:

```text
fc88fd4625a573182f533896107e929782321ac3bb80bda23626beda1ed9de62
```

Metadata:

```text
2462536e06dbe386fa45412cbe2dcae9f3cd4ce3077d919131ff08ac1822931b
```

The held-out test partition was neither evaluated again nor used for deployment fitting during artifact export.

---

# Final Phase 6 Outcome

Phase 6 successfully produced a frozen, evaluated, and reproducible learning-state prediction model.

Final selected model:

```text
Parsimonious Ablated Decision Tree
```

Final behavioural predictor count:

```text
12
```

Frozen configuration:

```text
max_depth = 15
min_samples_leaf = 100
class_weight = None
```

Final held-out research performance:

```text
Accuracy:          0.8758
Balanced Accuracy: 0.8731
Macro F1:          0.8717
Weighted F1:       0.8760
```

The final deployment model was refitted on train plus validation data and exported together with its fitted imputer and metadata.

## Important Methodological Interpretation

The experiments demonstrated two distinct results.

### 1. Label-Reproduction Result

Models given the direct label-defining historical accuracy features achieved approximately perfect performance.

This occurs because the target learning states were themselves derived primarily from those historical performance values.

These results demonstrate that machine-learning models can reconstruct the heuristic labeling function, but they must not be interpreted as independent evidence that the learning-state construct has been externally validated.

### 2. Behavioural-Signal Prediction Result

After removing the direct label-defining predictors, the selected model used indirect historical behavioural signals such as:

```text
skill-performance trend
overall-performance trend
hint usage
multi-attempt behaviour
history depth
response behaviour
```

The resulting held-out student-level test Macro F1 was:

```text
0.8717
```

This is the primary behavioural prediction result of the modelling phase.

## Known Limitation

Performance varied substantially according to evidence availability.

In particular:

```text
FULL_SKILL Macro F1:    0.9715
PARTIAL_SKILL Macro F1: 0.7058
```

Therefore, predictions made with limited skill-specific history should later be accompanied by evidence/confidence information rather than being treated as equally reliable.

## Phase 6 Status

```text
PHASE 6: COMPLETE
```

Completed outputs:

```text
âœ“ Leakage-safe modelling dataset
âœ“ Student-aware evaluation design
âœ“ Training-only preprocessing
âœ“ Dummy baseline
âœ“ Logistic Regression reference
âœ“ Full Decision Tree diagnostic
âœ“ Direct-label-feature ablation
âœ“ Parsimonious behavioural Decision Tree
âœ“ Random Forest comparison
âœ“ Frozen model selection
âœ“ One-time held-out test evaluation
âœ“ Deployment refit
âœ“ Saved model artifact
âœ“ Saved imputer artifact
âœ“ Saved metadata artifact
âœ“ Artifact reload/reproducibility validation
```

The component is now ready to proceed from offline model development into prediction/integration engineering.
---

# Phase 7 — Production Prediction Engine

## Objective

Phase 7 moved the project from offline model development into a reusable production prediction layer.

The goal was to load the frozen Phase 6 artifacts safely and expose a consistent prediction interface without coupling model inference directly to the database or API layers.

## Prediction Engine

The production prediction engine was implemented in:

```text
src/models/learning_state_model.py
```

Its responsibilities include:

```text
artifact loading
exact feature validation
feature ordering
missing-value handling through the saved imputer
cold-start handling
learning-state prediction
evidence-context generation
typed prediction responses
```

The prediction engine loads the frozen artifacts created during Phase 6:

```text
artifacts/learning_state_model.joblib
artifacts/learning_state_imputer.joblib
artifacts/learning_state_model_metadata.json
```

The exact 12-feature model contract is preserved during inference.

## Evidence-Aware Prediction

Production predictions distinguish between different amounts of available historical evidence.

The supported evidence contexts are:

```text
COLD_START
OVERALL_ONLY
PARTIAL_SKILL
FULL_SKILL
```

These are mapped to evidence strengths:

```text
NONE
LOW
MEDIUM
HIGH
```

This distinction is important because Phase 6 showed that predictive performance varies according to the amount of skill-specific history available.

Cold-start cases return:

```text
learning_state = UNAVAILABLE
model_used = false
```

rather than forcing the model to make an unsupported prediction.

## Typed Prediction Contract

Prediction results were integrated with the project's Pydantic schemas so that model outputs can be serialized consistently and later returned through service and API layers.

The production prediction response preserves:

```text
learning state
evidence level
evidence strength
model-used status
historical evidence context
```

## Student-State Service

A dedicated service layer was implemented in:

```text
src/services/student_state_service.py
```

Its responsibility is orchestration between prepared historical features and the prediction engine.

The separation is:

```text
learning_state_model.py
→ artifact loading and inference

student_state_service.py
→ prediction orchestration

memory_routes.py
→ HTTP interface
```

This prevents model logic from being embedded directly inside API routes.

## End-to-End Prediction Validation

The complete prediction path was tested for:

```text
cold start
overall-only history
partial-skill history
full-skill history
typed serialization
deterministic repeated prediction
evidence-boundary transitions
approved prediction-state domain
cold-start model bypass
```

At Phase 7 completion:

```text
44 tests passed
```

for the complete prediction-path validation suite available at that stage.

## Phase 7 Status

```text
PHASE 7: COMPLETE
```

The frozen Phase 6 model was successfully converted into a reusable, evidence-aware production prediction engine.

---

# Phase 8 — Persistent Student Memory and Production Integration

## Objective

Phase 8 implemented the persistent memory backend required for interaction with the Planner, Tutor, and Evaluator components.

The primary external requirement was that an evaluator should be able to send completed assessment information containing data such as:

```text
student_id
topic
subtopic
assessment questions
student answers
correct/wrong results
identified errors or misconceptions
overall evaluator feedback
```

The memory component must then persist this information, update the student's historical memory, generate model features, predict the student's current learning state, and make that state available to other components.

## Important Production Constraint

The evaluator payload does not necessarily contain all behavioural measurements used during offline model development.

Potentially unavailable values include:

```text
attempt count
hint usage
response time
```

The production design therefore explicitly preserves unavailable behavioural measurements as:

```text
NULL / None
```

They are never fabricated as zero measurements.

## SQLite Persistence Layer

A SQLite persistence layer was implemented.

The database supports storage for:

```text
assessments
assessment interactions
misconception memory
learning-state snapshots
current student memory
```

Database initialization and repository behaviour were tested using temporary databases before the real production database was touched.

## Assessment and Interaction Storage

Completed evaluator assessments are stored together with their individual question-level interactions.

The storage layer supports:

```text
multiple assessments per student
transactional assessment storage
duplicate-question protection
rollback on interaction failure
optional behavioural measurements
```

Unavailable behavioural measurements remain `NULL`.

## Persistent Misconception Memory

A persistent misconception-memory mechanism was implemented.

It provides:

```text
conservative text normalization
deduplication within an assessment
occurrence counting across assessments
student isolation
topic isolation
subtopic isolation
frequency-ordered retrieval
```

Question-level and top-level identified errors are combined conservatively.

Differently worded misconceptions are not automatically assumed to have the same semantic meaning.

## Historical Retrieval

The persistence layer supports:

```text
chronological overall student history
topic/subtopic-specific history
assessment interaction retrieval
leakage-safe history cutoffs
student/context isolation
```

The history retrieval mechanism can exclude the current assessment when required, preventing accidental future-information leakage during feature construction.

## Production Feature Builder

A production feature builder was implemented for the frozen 12-feature model contract.

The builder produces the exact feature ordering expected by the saved model.

It generates historical signals including:

```text
overall history
topic/skill history
accuracy trends
hint behaviour
attempt behaviour
response-time behaviour
history availability
```

Missing behavioural measurements remain unavailable rather than being replaced with artificial observations.

## Saved-Artifact Compatibility

The production feature builder was tested against the saved Phase 6 imputer and model.

A Binuri-style evaluator payload with missing attempt, hint, and response-time information was successfully:

```text
converted into the exact feature contract
accepted by the saved imputer
transformed without NaN or infinite output
accepted by the saved Decision Tree
converted into a valid prediction
```

This demonstrated technical compatibility between the production pipeline and the frozen artifacts.

## Behavioural Coverage

Because production inputs may have different amounts of behavioural telemetry, an explicit behavioural-coverage classification was added.

The supported levels are:

```text
FULL_BEHAVIOURAL_COVERAGE
PARTIAL_BEHAVIOURAL_COVERAGE
CORRECTNESS_ONLY_COVERAGE
```

The system also preserves observation counts for:

```text
attempt measurements
hint measurements
response-time measurements
```

A genuine observed zero counts as an observation.

An unavailable value does not.

## History-to-Prediction Integration

The production read path was connected as:

```text
SQLite history
→ overall and topic history retrieval
→ 12-feature construction
→ behavioural coverage classification
→ StudentStateService
→ saved model prediction
→ typed production response
```

Validated cases include:

```text
cold start
correctness-only history
partial behavioural history
full behavioural history
overall-only evidence
partial-skill evidence
full-skill evidence
assessment-cutoff leakage prevention
```

## Learning-State Persistence

Two forms of state memory are maintained.

### Immutable State History

```text
learning_state_snapshots
```

Each prediction can be preserved as an immutable historical snapshot.

### Current Student Memory

```text
student_memory
```

This stores the latest learning state for a:

```text
student
topic
subtopic
```

context.

The persistence layer also stores:

```text
learning state
evidence level
evidence strength
behavioural coverage
model-used status
observation counts
assessment reference
snapshot reference
```

## Atomic Memory Update Workflow

Initially, assessment storage, misconception updates, and state persistence owned independent SQLite transactions.

This was identified as unsafe for the complete production workflow.

The repositories were therefore extended to support a shared existing connection while preserving their standalone behaviour.

The final memory-update transaction is:

```text
BEGIN
→ assessment and interactions
→ misconception memory
→ uncommitted history retrieval
→ feature generation
→ behavioural coverage
→ learning-state prediction
→ immutable state snapshot
→ current student memory
→ COMMIT
```

If any stage fails:

```text
assessment
interactions
misconceptions
snapshot
current memory
```

from that update are rolled back together.

Previously committed student memory remains intact.

## Memory-Update Service

The complete orchestration path became:

```text
Evaluator assessment
→ assessment storage
→ interaction storage
→ misconception update
→ completed-history feature generation
→ behavioural coverage classification
→ learning-state prediction
→ immutable snapshot persistence
→ current-memory update
→ typed response
```

The completed assessment becomes part of the carried-forward memory used for the student's next learning activity.

## Current Memory and History Retrieval

Read-only retrieval functions were implemented for:

```text
current student memory
chronological learning-state history
```

These operations:

```text
do not rerun the model
do not rebuild features
do not modify database state
```

Correct handling was implemented for:

```text
unknown records
None subtopics
student isolation
topic isolation
subtopic isolation
chronological snapshot ordering
```

## Phase 8 Validation

At Phase 8 completion:

```text
173 tests passed
```

The real database remained untouched throughout Phase 8; all database validation used temporary SQLite databases.

## Phase 8 Status

```text
PHASE 8: COMPLETE
```

The project now had a complete persistent student-memory backend capable of receiving assessment data, accumulating historical evidence, predicting learning state, storing state history, and serving current memory to other components.

---

# Phase 9 — FastAPI Integration and Real Local Validation

## Objective

Phase 9 exposed the completed memory backend as an HTTP service so that the Evaluator, Planner, Tutor, and other project components can interact with it independently.

The API contract was frozen before implementation.

## Local Service Contract

The development service uses:

```text
Base URL:
http://127.0.0.1:8001
```

The three primary memory endpoints are:

```text
POST /memory/update

GET /memory/{student_id}/current

GET /memory/{student_id}/history
```

For current and history retrieval:

```text
topic
```

is a required query parameter and:

```text
subtopic
```

is optional.

## FastAPI Application

The API application structure was implemented under:

```text
src/api/
```

including:

```text
app.py
memory_routes.py
__init__.py
```

A health endpoint was also provided:

```text
GET /health
```

which returns the running service status.

## Memory Update Endpoint

The evaluator-facing endpoint is:

```text
POST /memory/update
```

It accepts the existing Phase 8 `AssessmentMemoryUpdateRequest` schema and delegates directly to the tested `MemoryUpdateService`.

The API route does not duplicate:

```text
database logic
misconception logic
feature engineering
prediction logic
state persistence
```

The successful HTTP path is:

```text
HTTP request
→ Pydantic validation
→ MemoryUpdateService
→ atomic SQLite workflow
→ typed response
→ HTTP 200
```

## Current Memory Endpoint

The current-memory endpoint is:

```text
GET /memory/{student_id}/current
```

with:

```text
topic = required
subtopic = optional
```

It performs direct read-only retrieval from persisted current student memory.

It does not rerun model inference.

Unknown current memory returns:

```text
404 Not Found
```

## Learning-State History Endpoint

The state-history endpoint is:

```text
GET /memory/{student_id}/history
```

with the same topic/subtopic context rules.

It returns immutable state snapshots in chronological order.

Unknown history returns:

```text
200 OK
[]
```

rather than `404`.

## API Validation and Hardening

The HTTP layer was hardened against invalid production input.

Validation includes:

```text
blank student identifiers
blank required topic text
blank question identifiers
negative attempt counts
negative hint counts
negative hint totals
negative response times
hint_count greater than hint_total
blank identified-error descriptions
```

Whitespace normalization was added for identifiers and relevant text fields.

A blank optional subtopic is normalized to:

```text
None
```

rather than being treated as a separate learning context.

## Error Handling

Expected and unexpected internal failures are converted into sanitized HTTP `500` responses.

Internal details such as:

```text
database implementation information
filesystem paths
internal exception messages
model/repository failure details
```

are not exposed to API callers.

The final HTTP semantics include:

```text
successful update           → 200
invalid request             → 422
unknown current memory      → 404
empty history               → 200 with []
internal processing failure → sanitized 500
```

## HTTP End-to-End Validation

The complete external workflow was tested through FastAPI:

```text
POST assessment update
→ GET current state
→ GET state history
→ POST second assessment
→ GET latest current state
→ GET expanded chronological history
```

The tests confirmed:

```text
student isolation
topic isolation
None-subtopic handling
history accumulation
current-memory replacement
chronological immutable snapshots
accumulated history influencing later updates
```

All automated API tests used temporary SQLite databases.

## Automated Validation at Phase 9 Freeze

The final automated results were:

```text
API tests:        61 passed
Full test suite: 234 passed
```

The only remaining warning was a non-blocking FastAPI/Starlette `TestClient` deprecation warning.

## First Controlled Real Local Run

After all isolated tests passed, the actual FastAPI service was started with:

```text
http://127.0.0.1:8001
```

The real health endpoint returned:

```json
{
  "status": "ok",
  "service": "student-personalization-memory"
}
```

A controlled integration record was then intentionally written to the real database using:

```text
student_id = integration_test_s1
topic = Algebra
subtopic = Linear equations
```

The assessment contained two questions:

```text
1 correct
1 incorrect
```

with:

```text
addition error
```

stored as the identified misconception.

No attempt, hint, or response-time measurements were supplied.

## Real Production-Path Result

The real memory update returned:

```text
assessment_id:                  1
snapshot_id:                    1
learning_state:                 DEVELOPING
evidence_level:                 PARTIAL_SKILL
evidence_strength:              MEDIUM
behavioural_coverage:           CORRECTNESS_ONLY_COVERAGE
model_used:                     true
previous_interaction_count:     2
previous_skill_interaction_count: 2
attempt_observation_count:      0
hint_observation_count:         0
response_time_observation_count: 0
misconception_count:            1
memory_updated:                 true
```

This confirmed that a correctness-only evaluator payload can pass through the complete production pipeline without fabricated behavioural observations.

## Real Current-Memory Retrieval

The current-memory endpoint returned the persisted state:

```text
learning_state:      DEVELOPING
evidence_level:      PARTIAL_SKILL
evidence_strength:   MEDIUM
last_assessment_id:  1
last_snapshot_id:    1
```

No model inference was performed during this retrieval.

## Real History Retrieval

The history endpoint returned exactly one immutable snapshot corresponding to:

```text
assessment_id = 1
snapshot_id = 1
learning_state = DEVELOPING
```

## Persistence Across Server Restart

The API server was stopped and restarted on:

```text
127.0.0.1:8001
```

No additional memory update was submitted.

After restart, current-memory retrieval still returned:

```text
student_id = integration_test_s1
learning_state = DEVELOPING
last_assessment_id = 1
last_snapshot_id = 1
```

This confirmed persistent SQLite memory across application restarts.

## Real Database Status

Unlike earlier phases, the real database is no longer untouched.

It now contains the intentionally created controlled integration record:

```text
integration_test_s1
```

This record exists specifically to validate the real production-style API and persistence path.

## Final Phase 9 Architecture

The externally usable component now operates as:

```text
Evaluator
   ↓
POST /memory/update
   ↓
FastAPI
   ↓
MemoryUpdateService
   ↓
SQLite transaction
   ├── assessment memory
   ├── interaction history
   ├── misconception memory
   ├── historical feature generation
   ├── behavioural coverage
   ├── Decision Tree prediction
   ├── immutable learning-state snapshot
   └── current student memory
            ↓
      Planner / Tutor
        ↙         ↘
GET current     GET history
```

## Phase 9 Status

```text
PHASE 9: COMPLETE AND FROZEN
```

Confirmed at freeze:

```text
234 automated tests passed
61 API tests passed
real health check passed
real memory update passed
real current retrieval passed
real history retrieval passed
SQLite persistence passed
server-restart persistence passed
```

No further API changes were made after the Phase 9 freeze.

The component is now ready for final system-readiness work, documentation, integration handoff, final validation, and project freeze.
---

# Phase 10 — Final System Readiness, Documentation and Handoff

## Objective

Phase 10 prepared the completed Student Personalization Memory component for reproducible execution, integration handoff, submission, and final project freeze.

## Runtime and Configuration Audit

The final audit confirmed:

```text
Python runtime: 3.11.0
API entry point: src.api.app:app
Host: 127.0.0.1
Port: 8001
Database: SQLite
Runtime database: database/student_memory.db
Model artifacts: present and checksum-verified
```

No embedded credentials or apparent secrets were identified.

## Dependency Reproducibility

Dependencies were reorganized into:

```text
requirements.txt
requirements-dev.txt
requirements-lock.txt
```

Runtime dependencies were pinned to the known-working environment.

The project also records:

```text
.python-version
.env.example
```

The final environment passed:

```text
pip check
```

without broken dependencies.

## README and Startup Documentation

`README.md` was completed with:

```text
component purpose
architecture
environment setup
runtime installation
development installation
Uvicorn startup command
health check
API contracts
model artifact information
database policy
known model limitations
testing instructions
```

The documented startup command is:

```text
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8001
```

## Database Schema and Runtime Policy

`database/schema.sql` was synchronized from the authoritative `SCHEMA_SQL` in:

```text
src/database/connection.py
```

Fresh-database validation confirmed:

```text
5 tables
8 indexes
```

The final runtime policy is:

```text
database/schema.sql
→ retained as the schema reference

database/student_memory.db
→ mutable runtime state
→ generated automatically when required
→ not required in a clean project distribution
```

## Runtime Manifest

A frozen runtime manifest was created:

```text
artifacts/runtime_manifest.json
```

It records:

```text
Python runtime
API configuration
database paths
artifact paths
artifact SHA-256 hashes
Decision Tree configuration
12-feature contract
held-out metrics
validation baseline
runtime-data policy
```

All three frozen artifact hashes were revalidated successfully.

## Project Cleanup

Final cleanup included:

```text
unused scaffold modules removed
obsolete empty tests removed
package __init__.py files preserved
.gitignore configured
Python caches removed
pytest caches removed
runtime integration database removed
```

The project is designed to begin with no student database and initialize persistent memory at runtime.

## Integration Handoff

The final integration handoff for the Planner, Tutor, and Evaluator components documents:

```text
Base URL
POST /memory/update
GET /memory/{student_id}/current
GET /memory/{student_id}/history
health endpoint
request schema
response schemas
optional behavioural measurements
missing-data policy
HTTP status codes
recommended component interaction flow
```

Downstream components do not need to calculate learning state, historical trends, evidence strength, behavioural coverage, or misconception frequency.

## Final Clean-State Smoke Test

The component was started from a clean state with no runtime database.

Confirmed sequence:

```text
API startup
→ PASS

GET /health
→ PASS
→ database remained absent

first POST /memory/update
→ runtime database automatically created
→ assessment stored
→ prediction generated
→ current memory stored

GET /current
→ returned persisted state

GET /history
→ returned matching chronological snapshot
```

The controlled final smoke-test student was:

```text
final_smoke_s1
```

The returned state was:

```text
learning_state = NEEDS_SUPPORT
evidence_level = PARTIAL_SKILL
evidence_strength = MEDIUM
behavioural_coverage = CORRECTNESS_ONLY_COVERAGE
```

After validation, the smoke-test runtime database was removed so that the final project returns to a clean runtime state.

## Final Automated Validation

Final frozen validation:

```text
Full automated test suite: 234 passed
pip check: PASS
Model artifact hash: MATCH
Imputer artifact hash: MATCH
Metadata artifact hash: MATCH
```

The remaining FastAPI/Starlette `TestClient` deprecation warning is non-blocking.

## Final Project Status

```text
PHASE 10: COMPLETE
```

All planned phases of the Student Personalization Memory component are complete.

The project is ready for:

```text
team integration
demonstration
academic evaluation
submission
future deployment work
```

# PROJECT STATUS

```text
PROJECT COMPLETE AND FROZEN
```

---

# Phase 11 — Complete Requirements and Final-System Design

## Step 1 — Complete Requirements & Current-System Audit

Development continued after Phase 10 because the final client scope requires the completed learning-state memory core to become part of a broader **Student Personalization Memory Layer**.

The continuing project is treated as one system. The established Phase 1–10 behavior remains protected, while missing client capabilities are added through evidence-gated future phases.

Phase 11 Step 1 was requirements and design work only:

```text
Production source changes: 0
Database schema changes:   0
API changes:               0
ML artifact changes:       0
Model retraining:          0
```

### Audit scope

The active repository was inspected to verify the current implementation of:

```text
learning-state Decision Tree
12-feature production builder
saved model, imputer, and metadata
assessment and interaction storage
misconception memory
current memory and immutable snapshots
SQLite persistence
atomic memory updates
FastAPI routes
evidence level and strength
behavioral coverage
validation and testing
runtime and artifact documentation
```

The audit then mapped the full client scope, including:

```text
Topic Extractor
canonical skill ontology
raw interaction source of truth
Short-Term Memory
Long-Term Memory
Concept-Based Memory
existing learning-state intelligence
misconception memory
Tutor and Planner context
Evaluator integration
FAPR-LB context
repair outcomes
Meta-Agent signals
live student-question flow
PostgreSQL and migrations
expanded API
professional UI
security, reliability, and reproducibility
```

### Single-system policy

The current learning-state model, feature builder, evidence semantics, artifacts, and tested update flow remain part of the final system. They are not treated as the definition of the entire Memory component.

The final architecture must preserve these responsibilities:

```text
Memory
→ remembers evidence, builds reproducible summaries,
  extracts canonical skills, predicts the existing learning state,
  and supplies auditable context/signals

Tutor
→ teaches

Evaluator
→ judges correctness, errors, and feedback

Planner
→ plans instruction

FAPR-LB
→ selects repair action

Meta-Agent
→ performs BKT/mastery, knowledge-graph,
  and learning-path reasoning
```

### Requirements results

Thirty requirements were identified and assigned an owner, status, evidence, gap, future phase, expected interface, and validation method.

```text
Total requirements:                  30
COMPLETE:                             7
PARTIAL:                              9
MISSING:                              8
REPLACE / REFACTOR:                   1
EXTERNAL COMPONENT RESPONSIBILITY:    5

Unmapped client requirements:         0
Requirements without owner:           0
Requirements without status:          0
Open in-scope gaps without a phase:    0
```

The primary replacement/refactoring requirement is the production persistence layer. The final PostgreSQL design will be derived from the complete client domain rather than produced by mechanically copying the current SQLite tables. SQLAlchemy, Alembic, constraints, indexes, concurrency validation, and reproducible migrations remain required.

The primary missing research capability is automatic canonical skill extraction from natural-language student questions. Its future evidence path includes keyword and TF-IDF baselines, pretrained and fine-tuned MiniLM experiments, hybrid selection, uncertainty/abstention, held-out evaluation, saved artifacts, and integration tests.

### Durable evidence

Created:

```text
docs/PHASE_11_CURRENT_SYSTEM_AUDIT.md
docs/PHASE_11_REQUIREMENTS_TRACEABILITY_MATRIX.md
```

The traceability matrix is now the master completion checklist. It must be updated after every future phase, and an in-scope requirement may be marked complete only when implementation and durable validation evidence both exist.

### Step result

```text
PHASE 11 — STEP 1 RESULT: PASS
```

The next work is Phase 11 Step 2: freeze the final single-system architecture and the refactoring/migration strategy based on this audit.

## Step 2 — Final Single-System Architecture Cleanup Plan

Phase 11 Step 2 inspected the current source tree, imports, public definitions, services, repositories, feature code, model layer, schemas, API routes, and tests to freeze the exact refactoring direction before PostgreSQL and NLP implementation.

This remained architecture work only:

```text
Production source changes: 0
Database schema changes:   0
API behavior changes:      0
ML artifact changes:       0
Model retraining:          0
```

### Key dependency finding

The current behavior is validated, but several application layers are coupled directly to SQLite:

```text
StudentStateService
→ imports SQLite database paths, connections, and repository functions

MemoryUpdateService
→ opens the concrete SQLite connection and coordinates repository functions

feature and coverage builders
→ import HistoricalInteraction from the SQLite repository module

API current/history routes
→ call the state repository directly
```

The final architecture therefore introduces storage-neutral domain records, repository protocols, query/application services, and a unit-of-work abstraction before PostgreSQL becomes the production adapter.

### Frozen dispositions

```text
Learning-state model/artifacts       KEEP
12-feature definitions/order         KEEP
Behavioral coverage semantics        KEEP
Feature input type                    REFACTOR to domain history
FastAPI application                  KEEP / EXTEND
Existing endpoints                   KEEP / adapt behind services
Pydantic request/response behavior   KEEP / split compatibly
Atomic update workflow               KEEP / refactor to unit of work
Misconception normalization logic    KEEP
Misconception persistence            MIGRATE
SQLite repositories                  MIGRATE, then remove from production
SQLite runtime database              remove as production dependency
Current 234 tests                    KEEP as regression boundary
```

### Target architecture

The target layout separates:

```text
API routers
application services
domain records
repository and unit-of-work protocols
SQLAlchemy/PostgreSQL adapters
feature logic
memory aggregators
learning-state and Topic Extractor models
focused request/response schemas
```

Application services will depend on repository protocols rather than `sqlite3`, file paths, or SQLAlchemy sessions. Concrete PostgreSQL repositories will live in the infrastructure/database layer. Alembic migrations will be the future production schema authority.

### Compatibility rules

The PostgreSQL migration must preserve:

```text
existing endpoint paths and meanings
existing request/response fields
null behavior for unavailable behavioral measurements
current-assessment inclusion semantics
read-only current-state retrieval
chronological isolated history
append-only snapshots and latest current memory
frozen learning-state artifacts and feature order
```

SQLite compatibility code will not be removed until PostgreSQL repository parity, API compatibility, atomicity, constraints, concurrency behavior, and the full regression suite have passed.

### Durable evidence

Created:

```text
docs/PHASE_11_SINGLE_SYSTEM_ARCHITECTURE_CLEANUP_PLAN.md
```

Updated:

```text
docs/PHASE_11_REQUIREMENTS_TRACEABILITY_MATRIX.md
docs/PROJECT_DEVELOPMENT_LOG.md
```

### Step result

```text
PHASE 11 — STEP 2 RESULT: PASS
```

Phase 11 is complete. Phase 12 may begin the PostgreSQL implementation only under the frozen dependency, compatibility, and migration boundaries recorded in the architecture cleanup plan.
