# Semantic review of raw-MD7 randomized proposals

## Scope and rule

This review uses only each leakage-free problem/history snapshot and the frozen
MD7-R2 probability vector recomputed before the action. It does not inspect the
subsequent BKT outcome. It asks whether probability mass on multiple moves is a
semantically plausible proposal distribution; it does not create action rules,
thresholds, or exclusions.

Probabilities below are ordered `generic, probing, focus, telling`.

## Cases

1. **Very confident probing** —
   `0adabdf3-b861-4b0f-b802-451cc208dd0b:1:action:2`. The learner identifies
   the Pythagorean theorem but is unsure how to set it up when 13 is the
   hypotenuse. Vector: `(0.000094, 0.999225, 0.000363, 0.000317)`; entropy
   `0.0071`; non-top-1 probability `0.0008`. A diagnostic/reasoning question is
   coherent, and proportional sampling essentially preserves this strong MD7
   preference.

2. **Middle uncertainty, focus preferred** —
   `f3133173-8f14-44d9-b485-97a42dc45030:1:action:6`. After repeated questions
   about numerator/denominator division, the learner says they do not
   understand. Vector: `(0.025067, 0.262960, 0.689880, 0.022093)`; entropy
   `0.7840`; non-top-1 probability `0.3101`. Focus is a plausible primary move,
   while some probing mass reflects genuine uncertainty rather than a
   semantically implausible alternative.

3. **Highest entropy** —
   `5413cab2-22c8-4536-a66b-66619d887e84:1:action:12`. The learner has reached
   `x^2 + 4x = 96` after a long scaffolded exchange. Vector:
   `(0.188669, 0.184449, 0.358392, 0.268490)`; entropy `1.3472`; non-top-1
   probability `0.6416`. Support, a diagnostic prompt, a targeted next cue, or
   an explicit next method are all linguistically/pedagogically intelligible at
   this transition. The diffuse vector matches the ambiguity of the state.

4. **High telling, but generic top-1** —
   `4220e000-557f-487a-949f-a6b3662998ac:1:action:7`. The learner correctly
   proposes calculating 81 and 144 and then adding. Vector:
   `(0.457083, 0.040484, 0.100922, 0.401510)`; entropy `1.0855`; top gap
   `0.0556`. Generic recognition and telling the final square-root step are
   both plausible continuations. The substantial telling probability is not an
   obvious pathological proposal, though it should remain visible in future
   monitoring.

5. **High telling after continued difficulty** —
   `5413cab2-22c8-4536-a66b-66619d887e84:1:action:15`. The learner says, “I
   don't know about that” after extended work. Vector:
   `(0.401303, 0.036303, 0.223583, 0.338811)`; entropy `1.1884`; top gap
   `0.0625`. Supportive continuation, a targeted cue, and an explicit step all
   have defensible mass; further probing is comparatively disfavored.

6. **Close focus/telling competition** —
   `5413cab2-22c8-4536-a66b-66619d887e84:1:action:10`. The learner correctly
   obtains `4x` after earlier difficulty with distribution. Vector:
   `(0.158394, 0.120110, 0.390513, 0.330984)`; entropy `1.2796`; top gap
   `0.0595`. A focused cue for combining terms and an explicit next step are
   both coherent.

7. **Close probing/focus competition** —
   `b9a3b98d-ab91-4e24-b3f7-39a0623831b9:1:action:4`. The learner identifies
   4 and 5 as the factor pair. Vector: `(0.015517, 0.480723, 0.490936,
   0.012824)`; entropy `0.8219`; top gap `0.0102`. Asking the learner to reason
   from the pair and cueing the next factorization step are both plausible.

8. **Correct/recovering learner, generic strongly preferred** —
   `8a72eaca-f99f-4226-92d4-f73cc0492a2e:1:action:5`. The learner correctly
   expands and rearranges a quadratic and states the next plan. Vector:
   `(0.934026, 0.008435, 0.020008, 0.037532)`; entropy `0.3055`; non-top-1
   probability `0.0660`. Proportional sampling strongly preserves the coherent
   generic preference.

9. **Telling top-1** —
   `a6d964ef-ca38-47e1-a1f5-6009be1aa0b5:1:action:6`. The learner continues
   to confuse the meaning of a word-problem quantity. Vector:
   `(0.001980, 0.169375, 0.009446, 0.819198)`; entropy `0.5205`; non-top-1
   probability `0.1808`. The recalibrated selector's high telling probability
   is semantically plausible after the unresolved conceptual exchange.

## Result

The inspected vectors show the desired qualitative pattern: concentrated
probabilities in clear states and meaningful alternatives in ambiguous states.
No reviewed case supplies evidence for a handwritten rule or action filter.
This review supports using the unmodified frozen MD7 vector as a proposal
policy, subject to exact propensity logging and later empirical safety review.
