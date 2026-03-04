# Tutor move-realization runtime smoke report

## Result

Twelve retained Tutor responses were generated through the actual
`TutorAgent.teach_turn -> finalizer -> TeachingResponseVerifier` path: three
representative learner states crossed with generic, probing, focus, and
telling. Manual structural review found **12/12 contract-conformant** responses.
This audit tests linguistic realization after a move is selected; it does not
judge whether the selected move was pedagogically optimal.

The move label was supplied internally to the generation path and was not
included in learner-visible response text.

## Retained responses and manual audit

| Problem / learner state | Selected move | Tutor response | Manual finding |
|---|---|---|---|
| `(-3)(-5)+2`; learner says two negatives give a negative | generic | Thanks for sharing your steps. | PASS - supportive/declarative; no diagnostic question |
| same | probing | What happens when you multiply a negative number by another negative number? | PASS - diagnostic question; method not supplied first |
| same | focus | You're right to start with the multiplication. Take another look at the rule for multiplying two negative numbers. | PASS - targeted directive, not probing |
| same | telling | When you multiply two negative numbers, the result is a positive number. So, (-3)(-5) would be positive 15. Can you take it from there? | PASS - rule and numerical step explicitly supplied before follow-up |
| rectangle area 48, width 6; learner proposes addition | generic | Thanks for sharing your idea. | PASS - supportive/declarative; no diagnostic question |
| same | probing | What is the formula for the area of a rectangle? | PASS - diagnostic question; method not supplied first |
| same | focus | Remember that the area of a rectangle is found by multiplying its length and width. You know the area is 48 and the width is 6. | PASS - targeted cue; does not ask learner to diagnose |
| same | telling | Adding the area and width won't give you the length. To find the length of a rectangle, you need to divide the area by the width. What do you get when you divide 48 by 6? | PASS - correction and method explicitly supplied |
| `2(x-3)=10`; learner asks for the first step | generic | It's completely normal to feel a bit stuck sometimes. You've got this! | PASS - supportive/declarative; no diagnostic question |
| same | probing | What does the '2' outside the parentheses indicate you should do with the expression inside? | PASS - reasoning question; no step supplied first |
| same | focus | Consider how you can remove the parentheses on the left side of the equation. | PASS - targeted directive, not merely a question |
| same | telling | To begin, you can divide both sides of the equation by 2. This will simplify the left side. What do you get when you do that? | PASS - valid method/first step explicitly supplied |

Telling responses may end with a learner check; they passed because the
required method or step was first stated explicitly rather than hidden behind
leading questions.

## External API accounting

- External API: Gemini Developer API through the locally configured Tutor
  client.
- Exact `generate_content` attempts: **34**.
- Exact logical completion calls: **34** = 17 Tutor-finalizer calls + 17
  verifier calls.
- Retained responses: 12.

The 34 attempts include six calls from one failed rectangle/telling generation
cycle (three finalizer/verifier pairs). Its correct formula was rejected by the
contextual-formula verifier until the trusted geometry-formula context was
supplied. Those calls are included rather than hidden. The retained retry then
passed. No large dataset was produced.

The exact machine-readable prompts, histories, responses, and call accounting
are retained in `runtime_smoke_raw.json`; `run_runtime_smoke.py` is the small,
resumable harness used for this audit.

