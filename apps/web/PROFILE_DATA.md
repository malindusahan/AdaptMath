# Student profile data provenance

The active `/profile` page makes one authenticated request to the Tutor API's
`GET /profile` endpoint. The request contains the existing Memory-issued Bearer
token and never contains a student ID.

| Profile field | Existing authoritative source |
| --- | --- |
| Username and age | Memory `/auth/me`, validated by the Tutor authentication dependency |
| Recent completed session | Student Model `sessions` rows, excluding internal dialogue sub-sessions |
| Session skill and answer counts | BKT `attempts` belonging to that completed session |
| Last-seven-days summary | Bounded aggregation of completed Student Model sessions and their BKT attempts |
| Skill mastery and status | BKT `mastery_probability` and stored `mastery_label` |
| Skill trend | Comparison of BKT `mastery_probability` with `previous_mastery_probability`; no trend is shown without previous evidence |
| Focus Next | First item from the existing prerequisite-aware `generate_learning_path(...).recommended_order` |

The UI maps the existing BKT labels only for presentation: `strong` to
“Strong”, `partial` to “Developing”, and `weak` to “Needs practice”. It does
not recalculate or replace the BKT thresholds.

Student Memory's PostgreSQL completed-attempt receipts were inspected and used
to validate the session evidence. They are not used as a substitute for BKT
mastery. No LinTS, MD6, MRB1, C3, reward, prompt, or diagnostic values are
returned to the learner.
