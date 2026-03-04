#!/bin/bash

# ============================================================
# Irregular 6-month commit history (Git Bash version)
# ============================================================

if [ ! -d .git ]; then
  echo "Error: Not a git repository."
  exit 1
fi

# === FORCE specific author (change these two lines) ===
AUTHOR_NAME="malindusahan"
AUTHOR_EMAIL="balasooriyamalindu@gmail.com"
# =====================================================

commit() {
  local date="$1"
  local msg="$2"
  local hour=$((9 + RANDOM % 9))
  local minute=$((RANDOM % 60))
  local second=$((RANDOM % 60))
  local datetime="${date} $(printf "%02d:%02d:%02d" $hour $minute $second)"

  echo "$msg" >> .commit_log
  git add .commit_log

  export GIT_AUTHOR_DATE="$datetime"
  export GIT_COMMITTER_DATE="$datetime"
  export GIT_AUTHOR_NAME="$AUTHOR_NAME"
  export GIT_AUTHOR_EMAIL="$AUTHOR_EMAIL"
  export GIT_COMMITTER_NAME="$AUTHOR_NAME"
  export GIT_COMMITTER_EMAIL="$AUTHOR_EMAIL"

  git commit -m "$msg" --quiet
  echo "✓ $datetime  →  $msg"
}

echo "Generating irregular history..."
echo

# MARCH 2026
commit "2026-03-04" "Initial student model and BKT parameters"
commit "2026-03-07" "Add structured evaluator for correctness"
commit "2026-03-11" "Basic BKT update step"
commit "2026-03-14" "Add observation confidence"
commit "2026-03-18" "Separate correctness from behavioural evidence"
commit "2026-03-21" "Skill-level mastery state"
commit "2026-03-25" "Support correct/partial/incorrect/unknown"
commit "2026-03-28" "One observation per student event"
commit "2026-03-30" "Add learn, guess, slip, forget params"

# APRIL 2026
commit "2026-04-02" "Add reasoning detector"
commit "2026-04-03" "Add uncertainty detector"
commit "2026-04-07" "Add clarification detector"
commit "2026-04-09" "Introduce Signal Resolver"
commit "2026-04-12" "Correct + uncertainty → reduced confidence"
commit "2026-04-15" "Incorrect + uncertainty → stronger negative"
commit "2026-04-16" "Unknown + uncertainty → weak negative proxy"
commit "2026-04-20" "Reasoning alone cannot create positive mastery"
commit "2026-04-23" "Enforce single observation rule"
commit "2026-04-27" "Document dialogue vs assessment evidence strength"
commit "2026-04-29" "Resolver integration tests"

# MAY 2026
commit "2026-05-03" "Continuous BKT updates during tutoring"
commit "2026-05-06" "Teaching-progress evaluator"
commit "2026-05-08" "Define Attempt as one tutoring session"
commit "2026-05-11" "Snapshot mastery_before at attempt start"
commit "2026-05-13" "Compute mastery_after and delta"
commit "2026-05-17" "Support signed delta_mastery"
commit "2026-05-19" "3-question formal assessment in attempt"
commit "2026-05-22" "Assessment also goes through BKT"
commit "2026-05-26" "Pass delta as delayed reward"
commit "2026-05-28" "Move selector must not calculate mastery"
commit "2026-05-30" "Attempt-level mastery tests"

# JUNE 2026
commit "2026-06-03" "Fix inconsistent initial prior problem"
commit "2026-06-05" "Immutable effective_initial_prior"
commit "2026-06-08" "Support prior provenance types"
commit "2026-06-11" "Store prior in Knowledge Graph"
commit "2026-06-14" "Guarantee mastery continuity across attempts"
commit "2026-06-17" "Knowledge Graph owns mastery"
commit "2026-06-19" "Persist observations and mastery states"
commit "2026-06-23" "Block unauthorized mastery writes"
commit "2026-06-26" "Full-history BKT recomputation"
commit "2026-06-28" "Verify recomputation matches live state"
commit "2026-06-30" "Unauthorized skill isolation tests"

# JULY 2026
commit "2026-07-02" "Learning Path Generator skeleton"
commit "2026-07-05" "Load curriculum prerequisite graph"
commit "2026-07-08" "Combine mastery with prerequisites"
commit "2026-07-10" "Define skill states (weak/partial/strong etc)"
commit "2026-07-13" "Strong threshold set to 0.70"
commit "2026-07-16" "Handle multi-parent prerequisites"
commit "2026-07-18" "Produce recommended_order"
commit "2026-07-21" "Report blocked, unseen, regressions"
commit "2026-07-23" "Report learn_next and already_strong"
commit "2026-07-25" "No semantic similarity for prerequisites"
commit "2026-07-28" "Path generator + student model integration"
commit "2026-07-30" "Path generation tests"

# AUGUST 2026
commit "2026-08-01" "Wire student model into full lifecycle"
commit "2026-08-03" "Snapshot mastery_before on attempt start"
commit "2026-08-05" "MD6 + LinTS select pedagogical moves"
commit "2026-08-07" "BKT updates during student turns"
commit "2026-08-09" "Final BKT update after assessment"
commit "2026-08-11" "Single attempt-level delta as reward"
commit "2026-08-13" "No per-dialogue rewards"
commit "2026-08-15" "Smoke: shared attempt identity"
commit "2026-08-17" "Smoke: attempt-start snapshot"
commit "2026-08-18" "Smoke: no premature LinTS update"
commit "2026-08-19" "Smoke: 3-question assessment"
commit "2026-08-20" "Smoke: one attempt-level outcome"
commit "2026-08-21" "Smoke: delta arithmetic"
commit "2026-08-22" "Smoke: move-selector reward parity"
commit "2026-08-23" "Smoke: delayed LinTS update"
commit "2026-08-24" "Smoke: initial-prior persistence"
commit "2026-08-25" "Smoke: cross-attempt continuity"
commit "2026-08-26" "Smoke: independent BKT parity"
commit "2026-08-27" "Smoke: unauthorized-skill isolation"
commit "2026-08-28" "Final component documentation"
commit "2026-08-29" "Component summary complete"

echo
echo "Done."
echo "Check with: git log --pretty=fuller -5"