import json
from pathlib import Path
from datetime import datetime, timezone


# ============================================================
# PATHS
# ============================================================

RESULT_DIR = Path(
    "results/selection"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_PATH = (
    RESULT_DIR
    / "frozen_supervised_selector.json"
)


CLUSTER_BOOTSTRAP_PATH = Path(
    "results/baselines/"
    "selector_cluster_bootstrap_comparison.json"
)


# ============================================================
# LOAD FINAL STATISTICAL ANALYSIS
# ============================================================

with CLUSTER_BOOTSTRAP_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    bootstrap = json.load(f)


# ============================================================
# EXTRACT VALIDATED VALUES
# ============================================================

hybrid = (
    bootstrap[
        "point_results"
    ][
        "context_hybrid"
    ]
)


hybrid_minus_md2 = (
    bootstrap[
        "paired_differences"
    ][
        "context_hybrid_minus_MD2"
    ]
)


hybrid_minus_md4 = (
    bootstrap[
        "paired_differences"
    ][
        "context_hybrid_minus_MD4"
    ]
)


# ============================================================
# SANITY CHECKS
# ============================================================

assert abs(
    hybrid["macro_f1"]
    - 0.486768
) < 1e-5


assert abs(
    hybrid["accuracy"]
    - 0.500541
) < 1e-5


# ============================================================
# FREEZE
# ============================================================

frozen = {

    "status":
        "FROZEN",

    "stage":
        "supervised_move_selector",

    "selection_metric":
        "macro_f1",

    "architecture":
        "context_routed_hybrid",

    "routing": {

        "tokenizer":
            "distilbert/distilroberta-base",

        "measurement":
            (
                "original input token length "
                "before truncation"
            ),

        "threshold":
            512,

        "short_condition":
            "<=512",

        "short_model":
            "MD4",

        "short_model_description":
            (
                "DistilRoBERTa with "
                "inverse-frequency weighted "
                "cross-entropy"
            ),

        "long_condition":
            ">512",

        "long_model":
            "MD3",

        "long_model_description":
            (
                "ModernBERT-base "
                "long-context classifier"
            ),
    },

    "validation": {

        "examples":
            1850,

        "qids":
            111,

        "accuracy":
            hybrid["accuracy"],

        "macro_f1":
            hybrid["macro_f1"],

        "per_class_f1": {

            "generic":
                0.599764,

            "probing":
                0.416446,

            "focus":
                0.552846,

            "telling":
                0.378016,
        },

        "short_examples":
            1507,

        "long_examples":
            343,
    },

    "comparison_to_md2": {

        "md2_macro_f1":
            0.466976,

        "point_difference":
            hybrid_minus_md2[
                "point_difference"
            ],

        "cluster_bootstrap_95_ci": [

            hybrid_minus_md2[
                "lower_95"
            ],

            hybrid_minus_md2[
                "upper_95"
            ],
        ],

        "fraction_positive":
            hybrid_minus_md2[
                "fraction_positive"
            ],

        "interpretation":
            (
                "Hybrid has higher validation "
                "Macro-F1, but the qid-clustered "
                "95% paired interval includes zero."
            ),
    },

    "comparison_to_md4": {

        "md4_macro_f1":
            0.472096,

        "point_difference":
            hybrid_minus_md4[
                "point_difference"
            ],

        "cluster_bootstrap_95_ci": [

            hybrid_minus_md4[
                "lower_95"
            ],

            hybrid_minus_md4[
                "upper_95"
            ],

        ],

        "fraction_positive":
            hybrid_minus_md4[
                "fraction_positive"
            ],

        "interpretation":
            (
                "Hybrid improvement over MD4 "
                "remains positive under "
                "qid-clustered paired bootstrap."
            ),
    },

    "test_policy": {

        "status":
            "UNTOUCHED",

        "rule":
            (
                "Do not inspect or evaluate on "
                "the frozen MathDial test split "
                "until final system evaluation."
            ),
    },

    "future_supervised_tuning":
        "STOPPED",

    "next_stage":
        "self_improvement",

    "notes": [

        (
            "Primary metric was fixed as "
            "Macro-F1 before model selection."
        ),

        (
            "Routing threshold 512 follows "
            "DistilRoBERTa's context limitation "
            "rather than validation threshold search."
        ),

        (
            "Validation-tuned probability ensembles "
            "were not selected despite similar "
            "performance because the context router "
            "is simpler and more interpretable."
        ),
    ],
}


# ============================================================
# SAVE
# ============================================================

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        frozen,
        f,
        indent=2,
    )


print("=" * 70)

print(
    "SUPERVISED SELECTOR FROZEN"
)

print("=" * 70)


print(
    "\nArchitecture:",
    frozen["architecture"],
)

print(
    "Validation Macro-F1:",
    frozen["validation"]["macro_f1"],
)

print(
    "Validation Accuracy:",
    frozen["validation"]["accuracy"],
)


print(
    "\nRouting:"
)

print(
    "  <=512 tokens -> MD4 weighted DistilRoBERTa"
)

print(
    "  >512 tokens  -> MD3 ModernBERT"
)


print(
    "\nTest status:",
    frozen["test_policy"]["status"],
)


print(
    "\nNext stage:",
    frozen["next_stage"],
)


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)
