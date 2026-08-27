import json
from pathlib import Path


LABELS = [
    "generic",
    "probing",
    "focus",
    "telling",
]


DEFAULT_PREDICTIONS_PATH = Path(
    "artifacts/frozen_selector/"
    "recovered_context_routed_hybrid_validation_predictions.jsonl"
)


class FrozenSelectorReplay:
    """
    Offline replay interface for the recovered frozen selector.

    IMPORTANT:
    This class does NOT run MD3 or MD4 inference.

    It replays the exact frozen-selector outputs that were
    recovered from saved validation predictions.

    Intended use:
        - integration testing
        - LinTS overlay testing
        - offline experiments on validation examples

    Not intended use:
        - inference on unseen conversations
        - deployment
    """

    def __init__(
        self,
        predictions_path=DEFAULT_PREDICTIONS_PATH,
    ):
        self.predictions_path = Path(
            predictions_path
        )

        if not self.predictions_path.exists():
            raise FileNotFoundError(
                f"Recovered frozen-selector predictions "
                f"not found: {self.predictions_path}"
            )

        self.records = {}

        self._load()

    def _load(self):

        with open(
            self.predictions_path,
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                record = json.loads(
                    line
                )

                example_id = record[
                    "example_id"
                ]

                if example_id in self.records:
                    raise ValueError(
                        f"Duplicate example_id "
                        f"{example_id} "
                        f"at line {line_number}"
                    )

                self._validate_record(
                    record,
                    line_number,
                )

                self.records[
                    example_id
                ] = record

    def _validate_record(
        self,
        record,
        line_number,
    ):

        required = {
            "example_id",
            "qid",
            "route",
            "source_model",
            "true_move",
            "predicted_move",
            "confidence",
            "probabilities",
        }

        missing = (
            required
            -
            set(
                record.keys()
            )
        )

        if missing:
            raise ValueError(
                f"Line {line_number}: "
                f"missing fields "
                f"{sorted(missing)}"
            )

        probabilities = record[
            "probabilities"
        ]

        if set(
            probabilities.keys()
        ) != set(
            LABELS
        ):
            raise ValueError(
                f"Line {line_number}: "
                f"unexpected probability labels"
            )

        probability_sum = sum(
            float(
                probabilities[label]
            )
            for label in LABELS
        )

        if abs(
            probability_sum
            -
            1.0
        ) > 1e-5:
            raise ValueError(
                f"Line {line_number}: "
                f"probabilities sum to "
                f"{probability_sum}"
            )

        predicted_from_probs = max(
            LABELS,
            key=lambda label:
                probabilities[label],
        )

        if (
            predicted_from_probs
            !=
            record[
                "predicted_move"
            ]
        ):
            raise ValueError(
                f"Line {line_number}: "
                f"predicted_move does not "
                f"match probability argmax"
            )

    def __len__(self):
        return len(
            self.records
        )

    def has_example(
        self,
        example_id,
    ):
        return (
            example_id
            in
            self.records
        )

    def predict(
        self,
        example_id,
    ):
        """
        Return the exact recovered frozen-selector output
        for one validation example.
        """

        if example_id not in self.records:
            raise KeyError(
                f"Unknown validation example_id: "
                f"{example_id}"
            )

        record = self.records[
            example_id
        ]

        return {
            "example_id":
                record[
                    "example_id"
                ],

            "qid":
                record[
                    "qid"
                ],

            "route":
                record[
                    "route"
                ],

            "source_model":
                record[
                    "source_model"
                ],

            "predicted_move":
                record[
                    "predicted_move"
                ],

            "confidence":
                float(
                    record[
                        "confidence"
                    ]
                ),

            "probabilities": {
                label:
                    float(
                        record[
                            "probabilities"
                        ][
                            label
                        ]
                    )

                for label
                in LABELS
            },
        }

    def get_true_move(
        self,
        example_id,
    ):
        """
        Ground-truth label helper for offline evaluation only.

        This must NOT be treated as runtime input to the selector
        or LinTS policy.
        """

        if example_id not in self.records:
            raise KeyError(
                f"Unknown validation example_id: "
                f"{example_id}"
            )

        return self.records[
            example_id
        ][
            "true_move"
        ]

    def example_ids(
        self,
    ):
        return list(
            self.records.keys()
        )