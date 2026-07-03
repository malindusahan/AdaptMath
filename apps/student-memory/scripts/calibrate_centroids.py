"""Centroid Calibration & Operational Enhancement for 111 Canonical Skills."""

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "artifacts" / "topic_extractor" / "minilm_finetuned_v2"
CENTROIDS_NPZ = ROOT / "artifacts" / "topic_extractor" / "minilm_finetuned_v2_skill_centroids.npz"
ONTOLOGY_JSON = ROOT / "data" / "processed" / "canonical_skill_ontology.json"

# Operational disambiguation & compound problem exemplars for skills
OPERATIONAL_EXEMPLARS: dict[str, list[str]] = {
    "math :: proportional reasoning & percents :: percent of": [
        "A theatre has 480 seats. 35% of the seats were sold online. How many seats is 35% of 480?",
        "Find 35% of 480.",
        "Calculate thirty-five percent of four hundred eighty.",
        "What is 20% of 150?",
        "Finding percent of a number: multiply the percentage by the total quantity.",
        "A school has 600 students and 45% ride the bus. How many students is that?",
        "Compute the part when given percentage and whole: part = percent * whole.",
        "Percent of a total amount or quantity."
    ],
    "math :: proportional reasoning & percents :: finding percents": [
        "168 out of 480 seats were sold. What percentage of the seats were sold?",
        "What percent of 50 is 12?",
        "Find what percentage 30 is of 120.",
        "Calculate the percent given part and whole: percent = (part / whole) * 100.",
        "What percentage did the student score if they got 45 out of 60 correct?"
    ],
    "math :: algebra & functions :: equation solving two or fewer steps": [
        "The length of a rectangular field is 7 metres more than twice its width. If the perimeter is 86 metres, set up and solve the equation 2(2w + 7) + 2w = 86 for length and width.",
        "Solve two-step linear equation: 2(2x + 7) + 2x = 86 to find x.",
        "The perimeter of a rectangle is 86, length is 2w + 7. Solve 6w + 14 = 86 for w.",
        "Setting up and solving an equation with one variable from geometric or word problem descriptions.",
        "Equation solving two or fewer steps: solve for the unknown variable x in ax + b = c.",
        "Solve 3x + 5 = 20.",
        "Solve 4w - 8 = 32."
    ],
    "math :: geometry & measurement :: perimeter of a polygon": [
        "Find the perimeter of a polygon with side lengths 5 cm, 7 cm, and 9 cm.",
        "Calculate the perimeter of a triangle by adding all three outer boundary side lengths.",
        "What is the perimeter of a regular pentagon with sides of 6 cm? 5 * 6 = 30 cm.",
        "Perimeter formula P = sum of all outer boundary sides of a polygon.",
        "Compute boundary length of a polygon shape."
    ],
    "math :: proportional reasoning & percents :: proportion": [
        "A water tank is 3/5 full. After 120 litres of water are added, it becomes 9/10 full. What is the total capacity of the tank?",
        "A tank fraction changes from 3/5 to 9/10 when 120 L is added. 9/10 - 3/5 = 3/10 capacity = 120 L. Find total capacity using proportion.",
        "Proportional reasoning with fractional parts: fraction difference equals quantity, find total whole.",
        "Solve proportions: a/b = c/d to find unknown whole capacity.",
        "If 3/10 of a container is 120 liters, what is the full container capacity?",
        "Direct proportion and proportional scaling from fraction difference to whole."
    ],
    "math :: number sense & operations :: addition and subtraction fractions": [
        "Calculate 3/5 + 2/7 with common denominator.",
        "Subtract fractions: 9/10 - 3/5 = 9/10 - 6/10 = 3/10.",
        "Find common denominator and add or subtract fractions: 1/4 + 2/3.",
        "Adding and subtracting positive and negative fractions."
    ],
    "math :: algebra & functions :: recognize quadratic pattern": [
        "The first four numbers in a sequence are 4, 9, 16, 25. Determine the next two numbers and describe the mathematical quadratic rule n^2.",
        "Identify quadratic sequence: 4, 9, 16, 25, 36, 49 following n squared (n+1)^2.",
        "Recognize quadratic pattern with second differences constant or square numbers.",
        "Quadratic sequence pattern: 1, 4, 9, 16, 25, 36; rule is square of position.",
        "Find the nth term of quadratic sequence a*n^2 + b*n + c.",
        "Pattern of square numbers: 2^2, 3^2, 4^2, 5^2, 6^2, 7^2."
    ],
    "math :: algebra & functions :: pattern finding": [
        "Find the next number in the arithmetic sequence 3, 7, 11, 15 with constant difference +4.",
        "Describe the visual pattern rule for repeating geometric shapes.",
        "Pattern finding: linear progression where each term increases by a fixed addition.",
        "What comes next in the sequence 5, 10, 15, 20?"
    ],
    "math :: algebra & functions :: solving systems of linear equations": [
        "A taxi company charges a fixed starting fee plus the same amount for every kilometre travelled. A 6 km journey costs Rs. 1,100, while a 10 km journey costs Rs. 1,700. Solve the system of linear equations 6r + f = 1100 and 10r + f = 1700 to find rate r and fixed fee f.",
        "Solve a system of two linear equations with two unknowns: elimination or substitution.",
        "Simultaneous equations: 6r + f = 1100 and 10r + f = 1700. Subtract equations to get 4r = 600, r = 150, f = 200.",
        "Two variables word problem with fixed fee and variable rate: solve simultaneous linear system.",
        "System of equations: 2x + 3y = 12 and x - y = 1. Find x and y.",
        "Solving systems of linear equations in two variables."
    ],
    "math :: proportional reasoning & percents :: rate": [
        "A car travels 150 miles in 3 hours. What is its speed rate in miles per hour?",
        "Unit rate calculation: distance divided by time, cost per unit item.",
        "A typist types 240 words in 4 minutes. What is the typing rate per minute?",
        "Calculate rate of speed or unit cost: 150 / 3 = 50 mph."
    ]
}


def calibrate() -> None:
    print(f"Loading Sentence Transformer model from {MODEL_DIR}...")
    model = SentenceTransformer(str(MODEL_DIR), device="cpu")

    with open(ONTOLOGY_JSON, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    ft_data = np.load(CENTROIDS_NPZ, allow_pickle=True)
    orig_centroids = ft_data["centroids"]
    canonical_names = list(ft_data["canonical_names"])
    metadata_json = str(ft_data["metadata_json"])

    name_to_idx = {name: i for i, name in enumerate(canonical_names)}
    new_centroids = np.copy(orig_centroids)

    for skill in ontology:
        c_name = skill["canonical_name"].strip()
        idx = name_to_idx.get(c_name)
        if idx is None:
            continue

        # Gather rich descriptive exemplars for this skill
        texts = [
            c_name,
            skill["display_name"],
            f"Topic: {skill['display_name']}. Category: {skill.get('category', '')}.",
            skill.get("description", ""),
        ]
        texts.extend(skill.get("aliases", []))

        # Add operational exemplars if provided
        if c_name in OPERATIONAL_EXEMPLARS:
            texts.extend(OPERATIONAL_EXEMPLARS[c_name])

        # Encode and compute unit normalized mean vector
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        mean_vec = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-12:
            unit_vec = mean_vec / norm
        else:
            unit_vec = orig_centroids[idx]

        new_centroids[idx] = unit_vec

    # Save calibrated centroids
    np.savez_compressed(
        CENTROIDS_NPZ,
        centroids=new_centroids,
        canonical_names=canonical_names,
        metadata_json=metadata_json,
    )
    print(f"Successfully calibrated and saved {len(canonical_names)} centroids to {CENTROIDS_NPZ}!")


if __name__ == "__main__":
    calibrate()
