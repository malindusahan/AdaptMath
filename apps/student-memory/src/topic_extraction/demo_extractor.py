"""Demonstration of hybrid topic extractor across all 4 operational tiers."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topic_extraction.topic_extractor import get_topic_extractor

extractor = get_topic_extractor()

samples = [
    ("Keyword Match", "Can you explain PEMDAS and how it works?"),
    ("MiniLM Fine-Tuned Semantic Match", "How do I work out the steepness of a line from two points on a graph?"),
    ("TF-IDF Fallback Corroboration", "What is the difference between congruent figures and similar figures?"),
    ("Safe Abstention", "Can you help me?"),
]

for label, query in samples:
    print(f"\n{'='*70}")
    print(f"Tier / Test: {label}")
    print(f"Student Input: '{query}'")
    print("=" * 70)
    result = extractor.extract(query)
    print(json.dumps(result.to_dict(), indent=2))
