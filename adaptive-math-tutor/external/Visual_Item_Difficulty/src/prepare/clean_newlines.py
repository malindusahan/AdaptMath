"""
Use GPT-5.5 to judge which newlines in question/choices are layout artifacts
vs. semantically meaningful, and remove the layout ones.

Only processes entries that contain newlines. Updates question_parse.json in-place.

Run from project root:
  python scripts/clean_newlines.py
"""

import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

MODEL = "gpt-5.5"
BASE_URL = "https://us.api.openai.com/v1"
CONCURRENCY = 15
SAVE_EVERY = 50
INPUT_FILE = Path("data/question_parsed.json")

SYSTEM_PROMPT = """\
You are cleaning up OCR artifacts in math question text. Newlines in the text \
may be either (A) layout artifacts — the text wrapped to a new line inside an \
answer bubble or question box due to limited width, with no semantic meaning — \
or (B) genuine semantic breaks — separating distinct premises, conditions, or \
equations that should stay on separate lines.

You will receive a JSON object with "question" and "choices" fields. \
Return a cleaned JSON object with the same structure where:
- Layout-artifact newlines are replaced with a single space
- Genuine semantic newlines are kept as \\n
- Leading/trailing whitespace around each newline is collapsed

Rules of thumb:
- Newlines inside a short phrase like "Only\\nTom" or "Both Tom\\nand Katie" → space
- Newlines between two complete sentences or mathematical expressions → keep
- Newlines that split a word or break mid-phrase → space

Return ONLY the cleaned JSON object, no explanation.\
"""


def needs_cleaning(entry: dict) -> bool:
    if "\n" in entry.get("question", ""):
        return True
    return any("\n" in v for v in entry.get("choices", {}).values())


async def clean_one(
    client: AsyncOpenAI,
    entry: dict,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        payload = {
            "question": entry.get("question", ""),
            "choices": entry.get("choices", {}),
        }
        for attempt in range(3):
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                max_completion_tokens=1000,
                response_format={"type": "json_object"},
            )
            content = r.choices[0].message.content
            if content and content.strip():
                break
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
        else:
            return entry  # failed — return original unchanged

        try:
            cleaned = json.loads(content)
        except json.JSONDecodeError:
            return entry  # malformed JSON — return original

        result = dict(entry)
        if "question" in cleaned:
            result["question"] = cleaned["question"]
        if "choices" in cleaned:
            result["choices"] = cleaned["choices"]
        return result


async def main() -> None:
    with open(INPUT_FILE) as f:
        data = json.load(f)

    to_clean = [(i, e) for i, e in enumerate(data) if needs_cleaning(e)]
    print(f"Total entries: {len(data)}  Need cleaning: {len(to_clean)}")

    client = AsyncOpenAI(base_url=BASE_URL)
    sem = asyncio.Semaphore(CONCURRENCY)

    coros = [clean_one(client, entry, sem) for _, entry in to_clean]
    errors = 0
    for batch_i, (coro, (orig_i, _)) in enumerate(
        zip(asyncio.as_completed(coros), to_clean), 1
    ):
        result = await coro
        data[orig_i] = result
        if result is to_clean[batch_i - 1][1]:
            errors += 1

        if batch_i % SAVE_EVERY == 0 or batch_i == len(to_clean):
            with open(INPUT_FILE, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  [{batch_i}/{len(to_clean)}] saved")

    print(f"\nDone. {len(to_clean) - errors} cleaned, {errors} kept original.")


if __name__ == "__main__":
    asyncio.run(main())
