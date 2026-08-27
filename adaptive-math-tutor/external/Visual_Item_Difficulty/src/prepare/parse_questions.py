"""
Parse math question images into structured JSON using a vision LLM (two-pass).

Pass 1 — extract: question text, choices, has_figure, figure_description
Pass 2 — verify:  re-examine the image + draft, fix any errors, return final JSON

Output: data/question_parsed.json  (the file shipped in this repository)

Run from the repository root; the Eedi images are not redistributed here, so
point VDIFF_IMAGES at your own copy (see data/README.md):

  export VDIFF_IMAGES=/path/to/eedi/images
  OPENAI_API_KEY=... python src/prepare/parse_questions.py
"""

import asyncio
import base64
import json
import os
from pathlib import Path

from openai import AsyncOpenAI

MODEL = "gpt-5.5"
BASE_URL = "https://us.api.openai.com/v1"
CONCURRENCY = 10
SAVE_EVERY = 50

IMAGE_DIR = Path(os.environ.get("VDIFF_IMAGES", "data/images"))
OUTPUT = Path("data/question_parsed.json")

IMAGE_CONTENT = lambda b64: {
    "type": "image_url",
    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
}

PARSE_SYSTEM = """\
You are a math question parser. Given an image of a multiple-choice math question \
from the Eedi platform, extract its content as a JSON object.

Return ONLY a JSON object with these fields:

- "question": the full question text. Preserve math symbols exactly \
(×, ÷, −, ², ½, fractions, etc.). If the question references a visual element \
("this shape", "the diagram", "the graph below"), keep that phrasing verbatim.
- "choices": an object with keys "A", "B", "C", "D". If a choice is text, write \
the text. If a choice is itself an image or diagram, write a brief description \
e.g. "[image: protractor aligned at 40°]". Never leave a choice blank.
- "has_figure": true if the image contains a diagram, geometric shape, graph, \
number line, table, flowchart, or any other visual element that is part of the \
mathematical problem — including when the answer choices are images themselves. \
The colored letter bubbles (A/B/C/D) and the Eedi logo/header do NOT count. \
Set to false only for purely text-based questions.
- "figure_description": included ONLY when has_figure is true. Describe all figures \
concisely but completely — type, labeled values, dimensions, arrows, shading, \
spatial relationships. If choices are images, describe each one. \
Omit this field entirely when has_figure is false.\
"""

VERIFY_SYSTEM = """\
You are a meticulous math question verifier. You will be given an image of a \
multiple-choice math question and a draft JSON parse of that image.

Your job:
1. Look at the image carefully and compare it against every field in the draft.
2. Fix any errors you find: missing or garbled math symbols, truncated text, \
   wrong has_figure value, incomplete or inaccurate figure_description.
3. If any answer choice is an image rather than text, its value must be a \
   brief description like "[image: ...]" — never blank or null.
4. Return the corrected JSON object — same schema as the draft, no extra fields, \
   no explanation outside the JSON.

Schema reminder:
- "question": full question text with exact math symbols
- "choices": {{"A": ..., "B": ..., "C": ..., "D": ...}} — image choices as "[image: ...]"
- "has_figure": bool — true for any diagram/shape/graph/table/flowchart or image-based \
  choices (NOT the A/B/C/D bubbles or the Eedi header)
- "figure_description": concise complete description of all figures including image choices; \
  omit entirely when has_figure is false\
"""


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def normalise(result: dict, qid: int) -> dict:
    result["question_id"] = qid
    if not result.get("has_figure"):
        result.pop("figure_description", None)
        result["has_figure"] = False
    return result


async def _call(client: AsyncOpenAI, **kwargs) -> str:
    """Call the API with up to 3 retries on empty response.
    On final failure, retry once without response_format to capture error text."""
    for attempt in range(3):
        r = await client.chat.completions.create(**kwargs)
        choice = r.choices[0]
        content = choice.message.content
        if content and content.strip():
            return content
        if attempt < 2:
            await asyncio.sleep(2 ** attempt)

    # Diagnostic retry: drop json_object constraint and token limit
    diag_kwargs = {k: v for k, v in kwargs.items() if k != "response_format" and k != "max_completion_tokens"}
    r = await client.chat.completions.create(**diag_kwargs)
    choice = r.choices[0]
    content = choice.message.content or ""
    # If diagnostic pass returned valid JSON, use it directly
    try:
        if content.strip():
            json.loads(content)
            return content
    except json.JSONDecodeError:
        pass
    diag = (
        f"finish_reason={choice.finish_reason} | "
        f"refusal={choice.message.refusal!r} | "
        f"content={content!r:.200}"
    )
    raise ValueError(f"Empty response after 3 attempts. Diagnostic: {diag}")


async def parse_one(
    client: AsyncOpenAI,
    qid: int,
    image_path: Path,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        b64 = encode_image(image_path)
        img = IMAGE_CONTENT(b64)

        # ── Pass 1: extract ───────────────────────────────────────────────────
        draft_raw = await _call(
            client,
            model=MODEL,
            messages=[
                {"role": "system", "content": PARSE_SYSTEM},
                {"role": "user", "content": [img]},
            ],
            max_completion_tokens=1500,
            response_format={"type": "json_object"},
        )

        # ── Pass 2: verify ────────────────────────────────────────────────────
        final_raw = await _call(
            client,
            model=MODEL,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        img,
                        {
                            "type": "text",
                            "text": f"Draft parse:\n```json\n{draft_raw}\n```\n\n"
                                    "Return the corrected JSON.",
                        },
                    ],
                },
            ],
            max_completion_tokens=1500,
            response_format={"type": "json_object"},
        )
        result = json.loads(final_raw)
        return normalise(result, qid)


async def main() -> None:
    all_images = sorted(IMAGE_DIR.glob("*.jpg"), key=lambda p: int(p.stem))

    existing: list[dict] = []
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            existing = json.load(f)
    done_ids = {r["question_id"] for r in existing}
    print(f"Total images: {len(all_images)}  Already parsed: {len(done_ids)}  Remaining: {len(all_images) - len(done_ids)}")

    pending = [
        (int(p.stem), p)
        for p in all_images
        if int(p.stem) not in done_ids
    ]

    if not pending:
        print("Nothing to do.")
        return

    client = AsyncOpenAI(base_url=BASE_URL)
    sem = asyncio.Semaphore(CONCURRENCY)

    results = list(existing)
    errors: list[tuple[int, str]] = []

    coros = [parse_one(client, qid, path, sem) for qid, path in pending]
    for i, coro in enumerate(asyncio.as_completed(coros), 1):
        try:
            r = await coro
            results.append(r)
        except Exception as e:
            qid_failed = pending[i - 1][0]
            errors.append((qid_failed, str(e)))
            print(f"  ERROR qid={qid_failed}: {e}")

        if i % SAVE_EVERY == 0 or i == len(coros):
            with open(OUTPUT, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  [{i}/{len(coros)}] saved {len(results)} results")

    print(f"\nDone. {len(results)} parsed, {len(errors)} errors.")
    if errors:
        print("Failed question_ids:", [e[0] for e in errors])


if __name__ == "__main__":
    asyncio.run(main())
