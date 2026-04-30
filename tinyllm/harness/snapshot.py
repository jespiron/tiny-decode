import hashlib
import json
from pathlib import Path

SNAPSHOT_PATH = Path("snapshots/greedy.json")


def _key(model: str, prompt: str, n_tokens: int) -> str:
    h = hashlib.sha256()
    h.update(f"{model}\x00{prompt}\x00{n_tokens}".encode())
    return h.hexdigest()[:16]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load() -> dict:
    if SNAPSHOT_PATH.exists():
        return json.loads(SNAPSHOT_PATH.read_text())
    return {}


def _save(snap: dict) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snap, indent=2))


def check_or_write(
    model: str, prompt: str, n_tokens: int, text: str, write: bool
) -> tuple[bool, str]:
    snap = _load()
    key = _key(model, prompt, n_tokens)
    actual_hash = _hash_text(text)

    if write:
        snap[key] = {
            "model": model,
            "prompt": prompt,
            "n_tokens": n_tokens,
            "output_sha256": actual_hash,
            "generated_text": text,
        }
        _save(snap)
        return True, f"wrote snapshot ({model}, n_tokens={n_tokens})"

    if key not in snap:
        return False, (
            f"no snapshot for ({model}, n_tokens={n_tokens}). "
            "Seed it with `--write-snapshot` first."
        )

    expected = snap[key]
    if expected["output_sha256"] == actual_hash:
        return True, f"snapshot match ({model}, n_tokens={n_tokens})"

    return False, (
        f"snapshot MISMATCH for ({model}, n_tokens={n_tokens})\n"
        f"  expected: {expected['output_sha256']}\n"
        f"  actual:   {actual_hash}\n"
        f"  expected text: {expected['generated_text'][:120]!r}...\n"
        f"  actual text:   {text[:120]!r}..."
    )
