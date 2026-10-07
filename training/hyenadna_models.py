"""Shared HyenaDNA model identifiers and reproducible Hugging Face revisions."""

DEFAULT_HYENADNA_MODEL = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
AUTO_MODEL_REVISION = "auto"

# Current Hugging Face revisions verified on 2026-09-27. Each model size lives in
# a separate repository, so its commit must be pinned independently.
HYENADNA_MODEL_REVISIONS = {
    "LongSafari/hyenadna-tiny-1k-seqlen-hf": (
        "e8c1effa8673814e257e627d2e1eda9ea5a373f6"
    ),
    "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf": (
        "6036d4e144922b44470294f510cbfd2539ba3b7e"
    ),
    "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf": (
        "d79fa37e2cd62dd338103c630f95be8f90812d46"
    ),
    "LongSafari/hyenadna-small-32k-seqlen-hf": (
        "8fe770c78eb13fe33bf81501612faeddf4d6f331"
    ),
    "LongSafari/hyenadna-medium-160k-seqlen-hf": (
        "7ebf71773d22c0ede2cc55cb2be15ee8c289e1ce"
    ),
    "LongSafari/hyenadna-medium-450k-seqlen-hf": (
        "42dedd4d374eac0fb8168549e546a3472fbd27ae"
    ),
    "LongSafari/hyenadna-large-1m-seqlen-hf": (
        "0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446"
    ),
}


def resolve_hyenadna_revision(
    model_name: str, requested_revision: str | None = None
) -> str | None:
    """Use an explicit revision, otherwise return the known model's pinned commit."""
    if requested_revision not in (None, AUTO_MODEL_REVISION):
        return requested_revision
    return HYENADNA_MODEL_REVISIONS.get(model_name)
