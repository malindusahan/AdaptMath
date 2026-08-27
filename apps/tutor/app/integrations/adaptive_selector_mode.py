"""Fail-closed selector modes and selector/policy lineage pairing."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum
from typing import Final

from app.integrations.manual_controlled_live import (
    MANUAL_CONTROLLED_LIVE_ENV,
    MANUAL_CONTROLLED_LIVE_VALUE,
)


SELECTOR_MODE_ENV: Final[str] = "ADAPTIVE_SELECTOR_MODE"


class AdaptiveSelectorMode(str, Enum):
    """The only supported runtime selector configurations."""

    NORMAL_MD7 = "ordinary-md7-r2-tell-c1-v1"
    MD7_R1_ROLLBACK = "md7r1-rollback-v1"
    MD6_ROLLBACK = "md6-rollback-v1"
    MANUAL_CONTROLLED_LIVE = "manual-controlled-live-md7r1-v1"

    @property
    def uses_md7_policy_lineage(self) -> bool:
        return self is not AdaptiveSelectorMode.MD6_ROLLBACK

    @property
    def uses_turn_lints(self) -> bool:
        return self is AdaptiveSelectorMode.NORMAL_MD7

    @property
    def learning_enabled(self) -> bool:
        return self is not AdaptiveSelectorMode.MANUAL_CONTROLLED_LIVE


def resolve_selector_mode(
    environment: Mapping[str, str] | None = None,
) -> AdaptiveSelectorMode:
    """Resolve one exact mode while retaining the versioned manual legacy flag."""

    source = os.environ if environment is None else environment
    raw_mode = source.get(SELECTOR_MODE_ENV)
    legacy_manual = source.get(MANUAL_CONTROLLED_LIVE_ENV)

    if legacy_manual not in {None, "", MANUAL_CONTROLLED_LIVE_VALUE}:
        raise ValueError(
            f"{MANUAL_CONTROLLED_LIVE_ENV} must be absent or exactly "
            f"{MANUAL_CONTROLLED_LIVE_VALUE!r}; got {legacy_manual!r}."
        )

    if raw_mode in {None, ""}:
        if legacy_manual == MANUAL_CONTROLLED_LIVE_VALUE:
            return AdaptiveSelectorMode.MANUAL_CONTROLLED_LIVE
        return AdaptiveSelectorMode.NORMAL_MD7

    try:
        mode = AdaptiveSelectorMode(raw_mode)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in AdaptiveSelectorMode)
        raise ValueError(
            f"{SELECTOR_MODE_ENV} must be one of: {allowed}; got {raw_mode!r}."
        ) from exc

    if (
        legacy_manual == MANUAL_CONTROLLED_LIVE_VALUE
        and mode is not AdaptiveSelectorMode.MANUAL_CONTROLLED_LIVE
    ):
        raise ValueError(
            f"{SELECTOR_MODE_ENV}={mode.value!r} conflicts with "
            f"{MANUAL_CONTROLLED_LIVE_ENV}={legacy_manual!r}."
        )
    return mode


__all__ = (
    "AdaptiveSelectorMode",
    "SELECTOR_MODE_ENV",
    "resolve_selector_mode",
)
