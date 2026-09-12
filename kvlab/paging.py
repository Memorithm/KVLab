"""Deterministic paged-KV accounting for K2 experiments.

This module is deliberately backend-neutral. It models page occupancy, physical
allocation, internal fragmentation, and prefix page sharing without claiming
that pagination or prefix reuse reduces the logical KV size of a request.
Those mechanisms may reduce allocator waste, duplicate physical residency, or
prefill work; experimental runners must measure any such savings separately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PageSpan:
    page_index: int
    token_start: int
    token_count: int
    capacity_tokens: int

    def __post_init__(self) -> None:
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if self.token_start < 0:
            raise ValueError("token_start must be non-negative")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive")
        if self.capacity_tokens <= 0:
            raise ValueError("capacity_tokens must be positive")
        if self.token_count > self.capacity_tokens:
            raise ValueError("token_count cannot exceed page capacity")


@dataclass(frozen=True, slots=True)
class PagedKvAccounting:
    token_count: int
    bytes_per_token: int
    page_size_tokens: int
    pages: tuple[PageSpan, ...]
    logical_kv_bytes: int
    allocated_bytes: int
    fragmentation_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("token_count", self.token_count),
            ("bytes_per_token", self.bytes_per_token),
            ("page_size_tokens", self.page_size_tokens),
            ("logical_kv_bytes", self.logical_kv_bytes),
            ("allocated_bytes", self.allocated_bytes),
            ("fragmentation_bytes", self.fragmentation_bytes),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.bytes_per_token == 0 and self.token_count > 0:
            raise ValueError("bytes_per_token must be positive for non-empty KV")
        if self.page_size_tokens == 0:
            raise ValueError("page_size_tokens must be positive")
        if self.allocated_bytes < self.logical_kv_bytes:
            raise ValueError("allocated bytes cannot be below logical KV bytes")
        if self.fragmentation_bytes != self.allocated_bytes - self.logical_kv_bytes:
            raise ValueError("fragmentation must equal allocated minus logical bytes")


@dataclass(frozen=True, slots=True)
class PrefixReuseAccounting:
    request_count: int
    prefix_tokens: int
    bytes_per_token: int
    page_size_tokens: int
    shared_full_pages: int
    logical_kv_bytes: int
    physical_bytes_without_sharing: int
    physical_bytes_with_sharing: int
    duplicate_bytes_avoided: int

    def __post_init__(self) -> None:
        for name, value in (
            ("request_count", self.request_count),
            ("prefix_tokens", self.prefix_tokens),
            ("bytes_per_token", self.bytes_per_token),
            ("page_size_tokens", self.page_size_tokens),
            ("shared_full_pages", self.shared_full_pages),
            ("logical_kv_bytes", self.logical_kv_bytes),
            ("physical_bytes_without_sharing", self.physical_bytes_without_sharing),
            ("physical_bytes_with_sharing", self.physical_bytes_with_sharing),
            ("duplicate_bytes_avoided", self.duplicate_bytes_avoided),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.request_count <= 0:
            raise ValueError("request_count must be positive")
        if self.page_size_tokens <= 0:
            raise ValueError("page_size_tokens must be positive")
        if self.prefix_tokens > 0 and self.bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive for non-empty KV")
        if self.physical_bytes_with_sharing > self.physical_bytes_without_sharing:
            raise ValueError("sharing cannot increase accounted physical bytes")
        if (
            self.duplicate_bytes_avoided
            != self.physical_bytes_without_sharing - self.physical_bytes_with_sharing
        ):
            raise ValueError("duplicate_bytes_avoided must match physical delta")


def account_paged_kv(
    *, token_count: int, bytes_per_token: int, page_size_tokens: int
) -> PagedKvAccounting:
    """Account page occupancy without changing logical KV size."""

    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if bytes_per_token < 0:
        raise ValueError("bytes_per_token must be non-negative")
    if page_size_tokens <= 0:
        raise ValueError("page_size_tokens must be positive")
    if token_count > 0 and bytes_per_token == 0:
        raise ValueError("bytes_per_token must be positive for non-empty KV")

    pages: list[PageSpan] = []
    remaining = token_count
    token_start = 0
    page_index = 0
    while remaining:
        occupied = min(remaining, page_size_tokens)
        pages.append(
            PageSpan(
                page_index=page_index,
                token_start=token_start,
                token_count=occupied,
                capacity_tokens=page_size_tokens,
            )
        )
        remaining -= occupied
        token_start += occupied
        page_index += 1

    logical = token_count * bytes_per_token
    allocated = len(pages) * page_size_tokens * bytes_per_token
    return PagedKvAccounting(
        token_count=token_count,
        bytes_per_token=bytes_per_token,
        page_size_tokens=page_size_tokens,
        pages=tuple(pages),
        logical_kv_bytes=logical,
        allocated_bytes=allocated,
        fragmentation_bytes=allocated - logical,
    )


def account_prefix_page_reuse(
    *, request_count: int, prefix_tokens: int, bytes_per_token: int, page_size_tokens: int
) -> PrefixReuseAccounting:
    """Account sharing of complete prefix pages across otherwise independent requests.

    Partial final prefix pages are deliberately not shared in this conservative
    model. Logical KV bytes count every request's semantic prefix and therefore
    do not decrease when physical pages are shared.
    """

    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if prefix_tokens < 0:
        raise ValueError("prefix_tokens must be non-negative")
    if bytes_per_token < 0:
        raise ValueError("bytes_per_token must be non-negative")
    if page_size_tokens <= 0:
        raise ValueError("page_size_tokens must be positive")
    if prefix_tokens > 0 and bytes_per_token == 0:
        raise ValueError("bytes_per_token must be positive for non-empty KV")

    one = account_paged_kv(
        token_count=prefix_tokens,
        bytes_per_token=bytes_per_token,
        page_size_tokens=page_size_tokens,
    )
    shared_full_pages = prefix_tokens // page_size_tokens
    shared_bytes = shared_full_pages * page_size_tokens * bytes_per_token
    without_sharing = request_count * one.allocated_bytes
    with_sharing = without_sharing - max(request_count - 1, 0) * shared_bytes

    return PrefixReuseAccounting(
        request_count=request_count,
        prefix_tokens=prefix_tokens,
        bytes_per_token=bytes_per_token,
        page_size_tokens=page_size_tokens,
        shared_full_pages=shared_full_pages,
        logical_kv_bytes=request_count * one.logical_kv_bytes,
        physical_bytes_without_sharing=without_sharing,
        physical_bytes_with_sharing=with_sharing,
        duplicate_bytes_avoided=without_sharing - with_sharing,
    )
