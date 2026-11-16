"""Rust-native canonical CDCH, validated against Muzyka's original ip-1.js.

Canonical source SHA-256: 110732b48bc6e8b2d5167eec30f7277245913bed98b9baaa3c65c0c22aab44d8

Functions
---------
cdch(rgb, center=None, angle_step=1)            -> (n_rays, 256) float64 canonical matrix
cdch_reference(rgb, center=None, angle_step=1)  -> same, literal reference port (slow)
histograms(rgb, center=None, angle_step=1)      -> (n_rays, 256) uint32 integer counts
sample_coordinates(width, height, center=None, angle_step=1) -> (ray_lengths, (N, 2) coords)
features(rgb, kind="A", center=None, fused=True) -> A (56,), B (98,), C (768,), D (24, 32)
features_multi(rgb, kinds, center=None)          -> {kind: array}
features_from_matrix(cdf, kind="A")              -> representation of an existing matrix
batch_features(images, kind="A", center=None, threads=None) -> (n, dim)
batch_cdch(images, center=None, angle_step=1, threads=None) -> (n, n_rays, 256)

``rgb`` is a decoded ``(height, width, 3)`` ``uint8`` array in RGB order.
``center`` defaults to the archived research protocol ``(width // 2, height // 2)``.
"""
from ._native import (  # noqa: F401
    A_DIM,
    B_DIM,
    C_DIM,
    CANONICAL_SHA256,
    __version__,
    batch_cdch,
    batch_features,
    build_info,
    cdch,
    cdch_reference,
    features,
    features_from_matrix,
    features_multi,
    histograms,
    sample_coordinates,
)

__all__ = [
    "A_DIM", "B_DIM", "C_DIM", "CANONICAL_SHA256", "__version__",
    "batch_cdch", "batch_features", "build_info", "cdch", "cdch_reference", "features",
    "features_from_matrix", "features_multi", "histograms", "sample_coordinates",
]
