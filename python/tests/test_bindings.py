"""cdch_ferris binding behaviour: batch == single (bitwise), layouts, validation."""
import numpy as np
import pytest

import cdch_ferris as cf

RNG = np.random.default_rng(20261116)
STACK = RNG.integers(0, 256, (9, 128, 128, 3), dtype=np.uint8)


def same(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


@pytest.mark.parametrize("threads", [1, 2, None])
@pytest.mark.parametrize("kind", ["A", "B", "C"])
def test_batch_features_equal_single(kind, threads):
    batch = cf.batch_features(STACK, kind, threads=threads)
    single = np.stack([cf.features(im, kind) for im in STACK])
    assert same(batch, single)


@pytest.mark.parametrize("threads", [1, 3, None])
def test_batch_cdch_equal_single_and_reference(threads):
    batch = cf.batch_cdch(STACK[:3], threads=threads)
    for im, m in zip(STACK[:3], batch):
        assert same(m, cf.cdch(im))
        assert same(m, cf.cdch_reference(im))


def test_fused_equals_unfused_and_from_matrix():
    im = STACK[0]
    m = cf.cdch(im)
    for kind in "ABC":
        a = cf.features(im, kind)
        assert same(a, cf.features(im, kind, fused=False))
        assert same(a, cf.features_from_matrix(m, kind))
    assert cf.features(im, "D").shape == (24, 32)
    assert same(cf.features(im, "D").ravel(), cf.features(im, "C"))


def test_histograms_reconstruct_matrix():
    im = STACK[1]
    h = cf.histograms(im).astype(np.float64)
    cum = np.cumsum(h, axis=1)
    assert same(cum / cum[:, -1:], cf.cdch(im))


def test_sample_coordinates_match_histogram_totals():
    lengths, coords = cf.sample_coordinates(128, 128)
    assert lengths.shape == (360,) and coords.shape == (lengths.sum(), 2)
    assert np.array_equal(cf.histograms(STACK[2]).sum(axis=1), lengths)


def test_non_contiguous_inputs_are_copied_correctly():
    big = RNG.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    view = big[10:138, 50:178]  # strided crop
    assert not view.flags.c_contiguous
    assert same(cf.cdch(view), cf.cdch(np.ascontiguousarray(view)))
    fortran = np.asfortranarray(STACK[3])
    assert fortran.flags.f_contiguous and not fortran.flags.c_contiguous
    assert same(cf.features(fortran, "B"), cf.features(STACK[3], "B"))
    assert same(cf.batch_features(np.asfortranarray(STACK), "A"), cf.batch_features(STACK, "A"))
    m = cf.cdch(STACK[3])
    assert same(cf.features_from_matrix(np.asfortranarray(m), "B"), cf.features_from_matrix(m, "B"))


def test_center_and_angle_step():
    im = STACK[4]
    assert cf.cdch(im, center=(63.5, 63.5)).shape == (360, 256)
    assert cf.cdch(im, angle_step=7).shape == (52, 256)
    assert same(cf.cdch(im), cf.cdch(im, center=(64.0, 64.0)))


@pytest.mark.parametrize("bad, exc", [
    (STACK[0].astype(np.float64), TypeError),
    (STACK[0][..., :2], ValueError),
    (STACK[0][..., 0], ValueError),
    ([[1, 2, 3]], TypeError),
])
def test_invalid_images(bad, exc):
    with pytest.raises(exc):
        cf.features(bad, "A")


def test_invalid_parameters():
    with pytest.raises(ValueError, match="representation"):
        cf.features(STACK[0], "Q")
    with pytest.raises(ValueError, match="not finite"):
        cf.cdch(STACK[0], center=(np.inf, 1.0))
    with pytest.raises(ValueError, match="angle step"):
        cf.cdch(STACK[0], angle_step=400)
    with pytest.raises(ValueError, match="empty ray"):
        cf.features(STACK[0], "A", center=(-10.0, -10.0))
    with pytest.raises(ValueError, match="shape"):
        cf.features_from_matrix(np.zeros((360, 255)), "A")
    with pytest.raises(ValueError, match="requires 360 rays"):
        cf.features_from_matrix(cf.cdch(STACK[0], angle_step=2), "A")
