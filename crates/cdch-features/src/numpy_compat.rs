//! Reductions that reproduce NumPy 2.x's floating-point evaluation order.
//!
//! The archived descriptors were computed with NumPy. Matching its summation order
//! makes most Rust feature values bit-identical to NumPy on the same platform. The
//! orders below were confirmed empirically against NumPy 2.3.5 (see
//! `RUST_PORT_VALIDATION.md`, "Feature arithmetic"):
//!
//! * contiguous 1-D `np.sum`/`mean`: pairwise summation (`DOUBLE_pairwise_sum`,
//!   8-way unrolled blocks of up to 128 elements);
//! * reductions over a non-innermost axis (`x.mean(axis=0)` on a `360 x k` array):
//!   plain sequential accumulation per output element.
//!
//! Matrix-vector products (`pmf @ bins`) are delegated by NumPy to BLAS, whose
//! order is library-specific (OpenBLAS on the archived Linux host, Accelerate on
//! macOS). Those cannot be matched in general; pairwise summation is used.

const PW_BLOCKSIZE: usize = 128;

/// NumPy's `DOUBLE_pairwise_sum` over `n` values produced by `f(i)`.
#[inline]
pub fn pairwise_sum_by(n: usize, f: &impl Fn(usize) -> f64) -> f64 {
    pairwise_rec(0, n, f)
}

fn pairwise_rec(start: usize, n: usize, f: &impl Fn(usize) -> f64) -> f64 {
    if n < 8 {
        let mut res = -0.0;
        for i in 0..n {
            res += f(start + i);
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0f64; 8];
        for (k, rk) in r.iter_mut().enumerate() {
            *rk = f(start + k);
        }
        let mut i = 8;
        while i < n - (n % 8) {
            for (k, rk) in r.iter_mut().enumerate() {
                *rk += f(start + i + k);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += f(start + i);
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_rec(start, n2, f) + pairwise_rec(start + n2, n - n2, f)
    }
}

/// [`pairwise_sum_by`] over a slice. Same order, but without closure recursion, so the
/// 8-accumulator block loop vectorizes.
pub fn pairwise_sum(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 8 {
        let mut res = -0.0;
        for &x in xs {
            res += x;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0f64; 8];
        r.copy_from_slice(&xs[..8]);
        let full = n - (n % 8);
        for chunk in xs[8..full].chunks_exact(8) {
            for k in 0..8 {
                r[k] += chunk[k];
            }
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &x in &xs[full..] {
            res += x;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&xs[..n2]) + pairwise_sum(&xs[n2..])
    }
}

/// `np.mean(x)` for a contiguous 1-D array.
#[inline]
pub fn mean(xs: &[f64]) -> f64 {
    pairwise_sum(xs) / xs.len() as f64
}

/// `np.std(x)` (population, ddof = 0) for a contiguous 1-D array, following
/// NumPy's `_var`: mean, deviations, squares, pairwise sum, divide, sqrt.
#[inline]
pub fn std(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let n = xs.len();
    (pairwise_sum_by(n, &|i| {
        let d = xs[i] - m;
        d * d
    }) / n as f64)
        .sqrt()
}

/// `np.linspace(0, 1, 256)`: `i * (1/255)` for `i < 255`, then exactly `1.0`.
pub fn linspace_0_1_256() -> [f64; 256] {
    let step = 1.0 / 255.0;
    let mut b = [0f64; 256];
    for (i, v) in b.iter_mut().enumerate() {
        *v = i as f64 * step;
    }
    b[255] = 1.0;
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_sums_are_sequential() {
        let x = [1e16, 1.0, -1e16, 1.0];
        assert_eq!(pairwise_sum(&x), ((1e16 + 1.0) + -1e16) + 1.0);
    }

    #[test]
    fn slice_and_closure_forms_agree_bitwise() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for n in [0usize, 1, 5, 7, 8, 9, 15, 16, 17, 127, 128, 129, 255, 256, 257, 360, 1000] {
            let xs: Vec<f64> = (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 11) as f64 / (1u64 << 53) as f64 - 0.3) * 10f64.powi((s % 9) as i32 - 4)
                })
                .collect();
            let a = pairwise_sum(&xs);
            let b = pairwise_sum_by(n, &|i| xs[i]);
            assert_eq!(a.to_bits(), b.to_bits(), "n = {n}");
        }
    }

    #[test]
    fn block_structure() {
        // For n = 16 the eight accumulators hold x[k] + x[k + 8].
        let x: Vec<f64> = (0..16).map(|i| (i as f64 + 0.1).powi(7)).collect();
        let r: Vec<f64> = (0..8).map(|k| x[k] + x[k + 8]).collect();
        let want = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        assert_eq!(pairwise_sum(&x).to_bits(), want.to_bits());
    }
}
