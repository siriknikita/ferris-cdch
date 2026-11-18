# ADR 0005: Bitwise equivalence with ip-1.js is the contract

**Status:** accepted (2026-11-16, reaffirmed 2026-11-18)

## Context
CDCH is defined by Muzyka's JavaScript (`ip-1.js`). Results produced in the browser, in
the archived Node pipeline and in Rust must be comparable.

## Problem
"Close enough" is not well defined for a histogram of rounded coordinates: a one-ulp
difference in a cosine can move a sample to another pixel. Tolerance-based equivalence
would hide such divergence.

## Decision
Ferris must reproduce every value `ip-1.js` produces (matrices, coordinates, counts,
centroids) **bit for bit**. Concretely: embed V8's trigonometric bit patterns; implement
ECMAScript `Math.round`; evaluate luma left to right in binary64; keep integer counts and
a single division per bin; keep the radius limit equal to the image width; keep `NaN` for
empty rays. Inputs the JavaScript handles meaninglessly (`NaN` centres, empty images) are
rejected rather than reproduced.

## Alternatives considered
- *Use host libm:* differs from V8 at 37 of 360 angles on macOS, and this changes sampled
  pixels for some centres (8 adversarial fixtures).
- *A "cleaned-up" CDCH* (radius = max dimension, nearest-even rounding, integer luma): a
  different algorithm; it would be registered as a new method, never as CDCH.
- *Tolerance-based tests:* cannot detect the mutants that matter.

## Consequences
The implementation keeps some deliberately odd behaviour (the width-limited radius for
tall images). Changes need the Node oracle to verify.

## Risks and trade-offs
Equivalence is with V8 as of Node 22; a different V8 could in principle return different
trigonometric bits (not observed for the archived protocol).
