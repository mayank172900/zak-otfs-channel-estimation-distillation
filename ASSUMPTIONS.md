# Assumptions

This file is updated as implementation details are fixed.

## Logged assumptions

1. Pilot location
The paper does not specify a non-zero point-pilot location in the excerpt. The default implementation uses the origin `(0, 0)` and logs this in metadata. Sensitivity hooks are included to test alternate pilot positions.

2. Effective channel from pulse shaping
The paper specifies GS pulse parameters and the matched-filter definition, but does not provide a fully expanded sampled `h_eff[k,l]` expression for the fractional-delay fractional-Doppler Vehicular-A case in the short conference paper. The implementation therefore uses two paths:
- a slow reference path based on the matched-filter overlap integrals from the cited GS-filter paper
- a faster crystalline-regime path using the paper-consistent phase structure
The fast path is retained only because it is numerically close to the slow reference path on controlled tests.

3. Path gains
The path gains are modeled as circularly symmetric complex Gaussian with variances proportional to the Vehicular-A power profile and normalized so that the average total channel power is one.

4. 8-QAM constellation
The default 8-QAM is a standard unit-average-energy cross 8-QAM. An alternate variant hook is included for sensitivity analysis if BER mismatch is material.

5. Thresholding usage
The paper footnote implies thresholding on the conventional read-off taps. The implementation evaluates both raw and thresholded read-off, records both, and uses the thresholded estimate for the default detection path.

6. Ambiguity evaluation window
The paper’s support set is an unwrapped integer-lattice window around the origin. The implementation now computes read-off on that unwrapped window directly rather than using wrapped `M x N` ambiguity indices.

7. Detection uses the unwrapped support-domain channel image
The estimator/CNN operate on a `27 x 43` support image. Embedding that image into a `31 x 37` torus aliases distinct Doppler taps because `43 > 37`. The implementation therefore applies MMSE detection directly with the unwrapped support-domain channel representation and only uses the `M x N` embedding for visualization/debugging.
# Update (2026-03-28)

- The detector now explicitly assumes the repo's transmitted DD data vector has per-entry variance `E_d / (MN)`, so the LMMSE ridge is `MN * N0 / E_d`.
- Fast-vs-reference effective-channel comparisons should be treated as a numerics choice, not a hidden implementation detail; use the explicit fast/reference configs in `configs/`.
- Final paper-faithful numbers have not been regenerated in this patch because this handoff is code-only for Windows execution.
