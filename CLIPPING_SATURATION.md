Dependence in the actual DP-SGD dynamics

The reviewer rightly highlights that the model 1 is formulated for the canary observation assuming temporal independence. Since our previous response, we have developed a martingale analysis that directly incorporates this temporal dependence. We do want to caution that this increased realism comes with substantially heavier mathematical machinery, which may obscure the main intuition. Therefore, we still prefer to first present model 1's formulation in the main text given that it leads to cleaner mathematical intuition, it's broadly understandable for readers, and the independence is not fundamental to the Gaussian perspective underlying our auditor.

We will include the newly developed formal statements and proofs in the appendix and add the following high-level explanation to the main text.

Let the history up to training step t-1 include the complete DP-SGD trajectory. The canary direction and magnitude at step t may depend on this history. Conditional on the history, however, the canary-sampling bit and the newly added DP noise remain fresh. Consequently, the centered per-step score can be represented as a martingale difference, while its predictable mean and conditional variance are determined by the current round canary magnitude. We summarize three concrete conclusions from the new analysis below.

First, for the Dirac gradient canaries used in our audit, the canary direction and saturated magnitude are fixed. The per-step observation is exactly a Bernoulli canary contribution plus fresh Gaussian noise. Therefore, Model 1 is exact for this construction, and the main-body convergence result applies directly.

Second, the same conclusion holds for a saturated input-space canary whose direction may be adaptive with the DP-SGD dynamics. Although the direction is history-dependent, it is fixed conditional on the past. Because the added Gaussian noise is isotropic, its projection onto this adaptive canary’s direction remains Gaussian with the same variance. If the canary magnitude is always fixed, the direction dependence therefore disappears from the scalar score law, recovering Model 1 exactly.
To demonstrate that it is possible to hit the clipping strength every step of the training run, we ran an experiment with a constant red image with label 6 (mislabelled). At every step we recorded the canary's pre-clip gradient norm.

| Quantity                                                  | Value                    |
|-----------------------------------------------------------|--------------------------|
| Minimum gradeint of canary ($\min_t \lVert g_t \rVert / C$) | 5.32 (at step 24)    |
| Fraction of steps with $\lVert g_t \rVert \ge C$          | **1.0000** (2500 / 2500) |
| Clipping ratio at step 1 / late training                  | 12.3 / 50–130            |

This shows that exact saturation is achievable by a genuine input-space canary in a DP-SGD run.

Third, natural input-space canaries need not remain saturated, so their magnitudes may also be adaptive with the DP-SGD dynamics. For this case, our martingale result requires a weaker stability condition: the first moment and second moment of the time-averaged canary magnitude must approach deterministic values. We show that such more natural input-space canaries can be constructed within actual DP-SGD dynamics. The martingale result then gives the same Gaussian-pair conclusion, with this limiting effective magnitude replacing the constant clipping norm in Model 1.
