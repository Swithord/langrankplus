# LMM fixed effects: Raw oracle gap

Reference categories are individual-distance methods, HRL targets, and mT5. Bold coefficients have p < 0.05.

| Term | Estimate [95% CI] | p | Direction |
| --- | --- | --- | --- |
| Baseline: individual, HRL, mT5 | **11.01 [4.40, 17.62]** | 0.001 | Higher loss |
| Composite vs individual, among HRL | **-3.76 [-5.08, -2.44]** | <0.001 | Lower loss |
| Trained vs individual, among HRL | -0.30 [-1.62, 1.02] | 0.656 | No clear effect |
| LRL vs HRL, for individual methods | 0.96 [-5.67, 7.59] | 0.776 | No clear effect |
| MRL vs HRL, for individual methods | 0.40 [-6.79, 7.60] | 0.912 | No clear effect |
| XLM-R vs mT5 | 0.36 [-2.05, 2.76] | 0.770 | No clear effect |
| Composite × LRL: change in LRL-HRL gap | 0.43 [-0.92, 1.77] | 0.536 | No clear effect |
| Trained × LRL: change in LRL-HRL gap | **-5.36 [-6.71, -4.01]** | <0.001 | Lower loss |
| Composite × MRL: change in MRL-HRL gap | 0.67 [-0.80, 2.13] | 0.372 | No clear effect |
| Trained × MRL: change in MRL-HRL gap | -1.45 [-2.91, 0.01] | 0.052 | No clear effect |
| Target-language variance component | 16.00 | — | Target heterogeneity |

