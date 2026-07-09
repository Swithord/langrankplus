# LMM fixed effects: Performance loss (%)

Reference categories are individual-distance methods, HRL targets, and mT5. Bold coefficients have p < 0.05.

| Term | Estimate [95% CI] | p | Direction |
| --- | --- | --- | --- |
| Baseline: individual, HRL, mT5 | **22.45 [6.43, 38.47]** | 0.006 | Higher loss |
| Composite vs individual, among HRL | **-6.56 [-9.76, -3.36]** | <0.001 | Lower loss |
| Trained vs individual, among HRL | -3.06 [-6.26, 0.13] | 0.060 | No clear effect |
| LRL vs HRL, for individual methods | **22.97 [6.89, 39.04]** | 0.005 | Higher loss |
| MRL vs HRL, for individual methods | -1.12 [-18.56, 16.33] | 0.900 | No clear effect |
| XLM-R vs mT5 | -0.91 [-6.74, 4.92] | 0.759 | No clear effect |
| Composite × LRL: change in LRL-HRL gap | **-8.19 [-11.45, -4.93]** | <0.001 | Lower loss |
| Trained × LRL: change in LRL-HRL gap | **-22.17 [-25.44, -18.91]** | <0.001 | Lower loss |
| Composite × MRL: change in MRL-HRL gap | 1.41 [-2.14, 4.95] | 0.437 | No clear effect |
| Trained × MRL: change in MRL-HRL gap | -1.20 [-4.74, 2.35] | 0.508 | No clear effect |
| Target-language variance component | 16.00 | — | Target heterogeneity |

