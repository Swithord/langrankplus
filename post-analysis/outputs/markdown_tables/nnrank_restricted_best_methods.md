# NNRank-restricted best methods

The best method is selected by mean performance loss on the NNRank-compatible subset.

| Dataset | Task | Model | Metric | Best method | Group | PL | Raw gap | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_mt5 | masakhaner | mt5 | f1_score | **Typological** | individual | **3.28 [1.04, 6.44]** | **2.66 [0.80, 5.32]** | 10 |
| masakhaner_xlm-r | masakhaner | xlm-r | f1_score | **Geographic** | individual | **5.31 [1.31, 9.59]** | **2.48 [0.72, 4.45]** | 10 |
| opus100_mt5 | opus100 | mt5 | bleu | **NNRank** | nnrank | **15.03 [9.49, 21.19]** | **1.59 [0.82, 2.56]** | 46 |
| sib200_mt5 | sib200 | mt5 | f1_score | **LightGBM** | trained | **10.02 [7.60, 12.87]** | **7.29 [5.37, 9.74]** | 74 |
| sib200_xlm-r | sib200 | xlm-r | f1_score | **LightGBM** | trained | **14.67 [9.85, 20.30]** | **8.86 [5.59, 13.00]** | 74 |
| taxi1500_mt5 | taxi1500 | mt5 | f1_score | **LightGBM** | trained | **18.63 [17.36, 19.92]** | **5.11 [4.64, 5.60]** | 732 |
| taxi1500_xlm-r | taxi1500 | xlm-r | f1_score | **LightGBM** | trained | **9.76 [8.61, 10.97]** | **2.38 [2.07, 2.72]** | 732 |
| tydiqa_mt5 | tydiqa | mt5 | f1_score | **LightGBM** | trained | **9.50 [4.18, 15.01]** | **5.95 [2.64, 9.46]** | 8 |
| tydiqa_xlm-r | tydiqa | xlm-r | f1_score | **Typological** | individual | **7.21 [1.68, 13.96]** | **4.21 [1.11, 8.01]** | 8 |
| ud_dep_xlm-r | ud_dep | xlm-r | f1_score | **Typological** | individual | **22.07 [16.15, 28.44]** | **8.47 [5.93, 11.37]** | 64 |
| ud_pos_mt5 | ud_pos | mt5 | f1_score | **LightGBM** | trained | **22.67 [17.60, 28.28]** | **7.26 [5.55, 9.23]** | 64 |
| ud_pos_xlm-r | ud_pos | xlm-r | f1_score | **NNRank** | nnrank | **14.27 [9.96, 19.35]** | **5.67 [4.12, 7.47]** | 64 |
| wikiann_mt5 | wikiann | mt5 | f1_score | **NNRank** | nnrank | **7.12 [4.88, 9.67]** | **5.10 [3.63, 6.79]** | 56 |
| wikiann_xlm-r | wikiann | xlm-r | f1_score | **NNRank** | nnrank | **12.29 [8.08, 17.43]** | **6.44 [4.74, 8.30]** | 56 |
| xnli_mt5 | xnli | mt5 | f1_score | **Wikipedia size** | individual | **2.05 [1.37, 2.75]** | **1.47 [1.00, 1.96]** | 10 |
| xnli_xlm-r | xnli | xlm-r | f1_score | **Wikipedia size** | individual | **0.55 [0.24, 0.91]** | **0.42 [0.18, 0.69]** | 10 |
| xquad_mt5 | xquad | mt5 | f1_score | **MLP** | trained | **15.79 [6.72, 25.95]** | **4.49 [1.85, 7.34]** | 9 |
| xquad_xlm-r | xquad | xlm-r | f1_score | **LightGBM** | trained | **3.71 [0.35, 7.58]** | **0.66 [0.06, 1.38]** | 9 |

