# English baseline comparison

The best ranker is selected by mean performance loss on the English-available target subset. Positive reductions mean the best ranker improves over always-English. Bold marks the lower loss.

| Dataset | Task | Model | Metric | n | English PL | Best ranker | Best ranker PL | PL reduction | English raw gap | Best raw gap | Raw-gap reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sib200_mt5 | sib200 | mt5 | f1_score | 185 | 21.15 [19.21, 23.25] | **Script** | **13.00 [10.82, 15.43]** | +8.15 | 14.97 [13.91, 16.19] | **8.86 [7.36, 10.55]** | +6.11 |
| sib200_xlm-r | sib200 | xlm-r | f1_score | 185 | **13.29 [11.61, 15.04]** | **Genetic** | 23.86 [19.79, 28.15] | -10.57 | **6.94 [6.36, 7.52]** | 12.19 [10.23, 14.34] | -5.26 |
| taxi1500_mt5 | taxi1500 | mt5 | f1_score | 762 | 57.27 [55.86, 58.63] | **LightGBM** | **18.85 [17.55, 20.16]** | +38.42 | 13.23 [12.91, 13.54] | **5.32 [4.80, 5.86]** | +7.91 |
| taxi1500_xlm-r | taxi1500 | xlm-r | f1_score | 762 | 44.97 [43.70, 46.22] | **LightGBM** | **9.96 [8.84, 11.13]** | +35.01 | 10.06 [9.73, 10.39] | **2.57 [2.22, 2.93]** | +7.49 |
| tydiqa_mt5 | tydiqa | mt5 | f1_score | 9 | 27.51 [16.04, 40.00] | **Composite-RRF** | **8.65 [3.64, 13.99]** | +18.87 | 22.09 [12.86, 32.33] | **5.47 [2.32, 8.83]** | +16.62 |
| tydiqa_xlm-r | tydiqa | xlm-r | f1_score | 9 | 22.90 [13.85, 33.56] | **Composite-RRF** | **6.07 [1.33, 11.87]** | +16.83 | 16.03 [9.32, 24.85] | **3.48 [0.84, 6.76]** | +12.55 |
| ud_pos_mt5 | ud_pos | mt5 | f1_score | 147 | 37.26 [34.41, 40.10] | **LightGBM** | **25.62 [21.91, 29.49]** | +11.64 | 13.30 [11.99, 14.68] | **7.96 [6.66, 9.32]** | +5.34 |
| ud_pos_xlm-r | ud_pos | xlm-r | f1_score | 147 | 34.09 [30.74, 37.44] | **Composite-RRF** | **15.04 [12.48, 17.78]** | +19.05 | 19.04 [16.52, 21.72] | **6.30 [5.12, 7.63]** | +12.74 |
| wikiann_mt5 | wikiann | mt5 | f1_score | 144 | 14.55 [13.17, 16.00] | **Typological** | **8.87 [7.25, 10.68]** | +5.68 | 11.64 [10.57, 12.77] | **6.50 [5.42, 7.67]** | +5.15 |
| wikiann_xlm-r | wikiann | xlm-r | f1_score | 144 | 26.44 [23.73, 29.39] | **Composite-Equal** | **13.40 [10.78, 16.28]** | +13.04 | 18.64 [16.90, 20.52] | **8.02 [6.70, 9.39]** | +10.62 |
| xnli_mt5 | xnli | mt5 | f1_score | 15 | **2.55 [1.65, 3.52]** | **Genetic** | 3.24 [1.12, 6.90] | -0.69 | **1.85 [1.20, 2.54]** | 2.29 [0.82, 4.82] | -0.44 |
| xnli_xlm-r | xnli | xlm-r | f1_score | 15 | 4.04 [2.93, 5.15] | **Composite-Equal** | **1.33 [0.52, 2.37]** | +2.70 | 3.05 [2.24, 3.84] | **0.98 [0.39, 1.74]** | +2.07 |
| xquad_mt5 | xquad | mt5 | f1_score | 12 | 22.21 [14.39, 30.71] | **LightGBM** | **21.08 [10.08, 33.73]** | +1.13 | 6.96 [4.45, 9.64] | **6.52 [3.10, 10.35]** | +0.44 |
| xquad_xlm-r | xquad | xlm-r | f1_score | 12 | 57.25 [53.91, 60.47] | **LightGBM** | **32.27 [17.77, 46.55]** | +24.98 | 20.75 [18.37, 23.42] | **10.74 [5.71, 15.73]** | +10.01 |

