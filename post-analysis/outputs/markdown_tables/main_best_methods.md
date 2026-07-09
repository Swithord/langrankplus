# Best method by dataset-model setting

The best method is selected by mean performance loss.

| Dataset | Task | Model | Metric | Best method | Group | PL | Raw gap | n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masakhaner_mt5 | masakhaner | mt5 | f1_score | **Genetic** | individual | **5.09 [3.20, 7.22]** | **4.01 [2.50, 5.72]** | 19 |
| masakhaner_xlm-r | masakhaner | xlm-r | f1_score | **LightGBM** | trained | **7.89 [4.66, 11.58]** | **3.80 [2.31, 5.48]** | 19 |
| opus100_mt5 | opus100 | mt5 | bleu | **MLP** | trained | **18.45 [13.44, 23.73]** | **2.10 [1.39, 2.89]** | 87 |
| sib200_mt5 | sib200 | mt5 | f1_score | **Script** | individual | **13.00 [10.82, 15.43]** | **8.86 [7.36, 10.55]** | 185 |
| sib200_xlm-r | sib200 | xlm-r | f1_score | **Genetic** | individual | **23.86 [19.79, 28.15]** | **12.19 [10.23, 14.34]** | 185 |
| taxi1500_mt5 | taxi1500 | mt5 | f1_score | **LightGBM** | trained | **18.85 [17.55, 20.16]** | **5.32 [4.80, 5.86]** | 762 |
| taxi1500_xlm-r | taxi1500 | xlm-r | f1_score | **LightGBM** | trained | **9.96 [8.84, 11.13]** | **2.57 [2.22, 2.93]** | 762 |
| tydiqa_mt5 | tydiqa | mt5 | f1_score | **Composite-RRF** | composite | **8.65 [3.64, 13.99]** | **5.47 [2.32, 8.83]** | 9 |
| tydiqa_xlm-r | tydiqa | xlm-r | f1_score | **Composite-RRF** | composite | **6.07 [1.33, 11.87]** | **3.48 [0.84, 6.76]** | 9 |
| ud_dep_xlm-r | ud_dep | xlm-r | f1_score | **LightGBM** | trained | **23.84 [19.45, 28.60]** | **10.00 [7.80, 12.42]** | 147 |
| ud_pos_mt5 | ud_pos | mt5 | f1_score | **LightGBM** | trained | **25.62 [21.91, 29.49]** | **7.96 [6.66, 9.32]** | 147 |
| ud_pos_xlm-r | ud_pos | xlm-r | f1_score | **Composite-RRF** | composite | **15.04 [12.48, 17.78]** | **6.30 [5.12, 7.63]** | 147 |
| wikiann_mt5 | wikiann | mt5 | f1_score | **Typological** | individual | **8.87 [7.25, 10.68]** | **6.50 [5.42, 7.67]** | 144 |
| wikiann_xlm-r | wikiann | xlm-r | f1_score | **Composite-Equal** | composite | **13.40 [10.78, 16.28]** | **8.02 [6.70, 9.39]** | 144 |
| xnli_mt5 | xnli | mt5 | f1_score | **Genetic** | individual | **3.24 [1.12, 6.90]** | **2.29 [0.82, 4.82]** | 15 |
| xnli_xlm-r | xnli | xlm-r | f1_score | **Composite-Equal** | composite | **1.33 [0.52, 2.37]** | **0.98 [0.39, 1.74]** | 15 |
| xquad_mt5 | xquad | mt5 | f1_score | **LightGBM** | trained | **21.08 [10.08, 33.73]** | **6.52 [3.10, 10.35]** | 12 |
| xquad_xlm-r | xquad | xlm-r | f1_score | **LightGBM** | trained | **32.27 [17.77, 46.55]** | **10.74 [5.71, 15.73]** | 12 |

