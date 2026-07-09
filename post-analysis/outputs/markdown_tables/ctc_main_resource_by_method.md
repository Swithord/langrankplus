# CTC resource-conditional summary

Rows use only dataset-model settings meeting the minimum-target threshold. Coverage values at or above 90% are bolded.

| Model | Method | Group | Resource | Dataset cells | Targets | Trial complexity | Pool fraction | Near-oracle coverage | Exact-best coverage | Best-in-set PL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mt5 | ASJP | individual | HRL | 5 | 9 | 37.57 | 65.1% | **91.4%** | 77.1% | 1.46 |
| mt5 | Composite-Equal | composite | HRL | 5 | 9 | 30.00 | 53.1% | **94.3%** | 80.0% | 1.60 |
| mt5 | Composite-RRF | composite | HRL | 5 | 9 | 31.40 | 56.5% | **94.3%** | 80.0% | 1.18 |
| mt5 | Genetic | individual | HRL | 5 | 9 | 33.89 | 54.6% | **94.3%** | 77.1% | 0.78 |
| mt5 | Geographic | individual | HRL | 5 | 9 | 32.46 | 60.1% | **100.0%** | 74.3% | 0.26 |
| mt5 | LightGBM | trained | HRL | 5 | 9 | 26.03 | 44.3% | 88.6% | 71.4% | 1.80 |
| mt5 | MLP | trained | HRL | 5 | 9 | 24.34 | 41.9% | **94.3%** | 68.6% | 1.42 |
| mt5 | Script | individual | HRL | 5 | 9 | 31.94 | 56.5% | **91.4%** | 74.3% | 1.61 |
| mt5 | Typological | individual | HRL | 5 | 9 | 33.71 | 61.9% | **91.4%** | 71.4% | 1.38 |
| mt5 | Wikipedia size | individual | HRL | 5 | 9 | 39.06 | 72.1% | **97.1%** | 88.6% | 0.37 |
| mt5 | ASJP | individual | MRL | 5 | 45 | 38.26 | 62.7% | **96.9%** | 79.3% | 0.84 |
| mt5 | Composite-Equal | composite | MRL | 5 | 45 | 32.83 | 52.8% | **97.9%** | 78.8% | 0.70 |
| mt5 | Composite-RRF | composite | MRL | 5 | 45 | 33.52 | 55.1% | **97.4%** | 84.5% | 0.61 |
| mt5 | Genetic | individual | MRL | 5 | 45 | 36.13 | 53.9% | **99.0%** | 75.6% | 0.42 |
| mt5 | Geographic | individual | MRL | 5 | 45 | 34.14 | 57.8% | **95.3%** | 76.7% | 1.01 |
| mt5 | LightGBM | trained | MRL | 5 | 45 | 28.35 | 46.9% | **95.9%** | 75.6% | 0.67 |
| mt5 | MLP | trained | MRL | 5 | 45 | 26.84 | 44.5% | **96.4%** | 74.1% | 0.85 |
| mt5 | Script | individual | MRL | 5 | 45 | 34.50 | 58.9% | **95.3%** | 71.0% | 1.02 |
| mt5 | Typological | individual | MRL | 5 | 45 | 34.94 | 60.9% | **98.4%** | 74.1% | 0.60 |
| mt5 | Wikipedia size | individual | MRL | 5 | 45 | 40.46 | 70.4% | **99.5%** | 83.4% | 0.30 |
| mt5 | ASJP | individual | LRL | 5 | 899 | 34.24 | 78.5% | **93.2%** | 81.2% | 1.09 |
| mt5 | Composite-Equal | composite | LRL | 5 | 899 | 28.21 | 63.1% | **91.8%** | 80.2% | 1.21 |
| mt5 | Composite-RRF | composite | LRL | 5 | 899 | 31.54 | 72.6% | **92.5%** | 80.9% | 1.08 |
| mt5 | Genetic | individual | LRL | 5 | 899 | 30.55 | 64.1% | **92.1%** | 79.8% | 1.17 |
| mt5 | Geographic | individual | LRL | 5 | 899 | 31.95 | 75.7% | **93.3%** | 80.8% | 1.11 |
| mt5 | LightGBM | trained | LRL | 5 | 899 | 17.47 | 35.4% | **92.2%** | 78.2% | 1.20 |
| mt5 | MLP | trained | LRL | 5 | 899 | 18.36 | 37.9% | 89.9% | 74.7% | 1.57 |
| mt5 | Script | individual | LRL | 5 | 899 | 22.76 | 48.7% | **90.8%** | 77.0% | 1.49 |
| mt5 | Typological | individual | LRL | 5 | 899 | 30.81 | 73.2% | **92.6%** | 78.3% | 1.23 |
| mt5 | Wikipedia size | individual | LRL | 5 | 899 | 34.62 | 82.0% | **90.1%** | 77.8% | 1.52 |
| xlm-r | ASJP | individual | HRL | 5 | 9 | 42.36 | 57.5% | **100.0%** | 83.3% | 0.18 |
| xlm-r | Composite-Equal | composite | HRL | 5 | 9 | 42.58 | 51.3% | **97.2%** | 77.8% | 0.60 |
| xlm-r | Composite-RRF | composite | HRL | 5 | 9 | 43.22 | 52.2% | **100.0%** | 86.1% | 0.15 |
| xlm-r | Genetic | individual | HRL | 5 | 9 | 47.08 | 56.9% | **97.2%** | 77.8% | 0.61 |
| xlm-r | Geographic | individual | HRL | 5 | 9 | 49.67 | 61.9% | **97.2%** | 80.6% | 0.45 |
| xlm-r | LightGBM | trained | HRL | 5 | 9 | 22.14 | 28.3% | 88.9% | 58.3% | 3.01 |
| xlm-r | MLP | trained | HRL | 5 | 9 | 22.44 | 36.2% | **97.2%** | 80.6% | 0.38 |
| xlm-r | Script | individual | HRL | 5 | 9 | 45.19 | 62.3% | **91.7%** | 66.7% | 1.60 |
| xlm-r | Typological | individual | HRL | 5 | 9 | 41.22 | 57.5% | **91.7%** | 83.3% | 1.01 |
| xlm-r | Wikipedia size | individual | HRL | 5 | 9 | 54.56 | 71.8% | **97.2%** | 88.9% | 0.36 |
| xlm-r | ASJP | individual | MRL | 5 | 45 | 43.06 | 52.8% | **96.4%** | 77.7% | 0.96 |
| xlm-r | Composite-Equal | composite | MRL | 5 | 45 | 44.88 | 48.6% | **94.3%** | 77.2% | 0.79 |
| xlm-r | Composite-RRF | composite | MRL | 5 | 45 | 45.68 | 49.4% | **94.3%** | 80.3% | 0.87 |
| xlm-r | Genetic | individual | MRL | 5 | 45 | 50.44 | 55.3% | **96.9%** | 76.7% | 0.72 |
| xlm-r | Geographic | individual | MRL | 5 | 45 | 53.35 | 58.2% | **95.3%** | 73.6% | 0.89 |
| xlm-r | LightGBM | trained | MRL | 5 | 45 | 21.20 | 26.0% | **90.7%** | 61.7% | 1.43 |
| xlm-r | MLP | trained | MRL | 5 | 45 | 22.16 | 32.6% | **93.8%** | 65.3% | 1.24 |
| xlm-r | Script | individual | MRL | 5 | 45 | 49.41 | 63.1% | **96.4%** | 74.1% | 0.75 |
| xlm-r | Typological | individual | MRL | 5 | 45 | 43.07 | 55.4% | **95.9%** | 76.2% | 0.74 |
| xlm-r | Wikipedia size | individual | MRL | 5 | 45 | 57.11 | 70.4% | **98.4%** | 82.9% | 0.43 |
| xlm-r | ASJP | individual | LRL | 5 | 899 | 37.41 | 74.5% | 88.4% | 79.4% | 2.70 |
| xlm-r | Composite-Equal | composite | LRL | 5 | 899 | 33.42 | 57.9% | 89.2% | 80.3% | 2.03 |
| xlm-r | Composite-RRF | composite | LRL | 5 | 899 | 35.67 | 63.3% | 88.7% | 80.8% | 2.21 |
| xlm-r | Genetic | individual | LRL | 5 | 899 | 37.85 | 65.3% | 88.4% | 78.6% | 2.25 |
| xlm-r | Geographic | individual | LRL | 5 | 899 | 43.87 | 80.9% | **91.6%** | 86.3% | 1.77 |
| xlm-r | LightGBM | trained | LRL | 5 | 899 | 13.88 | 23.6% | 88.3% | 76.6% | 2.32 |
| xlm-r | MLP | trained | LRL | 5 | 899 | 20.25 | 46.4% | 89.7% | 78.4% | 2.07 |
| xlm-r | Script | individual | LRL | 5 | 899 | 35.29 | 67.5% | **93.7%** | 85.1% | 1.19 |
| xlm-r | Typological | individual | LRL | 5 | 899 | 31.89 | 60.4% | 89.0% | 77.9% | 2.07 |
| xlm-r | Wikipedia size | individual | LRL | 5 | 899 | 40.31 | 73.8% | **93.2%** | 86.8% | 1.34 |

