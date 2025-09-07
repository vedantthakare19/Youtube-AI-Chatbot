# Youtube-AI-Chatbot

A collection of notebooks exploring data preparation, modeling, and marketing budget optimization using multiple approaches: linear programming, evolutionary algorithms (DEAP), and reinforcement learning (PPO). The workflow includes weekly modeling, feature engineering, and multi-channel budget allocation under seasonality and business constraints.

## Repository structure
- `GC_data_7.ipynb`: Data prep and GA-based monthly budget allocation with seasonality weighting.
- `Gc_Model_Weekly.ipynb`: End-to-end data ingestion from Kaggle, weekly aggregation, and baseline regression modeling.
- `gc-data-opt.ipynb`: Reinforcement learning (PPO) approach to optimize channel budgets using a simulated environment.
- `Optimized_Budget.ipynb`: Linear programming and DEAP-based channel/monthly optimization with channel importance and seasonality.
- `Optimization/Optimization.ipynb`: Colab-style GA implementation for monthly allocation; utilities for constrained crossover and mutation.
- `Optimization/data.csv`: Sample monthly dataset with channel spends, GMV, NPS, and Stock Index.
- `download.png`: Project image (optional).

## Data sources
- Kaggle dataset referenced: `nabayansaha/gc-data-compiled` (compiled transactional and media spend data). Some notebooks download/extract in Colab paths like `/content/gc_data/...`.
- Local example dataset: `Optimization/data.csv` used by `Optimized_Budget.ipynb`.

## Quickstart
1) Install prerequisites
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -U pip
pip install numpy pandas matplotlib seaborn scikit-learn scipy deap torch
```

2) Launch notebooks
```bash
pip install jupyter
jupyter notebook
```

3) Open the notebook of interest and run cells top-to-bottom. See notes below for path tweaks if running outside Colab.

## Notebook guide
### GC_data_7.ipynb
- Loads prepared data, scales spend fields, computes channel weights from average contribution, ROI, NPS and stock correlations.
- Optimizes monthly allocation across 12 months using a genetic algorithm with seasonality alignment and smoothness terms.

### Gc_Model_Weekly.ipynb
- Authenticates with Kaggle, downloads `gc-data-compiled` dataset, and performs EDA and weekly aggregation.
- Builds a baseline weekly model (e.g., linear regression) with temporal features and media channels, producing evaluation metrics.

### gc-data-opt.ipynb
- Defines a marketing environment and trains a PPO agent (PyTorch) to learn channel allocation policies under a fixed total budget.
- Prints episodic revenues and outputs the final budget distribution.

### Optimized_Budget.ipynb
- Loads `Optimization/data.csv`, scales budget values to currency, derives channel importance and seasonality.
- Implements linear programming (SciPy `linprog`) for monthly allocation and channel distribution with constraints (min spend, max per channel).
- Includes DEAP-based GA for alternative optimization and comparison.

### Optimization/Optimization.ipynb
- Colab-oriented GA implementation using DEAP with custom operators like `constrained_crossover` and `budget_preserving_mutation` to enforce budget constraints and non-negativity.

## Paths and environment notes
- Colab paths like `/content/...` and `/kaggle/input/...` appear in some notebooks. If running locally:
  - Replace `/content/...` with your local project paths (e.g., `Optimization/...`).
  - Download Kaggle datasets via the Kaggle API (`kaggle datasets download -d nabayansaha/gc-data-compiled`) and update read paths accordingly.
- Ensure the following columns exist when using `Optimization/data.csv` workflows: `Year, Month, Total Investment, TV, Digital, Sponsorship, Content Marketing, Online marketing, Affiliates, SEM, Radio, Other, gmv, Stock Index, NPS`.

## Reproducing results
- GA runs are stochastic. Set seeds where provided or increase generations/population for stability.
- LP solutions are deterministic given the same inputs, bounds, and constraints.
- PPO training is stochastic; results vary with seeds, learning rate, and episode count.

## Typical constraints and objectives used
- Monthly allocation objective: align spend with seasonality while maintaining smooth month-over-month changes.
- Channel allocation objective: maximize a weighted sum of historical GMV contribution, ROI proxy, and NPS correlation.
- Constraints: non-negativity, exact annual or monthly budget totals, minimum per-month spend, and optional cap on single-channel share.

## How to adapt to your data
- Update path constants to your CSV(s).
- Verify column names (e.g., `Online marketing` vs `Online Marketing`) and normalize to a consistent schema.
- Recompute channel weights if you change the weighting scheme between contribution, ROI, NPS, and stock correlations.

## Requirements
- Python 3.9–3.12
- Packages: numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, deap, torch, jupyter, statsmodels (for some EDA cells).

## Troubleshooting
- File not found: adjust notebook paths from Colab-style to local paths.
- Infeasible LP: check bounds and equality constraints sum; ensure budgets and minimums are consistent.
- GA convergence: tune population size, mutation/crossover rates, and generations.
- PPO instability: lower learning rate, increase episodes, or adjust reward scaling.

## License and attribution
- This project uses public datasets from Kaggle. Please review the dataset license on the Kaggle page.
- Code in the notebooks is provided for research and educational use.
