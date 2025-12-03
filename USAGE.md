# Usage

## Option A — Conda
```
conda env create -f environment.yml
conda activate default-risk-ml
jupyter lab
```

## Option B — pip + venv
```
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Then open `notebooks/Default-Risk-Prediction.ipynb`, set `DATA_PATH = '../data/freddiemac.csv'`, and run all cells.
