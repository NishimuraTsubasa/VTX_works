from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
required = [
    ROOT / 'outputs/data/Model_Outputs.xlsx',
    ROOT / 'outputs/summary/Summary.xlsx',
    ROOT / 'outputs/report/Model_Report.pdf',
    ROOT / 'data/input/Universe_Master.xlsx',
    ROOT / 'data/input/Cross_Asset_Data.xlsx',
    ROOT / 'data/processed/Monthly_Model_Dataset.xlsx',
]
for path in required:
    assert path.exists(), path
portfolio = pd.read_excel(ROOT / 'outputs/data/Model_Outputs.xlsx', sheet_name='Latest_Portfolio')
assert len(portfolio) == 12
assert abs(portfolio.dynamic_weight.sum()) < 1e-6
print('smoke test passed')
