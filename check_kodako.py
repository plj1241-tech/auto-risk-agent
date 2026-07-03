import pandas as pd

df = pd.read_parquet("data/processed/panel_annual_v2.parquet")
kodako = df[df['corp_name'] == '코다코'][
    ['year', 'total_liab', 'total_equity', 'current_assets',
     'current_liab', 'revenue', 'op_income', 'interest_exp',
     'debt_ratio', 'current_ratio', 'icr', 'op_margin', 'z_score', 'n_quarters']
]
print(kodako.to_string())
