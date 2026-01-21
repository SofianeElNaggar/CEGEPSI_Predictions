import csv
import pandas as pd

def count_csv_rows():
    csv_path = "./dataset/OMDS-CTD-meteogc-data.parquet"  # <-- Mets ici le path de ton CSV

    df = pd.read_parquet(csv_path)
    print(df.shape)

count_csv_rows()
