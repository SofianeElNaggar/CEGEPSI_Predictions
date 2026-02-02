import csv
import pandas as pd

def count_csv_rows():
    csv_path = "./dataset/Data Alain/OMDS-CTD-meteogc-data-v2.csv" 

    df = pd.read_csv(csv_path)
    print(df.shape)

count_csv_rows()
