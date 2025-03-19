import pandas as pd
import sys

def normalize_time(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    for r in range(len(df)):
        for c in range(6, 20):
            if df.iat[r, c] > 1:
                df.iat[r, c] =1
    
    df.to_csv(output_csv, index=False)

def normalize_pkt_size(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    for r in range(len(df)):
        for c in range(5, 20):
            if df.iat[r, c] > 1500:
                df.iat[r, c] =1
            else:
                df.iat[r, c] /= 1500
    
    df.to_csv(output_csv, index=False)

def normalize_window(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    for r in range(len(df)):
        for c in range(5, 20):
            if df.iat[r, c] > 60000:
                df.iat[r, c] =1
            else:
                df.iat[r, c] /= 60000
    
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python Normalization.py <input_csv> <output_csv> <function_num>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    function_num = int(sys.argv[3])

    match function_num:
        case 1:
            normalize_time(input_csv, output_csv)
        case 2:
            normalize_pkt_size(input_csv, output_csv)
        case 3:
            normalize_window(input_csv, output_csv)

