import pandas as pd
import sys

def format_featrue(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    mask = df["src_ip"].notna() & df["src_ip"].astype(str).str.startswith("192").eq(False)
    df.loc[mask, ["src_ip", "dst_ip", "src_port", "dst_port"]] = df.loc[mask, ["dst_ip", "src_ip", "dst_port", "src_port"]].values
    df.loc[mask, "direction"] = 1 - df.loc[mask, "direction"]

    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python FormatFeatrue.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    format_featrue(input_csv, output_csv)