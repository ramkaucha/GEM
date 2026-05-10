#!/usr/bin/env python3

import argparse
import pandas as pd
import re


def build_code_map(code_csv_path):
    code_df = pd.read_csv(code_csv_path)

    # code.csv column B: Code, column C: Description
    code_col = code_df.columns[1]
    desc_col = code_df.columns[2]

    return {
        str(int(row[code_col])).strip(): str(row[desc_col]).strip()
        for _, row in code_df.iterrows()
        if pd.notna(row[code_col]) and pd.notna(row[desc_col])
    }

def map_aha_description(value, code_map):
    if pd.isna(value):
        return ""
    
    value = str(value).strip()

    # Handling OR: 1;2 
    if ";" in value:
        parts = [p.strip() for p in value.split(";")]
        description = [code_map.get(p, f"[Unknown code: {p}]") for p in parts]
        return " OR ".join(description)
    
    # Handling AND: 1+2
    if "+" in value:
        parts = [p.strip() for p in value.split("+")]
        description = [code_map.get(p, f"[Unknown code: {p}]") for p in parts]
        return " AND ".join(description)
    
    # Single code
    return code_map.get(value, f"[Unknown code: {value}]")

def main():
    parser = argparse.ArgumentParser(description="Append AHA description to metadata")
    parser.add_argument("metadata_csv", help="Path to metadata.csv file")
    parser.add_argument("code_csv", help="Path to code.csv file with code-description mapping")
    parser.add_argument(
        "-o", "--output",
        default="metadata_with_description.csv",
        help="Output CSV Path"
    )
    
    args = parser.parse_args()

    code_map = build_code_map(args.code_csv)

    metadata_df = pd.read_csv(args.metadata_csv)

    # metadata.csv: column B = AHA_Code
    aha_col = metadata_df.columns[1]

    # Append as Column H/8th column
    new_col_name = "AHA_Description"
    descriptions = metadata_df[aha_col].apply(lambda x: map_aha_description(x, code_map))

    if new_col_name in metadata_df.columns:
        metadata_df[new_col_name] = descriptions
    else:
        metadata_df.insert(7, new_col_name, descriptions)

    metadata_df.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()