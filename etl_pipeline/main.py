import argparse
import sys
import os
from extractor import run_extractor
from transformer import run_transformer
from loader import load_to_vector_db

def main():
    parser = argparse.ArgumentParser(description="InnovLabs.AI Biological ETL Pipeline")
    parser.add_argument("--ids", nargs="+", help="List of PDB IDs to process", required=True)
    parser.add_argument("--skip-extract", action="store_true", help="Skip the extraction phase")
    parser.add_argument("--skip-transform", action="store_true", help="Skip the transformation phase")
    parser.add_argument("--skip-load", action="store_true", help="Skip the vector loading phase")

    args = parser.parse_args()
    pdb_ids = [pid.upper() for pid in args.ids]

    print("============================================================")
    print("      InnovLabs.AI — Biological ETL Pipeline")
    print("============================================================")
    print(f"Targeting PDB IDs: {', '.join(pdb_ids)}")

    # 1. Extraction
    if not args.skip_extract:
        print("\n[PHASE 1] Extracting raw data from RCSB PDB to S3...")
        run_extractor(pdb_ids)
    else:
        print("\n[PHASE 1] Skipping Extraction.")

    # 2. Transformation
    if not args.skip_transform:
        print("\n[PHASE 2] Transforming raw S3 JSON into deep structural summaries...")
        run_transformer(pdb_ids)
    else:
        print("\n[PHASE 2] Skipping Transformation.")

    # 3. Loading
    if not args.skip_load:
        print("\n[PHASE 3] Indexing transformed summaries into Vector DB...")
        load_to_vector_db(pdb_ids)
    else:
        print("\n[PHASE 3] Skipping Loading.")

    print("\n============================================================")
    print("              ETL FLOW COMPLETE")
    print("============================================================")

if __name__ == "__main__":
    main()
