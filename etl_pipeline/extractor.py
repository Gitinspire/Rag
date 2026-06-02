import os
import requests
import json
import boto3
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# AWS Configuration
S3_BUCKET = "etl-processing-pdb-storage"
S3_REGION = os.getenv("AWS_REGION", "us-east-1")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=S3_REGION
)

def upload_to_s3(data, s3_path):
    """Uploads a dictionary as JSON to S3."""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_path,
            Body=json.dumps(data, indent=2),
            ContentType="application/json"
        )
        print(f"Successfully uploaded to s3://{S3_BUCKET}/{s3_path}")
    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")

def fetch_pdb_entry(pdb_id):
    """Fetches full metadata for a PDB ID and uploads components to S3."""
    pdb_id = pdb_id.upper()
    print(f"\nProcessing PDB ID: {pdb_id}...")
    
    # 1. Main Entry
    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = requests.get(entry_url)
        if r.status_code == 200:
            entry_data = r.json()
            upload_to_s3(entry_data, f"raw/entries/{pdb_id}.json")
            
            # 2. Polymer Entities
            poly_ids = entry_data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
            for p_id in poly_ids:
                p_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{p_id}"
                p_res = requests.get(p_url)
                if p_res.status_code == 200:
                    upload_to_s3(p_res.json(), f"raw/polymer_entities/{pdb_id}_{p_id}.json")
            
            # 3. Non-Polymer Entities (Ligands)
            np_ids = entry_data.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids", [])
            for np_id in np_ids:
                np_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{np_id}"
                np_res = requests.get(np_url)
                if np_res.status_code == 200:
                    upload_to_s3(np_res.json(), f"raw/nonpolymer_entities/{pdb_id}_{np_id}.json")
        else:
            print(f"Failed to fetch entry {pdb_id}: HTTP {r.status_code}")
    except Exception as e:
        print(f"Error processing {pdb_id}: {str(e)}")

def run_extractor(pdb_ids):
    """Runs extraction for a list of PDB IDs."""
    for pid in pdb_ids:
        fetch_pdb_entry(pid)

if __name__ == "__main__":
    # Sample list for verification
    test_ids = ["4HHB", "1FJS", "7HHB", "6LU7", "1BOM"]
    run_extractor(test_ids)
