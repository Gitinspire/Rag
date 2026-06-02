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

def get_from_s3(s3_path):
    """Fetches a JSON object from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_path)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching from S3 ({s3_path}): {str(e)}")
        return None

def upload_to_s3(data, s3_path):
    """Uploads a dictionary as JSON to S3."""
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_path,
            Body=json.dumps(data, indent=2),
            ContentType="application/json"
        )
    except Exception as e:
        print(f"Error uploading to S3 ({s3_path}): {str(e)}")

def get_inchi_key(comp_id):
    """Fetches InChI key from RCSB Chemical Component API."""
    try:
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            # Correct path is rcsb_chem_comp_descriptor.in_ch_ikey
            desc = data.get("rcsb_chem_comp_descriptor", {})
            return desc.get("in_ch_ikey", "N/A")
    except:
        pass
    return "N/A"

def transform_pdb_entry(pdb_id):
    """Transforms raw PDB JSON components from S3 into a deep scientific summary."""
    pdb_id = pdb_id.upper()
    print(f"Transforming PDB ID: {pdb_id}...")
    
    entry_data = get_from_s3(f"raw/entries/{pdb_id}.json")
    if not entry_data:
        return

    # Helper objects
    refine = entry_data.get("refine", [{}])[0] if entry_data.get("refine") else {}
    cell = entry_data.get("cell", [{}])[0] if isinstance(entry_data.get("cell"), list) else entry_data.get("cell", {})
    sym = entry_data.get("symmetry", {})
    struct = entry_data.get("struct", {})
    entry_info = entry_data.get("rcsb_entry_info", {})
    container_ids = entry_data.get("rcsb_entry_container_identifiers", {})
    
    # 1. Identity & Classification
    identity = {
        "pdb_id": pdb_id,
        "title": struct.get("title"),
        "classification": entry_data.get("struct_keywords", {}).get("pdbx_keywords"),
        "organism": entry_info.get("polymer_entity_count_common_name"),
        "expression_system": [s.get("rcsb_gene_name", {}).get("value") for s in entry_data.get("rcsb_entity_source_gen", []) if s.get("rcsb_gene_name")],
        "mutations": f"{entry_info.get('deposited_polymer_entity_instance_count_mutated', 0)} mutation(s) detected" if entry_info.get('deposited_polymer_entity_instance_count_mutated') else None
    }

    # 2. Experimental Method
    experimental = {
        "method": entry_data.get("exptl", [{}])[0].get("method", "X-RAY DIFFRACTION"),
        "resolution": entry_info.get("resolution_combined", ["N/A"])[0],
        "structural_quality": {
            "r_work": refine.get("ls_R_factor_R_work"),
            "r_free": refine.get("ls_R_factor_R_free"),
            "r_observed": refine.get("ls_R_factor_obs"),
            "goodness_of_fit": entry_data.get("pdbx_vrpt_summary", {}).get("restraint_outlier_instances")
        }
    }

    # 3. Crystal / Unit Cell
    crystallography = {
        "space_group": sym.get("space_group_name_hm") or sym.get("space_group_name_H_M", "N/A"),
        "unit_cell": {
            "a": cell.get("length_a"),
            "b": cell.get("length_b"),
            "c": cell.get("length_c"),
            "alpha": cell.get("angle_alpha"),
            "beta": cell.get("angle_beta"),
            "gamma": cell.get("angle_gamma")
        }
    }

    # 4. Macromolecule Parameters
    macromolecule = {
        "weight_kda": entry_info.get("molecular_weight"),
        "atom_count": entry_info.get("deposited_atom_count"),
        "modeled_residue_count": entry_info.get("deposited_model_polymer_residue_count"),
        "deposited_residue_count": entry_info.get("deposited_polymer_monomer_count"),
        "unique_protein_chains": container_ids.get("polymer_entity_ids", []),
        "uniprot_accessions": container_ids.get("uniprot_ids", []),
        "global_symmetry": entry_info.get("struct_asym_count")
    }

    # 5. Ligand Parameters (Small Molecules)
    ligands = []
    np_ids = container_ids.get("non_polymer_entity_ids", [])
    for np_id in np_ids:
        np_data = get_from_s3(f"raw/nonpolymer_entities/{pdb_id}_{np_id}.json")
        if np_data:
            np_entity = np_data.get("rcsb_nonpolymer_entity", {})
            np_ids_cont = np_data.get("rcsb_nonpolymer_entity_container_identifiers", {})
            comp_id = np_data.get("pdbx_entity_nonpoly", {}).get("comp_id", np_id)
            
            # Robust InChI Key retrieval
            inchi = np_ids_cont.get("inchi_key")
            if not inchi or inchi == "N/A":
                inchi = get_inchi_key(comp_id)
            
            ligands.append({
                "id": np_id,
                "comp_id": comp_id,
                "name": np_entity.get("pdbx_description"),
                "formula": np_entity.get("formula"),
                "inchi_key": inchi if inchi != "N/A" else None,
                "molecule_count": np_entity.get("pdbx_number_of_molecules"),
                "chains": np_data.get("rcsb_nonpolymer_entity_container_identifiers", {}).get("auth_asym_ids", []),
                "role": "Subject of Investigation" if np_id == "1" else "Artifact/Buffer",
                "interactive_view_instructions": f"To see 3D contacts for {comp_id}, visit RCSB and select '3D View -> Ligand Interaction'."
            })

    # Biological Assembly
    assembly_raw_list = entry_data.get("rcsb_struct_assembly", [])
    assembly_raw = assembly_raw_list[0] if assembly_raw_list else {}
    assembly = {
        "id": assembly_raw.get("rcsb_id"),
        "stoichiometry": assembly_raw.get("rcsb_struct_assembly_prop", {}).get("stoichiometry"),
        "symmetry": assembly_raw.get("rcsb_struct_assembly_prop", {}).get("type")
    }

    # Sequence & Annotation
    annotations = []
    polymers = []
    for p_id in container_ids.get("polymer_entity_ids", []):
        p_data = get_from_s3(f"raw/polymer_entities/{pdb_id}_{p_id}.json")
        if p_data:
            # Extract Sequence
            polymers.append({
                "id": p_id,
                "name": p_data.get("polymer_entity", {}).get("pdbx_description", "N/A"),
                "chains": p_data.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", []),
                "sequence": p_data.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "Not documented")
            })
            # Extract Annotations (Suppress if no description or "Not documented")
            for annot in p_data.get("rcsb_polymer_entity_annotation", []):
                desc = annot.get("description")
                if annot.get("type") in ["Pfam", "Mutation"] and desc and desc not in ["N/A", "Not documented"]:
                    annotations.append({
                        "type": annot.get("type"),
                        "name": annot.get("name") or annot.get("annotation_id"),
                        "description": desc
                    })

    # Citation & Provenance
    provenance = {
        "deposited": entry_data.get("rcsb_accession_info", {}).get("deposit_date"),
        "released": entry_data.get("rcsb_accession_info", {}).get("initial_release_date"),
        "doi": entry_data.get("citation", [{}])[0].get("pdbx_database_id_DOI"),
        "pubmed_id": entry_data.get("citation", [{}])[0].get("pdbx_database_id_PubMed"),
        "funding": [f.get("organization") for f in entry_data.get("rcsb_external_references_funding", [])],
        "deposition_group": [g.get("group_description") for g in entry_data.get("pdbx_deposit_group", [])]
    }

    # Assemble Unified Deep Summary (Clean internal keys for RAG)
    summary = {
        "pdb_id": pdb_id,
        "identity": identity,
        "visual_url": f"https://cdn.rcsb.org/images/structures/{pdb_id.lower()}_assembly-1.jpeg",
        "experimental": experimental,
        "crystallography": crystallography,
        "macromolecule": macromolecule,
        "ligands": ligands,
        "software": [s.get("name") for s in entry_data.get("software", [])],
        "assembly": assembly,
        "sequence_annotations": annotations,
        "provenance": provenance
    }
    
    upload_to_s3(summary, f"transformed/{pdb_id}_summary.json")
    print(f"Successfully transformed and saved s3://{S3_BUCKET}/transformed/{pdb_id}_summary.json")

def run_transformer(pdb_ids):
    """Runs transformation for a list of PDB IDs."""
    for pid in pdb_ids:
        transform_pdb_entry(pid)

if __name__ == "__main__":
    test_ids = ["4HHB", "1FJS", "7HHB", "6LU7", "1BOM", "1AIE"]
    run_transformer(test_ids)
