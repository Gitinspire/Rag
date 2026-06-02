"""
protein_tools.py - Tools for fetching data from RCSB PDB and UniProt
==================================================================
This module provides a unified interface for the RAG agent to retrieve
real-time protein structural and functional data.
"""

import requests
import json
from rcsbapi.search import TextQuery, AttributeQuery
from rcsbapi.data import DataQuery

def search_rcsb_pdb(query_text: str, limit: int = 5):
    """
    Search RCSB PDB for protein IDs matching a text query.
    Returns a list of "ID: Title" strings.
    """
    try:
        q1 = TextQuery(query_text)
        results = q1.exec()
        pdb_results = []
        for i, pdb_id in enumerate(results):
            if i >= limit:
                break
            # Fetch a quick title for each ID to make results more readable
            try:
                url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    title = r.json().get("struct", {}).get("title", "No title")
                    pdb_results.append(f"{pdb_id}: {title}")
                else:
                    pdb_results.append(pdb_id)
            except:
                pdb_results.append(pdb_id)
        return pdb_results
    except Exception as e:
        return [f"Error searching RCSB: {str(e)}"]
def get_rcsb_pdb_metadata(pdb_id: str):
    """
    Fetch comprehensive scientific metadata for a PDB entry.
    Captures resolution, ligands, UniProt mappings, and experimental details.
    """
    pdb_id = pdb_id.upper()
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    
    # Robust request with retries
    import time
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                text = response.text.strip()
                if not text:
                    print(f" Empty response for {pdb_id} (Attempt {attempt+1})")
                    time.sleep(1)
                    continue
                entry_data = json.loads(text)
                break
            elif response.status_code == 404:
                return f"Error: RCSB PDB entry {pdb_id} not found."
            else:
                print(f" Status {response.status_code} for {pdb_id} (Attempt {attempt+1})")
                time.sleep(1)
        except Exception as e:
            print(f" Request error for {pdb_id} (Attempt {attempt+1}): {e}")
            time.sleep(1)
    else:
        return f"Error: RCSB PDB API failed for {pdb_id} after multiple attempts."

    try:
        # Determine the best image suffix
        model_count = entry_data.get("rcsb_entry_info", {}).get("deposited_model_count", 1)
        suffix = "models" if model_count > 1 else "assembly-1"
        
        # 1. Fetch Polymer Entities (Proteins/Nucleic Acids)
        polymers = []
        poly_ids = entry_data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
        for p_id in poly_ids:
            p_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{p_id}"
            try:
                p_res = requests.get(p_url, timeout=5)
                if p_res.status_code == 200:
                    p_data = p_res.json()
                    polymers.append({
                        "id": p_id,
                        "name": p_data.get("polymer_entity", {}).get("pdbx_description", "N/A"),
                        "type": p_data.get("entity_poly", {}).get("rcsb_entity_polymer_type", "N/A"),
                        "uniprot_ids": p_data.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", []),
                        "chains": p_data.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", []),
                        "sequence": p_data.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "N/A")
                    })
            except: continue

        # 2. Fetch Non-Polymer Entities (Ligands/Cofactors/Drugs)
        ligands = []
        np_ids = entry_data.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids", [])
        for np_id in np_ids:
            np_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{np_id}"
            try:
                np_res = requests.get(np_url, timeout=5)
                if np_res.status_code == 200:
                    np_data = np_res.json()
                    comp_id = np_data.get("pdbx_entity_nonpoly", {}).get("comp_id", np_id)
                    ligands.append({
                        "id": np_id,
                        "comp_id": comp_id,
                        "name": np_data.get("rcsb_nonpolymer_entity", {}).get("pdbx_description", "N/A"),
                        "chemical_formula": np_data.get("rcsb_nonpolymer_entity", {}).get("formula", "N/A"),
                        "formula_weight": np_data.get("rcsb_nonpolymer_entity", {}).get("formula_weight", "N/A"),
                        "molecule_count": np_data.get("rcsb_nonpolymer_entity", {}).get("pdbx_number_of_molecules", "N/A"),
                        "binding_chains": np_data.get("rcsb_nonpolymer_entity_container_identifiers", {}).get("auth_asym_ids", []),
                        "interactive_view_instructions": f"To see detailed interactions (H-bonds, stackings) for {comp_id}, tell the user to click '3D View -> Ligand Interaction' on the RCSB site."
                    })
            except: continue

        # 3. Assemble Scientific Summary
        sym = entry_data.get("symmetry", {})
        refine = entry_data.get("refine", [{}])[0] if entry_data.get("refine") else {}
        
        # 4. Proactive Identification of Bioactive Components (Ligands + Peptide Inhibitors)
        bioactive = []
        for l in ligands:
            bioactive.append({
                "id": l["comp_id"],
                "name": l["name"],
                "type": "Small Molecule / Ligand",
                "chains": l["binding_chains"]
            })
        for p in polymers:
            # Peptide inhibitors are usually short polymers (< 100 residues)
            seq = p.get("sequence", "")
            if seq and seq != "N/A" and len(seq) < 100:
                bioactive.append({
                    "id": p["id"],
                    "name": p["name"],
                    "type": "Peptide / Short Polymer Component",
                    "chains": p["chains"],
                    "sequence": seq
                })

        summary = {
            "pdb_id": pdb_id,
            "title": entry_data.get("struct", {}).get("title"),
            "bioactive_components_summary": bioactive,
            "resolution_angstrom": entry_data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
            "classification": entry_data.get("struct_keywords", {}).get("pdbx_keywords"),
            "experimental_method": entry_data.get("exptl", [{}])[0].get("method"),
            "uniprot_mappings": {p['id']: p['uniprot_ids'] for p in polymers if p['uniprot_ids']},
            "image_url": f"https://cdn.rcsb.org/images/structures/{pdb_id.lower()}_{suffix}.jpeg",
            "release_date": entry_data.get("rcsb_accession_info", {}).get("initial_release_date"),
            "organism_common": entry_data.get("rcsb_entry_info", {}).get("polymer_entity_count_common_name"),
            "macromolecule_weight_kda": entry_data.get("rcsb_entry_info", {}).get("molecular_weight"),
            "atom_count": entry_data.get("rcsb_entry_info", {}).get("deposited_atom_count"),
            "modeled_residue_count": entry_data.get("rcsb_entry_info", {}).get("deposited_modeled_polymer_monomer_count"),
            "structural_quality": {
                "r_work": refine.get("ls_R_factor_R_work"),
                "r_free": refine.get("ls_R_factor_R_free")
            },
            "crystallography_cell": entry_data.get("cell", {}),
            "space_group": sym.get("space_group_name_H_M") if sym else None,
            "software_pipeline": [s.get("name") for s in entry_data.get("software", []) if s.get("name")],
            "polymers": polymers,
            "ligands": ligands
        }
        
        # 6. Citations & Funding (Suppress placeholders if missing)
        citation_list = entry_data.get("citation", [{}])
        citation = citation_list[0] if citation_list else {}
        
        summary["citation"] = {
            "doi": citation.get("pdbx_database_id_DOI"),
            "pubmed_id": citation.get("pdbx_database_id_PubMed"),
            "journal": citation.get("title")
        }
        # If both are missing, set the whole citation to None
        if not summary["citation"]["doi"] and not summary["citation"]["pubmed_id"]:
            summary["citation"] = None
            
        summary["funding"] = [f.get("funding_organization") for f in entry_data.get("rcsb_external_references_funding", []) if f.get("funding_organization")]
        if not summary["funding"]:
            summary["funding"] = None

        return json.dumps(summary)
    except Exception as e:
        return f"Error processing scientific metadata for {pdb_id}: {str(e)}"

def get_uniprot_metadata(accession: str):
    """
    Fetch protein annotations and sequence from UniProt.
    """
    url = f"https://www.uniprot.org/uniprotkb/{accession}.json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        summary = {
            "accession": accession,
            "full_name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value"),
            "gene": data.get("genes", [{}])[0].get("geneName", {}).get("value"),
            "organism": data.get("organism", {}).get("scientificName"),
            "function": data.get("comments", [{}])[0].get("texts", [{}])[0].get("value") if data.get("comments") else "N/A",
            "sequence": data.get("sequence", {}).get("value"),
            "sequence_length": data.get("sequence", {}).get("length")
        }
        return json.dumps(summary)
    except Exception as e:
        return f"Error fetching UniProt data for {accession}: {str(e)}"


# Example usage for verification
if __name__ == "__main__":
    print("Testing RCSB Search (Insulin)...")
    print(search_rcsb_pdb("Insulin"))
    
    print("\nTesting RCSB Metadata (4HHB)...")
    print(get_rcsb_pdb_metadata("4HHB"))
    
    print("\nTesting UniProt (P04637 - p53)...")
    print(get_uniprot_metadata("P04637"))
