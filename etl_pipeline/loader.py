import os
import requests
import json
import boto3
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

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

# RAG Configuration
DB_PATH = "./chroma_db"
EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_from_s3(s3_path):
    """Fetches a JSON object from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_path)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching from S3 ({s3_path}): {str(e)}")
        return None

def load_to_vector_db(pdb_ids):
    """Downloads transformed summaries from S3 and indexes them in ChromaDB."""
    documents = []
    
    for pdb_id in pdb_ids:
        pdb_id = pdb_id.upper()
        s3_path = f"transformed/{pdb_id}_summary.json"
        data = get_from_s3(s3_path)
        
        if data:
            print(f"Loading {pdb_id} to Vector DB...")
            
            def fmt(val):
                if val is None or val == "" or val == "N/A":
                    return None
                if isinstance(val, list):
                    if not val: return None
                    return ", ".join(map(str, val))
                return str(val)

            items = []
            
            # 1. Identity
            ident = data.get('identity', {})
            title = fmt(ident.get('title'))
            sect = [f"[PDB ID: {pdb_id}] ### Protein Identity & Classification"]
            if title: sect.append(f"- **Full Title**: {title}")
            classification = fmt(ident.get('classification'))
            if classification: sect.append(f"- **Classification**: {classification}")
            organism = fmt(ident.get('organism'))
            if organism: sect.append(f"- **Organism**: {organism}")
            expression = fmt(ident.get('expression_system'))
            if expression: sect.append(f"- **Expression System**: {expression}")
            mutations = fmt(ident.get('mutations'))
            if mutations: sect.append(f"- **Mutations**: {mutations}")
            items.append("\n".join(sect))
            
            # ... (experimental sections remain similar)
            
            # 5. Ligands
            ligands = data.get('ligands', [])
            if ligands:
                sect = [f"[PDB ID: {pdb_id}] ### Ligand & Small Molecule Details"]
                for ligand in ligands:
                    l_name = ligand['name']
                    sect.append(f"#### Ligand: {l_name} ({ligand['comp_id']})")
                    sect.append(f"- **Chemical Formula**: {fmt(ligand['formula'])}")
                    sect.append(f"- **Binding Chains**: {fmt(ligand['chains'])}")
                    sect.append(f"- **InChI Key**: {fmt(ligand.get('inchi_key'))}")
                    sect.append(f"- **Scientific Role**: {fmt(ligand.get('role'))}")
                    sect.append(f"- **Interactive Viewing**: {fmt(ligand.get('interactive_view_instructions'))}")
                items.append("\n".join(sect))
            
            # (Rest of sections...)
            # (Software)
            software = data.get('software', [])
            if software:
                sect = [f"### Experimental Software Pipeline"]
                sect.append(f"- **Software Used**: {fmt(software)}")
                items.append("\n".join(sect))

            # 7. Assembly
            assm = data.get('assembly', {})
            if assm and any(assm.values()):
                sect = [f"### Biological Assembly"]
                sect.append(f"- **Assembly ID**: {fmt(assm.get('id'))}")
                sect.append(f"- **Stoichiometry**: {fmt(assm.get('stoichiometry'))}")
                sect.append(f"- **Symmetry**: {fmt(assm.get('symmetry'))}")
                items.append("\n".join(sect))

            # 8. Annotations
            annots = data.get('sequence_annotations', [])
            if annots:
                sect = [f"[PDB ID: {pdb_id}] ### Sequence Domains & Annotations"]
                for annot in annots:
                    sect.append(f"- **[{fmt(annot['type'])}]**: {fmt(annot['name'])} - {fmt(annot['description'])}")
                items.append("\n".join(sect))
            
            # 8.5. Protein Sequence
            polymers = data.get('polymers', [])
            if polymers:
                sect = [f"[PDB ID: {pdb_id}] ### Protein Sequence"]
                for poly in polymers:
                    poly_name = poly.get('name', 'Macromolecule')
                    seq = poly.get('sequence', 'Not documented')
                    sect.append(f"#### Chain(s) {fmt(poly.get('chains', 'N/A'))}: {poly_name}")
                    sect.append(f"Sequence: {seq}")
                items.append("\n".join(sect))

            # 9. Provenance
            prov = data.get('provenance', {})
            sect = [f"[PDB ID: {pdb_id}] ### Citation & Provenance"]
            doi = fmt(prov.get('doi'))
            if doi: sect.append(f"- **DOI**: {doi}")
            pmid = fmt(prov.get('pubmed_id'))
            if pmid: sect.append(f"- **PubMed ID**: {pmid}")
            funding = fmt(prov.get('funding'))
            if funding: sect.append(f"- **Funding**: {funding}")
            if len(sect) > 1:
                items.append("\n".join(sect))
            
            content = "\n\n".join(items) if items else ""
            
            if content is not None:
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "RCSB PDB/UniProt",
                        "pdb_id": pdb_id,
                        "user_id": "public",  # ALL scientific data is public by default
                        "type": "protein_metadata",
                        "image_path": data.get("visual_url"),
                        "doi": prov.get('doi')
                    }
                )
                documents.append(doc)

    if documents:
        print(f"Indexing {len(documents)} documents to {DB_PATH}...")
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=EMBEDDINGS
        )
        
        # Clean up existing documents for these IDs to avoid duplicates
        for pdb_id in pdb_ids:
            vector_store.delete(where={"pdb_id": pdb_id.upper()})
            
        vector_store.add_documents(documents)
        print("Indexing complete.")
    else:
        print("No documents found to index.")

if __name__ == "__main__":
    test_ids = ["4HHB", "1FJS", "7HHB", "6LU7", "1BOM", "1AIE"]
    load_to_vector_db(test_ids)
