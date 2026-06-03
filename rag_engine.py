"""
rag_engine.py - Core RAG logic (provider-agnostic, reusable)
=============================================================
This module is the "brain" shared between:
  - rag_app.py  (terminal chatbot)
  - api.py      (FastAPI web server)
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import protein_tools

load_dotenv()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROVIDER = os.getenv("PROVIDER", "gemini")   # "bedrock" or "gemini"

# Bedrock models
BEDROCK_LLM_MODEL       = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
AWS_REGION              = os.getenv("AWS_REGION", "us-east-1")

# Gemini models
GEMINI_LLM_MODEL        = "models/gemini-2.5-flash"
GEMINI_EMBEDDING_MODEL  = "models/gemini-embedding-001"

# NVIDIA models
NVIDIA_LLM_MODEL = "meta/llama-3.1-8b-instruct"

# RAG settings
BASE_DIR           = os.path.dirname(__file__)
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
CHROMA_DB_DIR      = os.getenv("CHROMA_DB_DIR", os.path.join(BASE_DIR, "chroma_db"))
CHUNK_SIZE         = 1000
CHUNK_OVERLAP      = 200
TOP_K              = 12


# ==============================================================================
# PROVIDER INIT
# ==============================================================================

def init_llm(provider_id=None, model_id=None):
    """Initialize a specific LLM based on provider and model IDs."""
    # Use defaults from .env if not provided
    provider_id = provider_id or os.getenv("PROVIDER", "gemini").lower()
    
    if provider_id == "bedrock":
        from langchain_aws import ChatBedrockConverse
        m_id = model_id or BEDROCK_LLM_MODEL
        llm = ChatBedrockConverse(
            model=m_id,
            region_name=AWS_REGION,
            temperature=0.1,
            max_tokens=1024
        )
        provider_info = {"provider": "AWS Bedrock", "model": m_id}

    elif provider_id == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        if not os.getenv("NVIDIA_API_KEY"):
            raise RuntimeError("Missing NVIDIA_API_KEY in .env")
        m_id = model_id or NVIDIA_LLM_MODEL
        llm = ChatNVIDIA(model=m_id, temperature=0.1)
        provider_info = {"provider": "NVIDIA NIM", "model": m_id}

    else: # Default to Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("Missing GOOGLE_API_KEY in .env")
        m_id = model_id or GEMINI_LLM_MODEL
        llm = ChatGoogleGenerativeAI(model=m_id, temperature=0.1)
        provider_info = {"provider": "Google Gemini", "model": m_id}

    return llm, provider_info


def init_embeddings():
    """Initialize embeddings (usually static for the vector store)."""
    provider = os.getenv("PROVIDER", "gemini").lower()
    if provider == "bedrock":
        from langchain_aws import BedrockEmbeddings
        return BedrockEmbeddings(model_id=BEDROCK_EMBEDDING_MODEL, region_name=AWS_REGION)
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)


# ==============================================================================
# VECTOR STORE (Persistent)
# ==============================================================================

def get_vector_store(embeddings):
    """Load existing vector store (Pinecone or Chroma DB) or build from scratch."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    if pinecone_api_key and pinecone_index_name:
        print(f" Connecting to Pinecone index: {pinecone_index_name}...")
        from langchain_pinecone import PineconeVectorStore
        vector_store = PineconeVectorStore(
            index_name=pinecone_index_name,
            embedding=embeddings,
            pinecone_api_key=pinecone_api_key
        )
        print(f"   Pinecone connection active")
        return vector_store

    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print(f" Loading existing index from disk...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings
        )
        print(f"   Index loaded")
    else:
        print(f"No existing index found. Building from knowledge base...")
        docs   = _load_documents()
        chunks = _chunk_documents(docs)
        print(f"Embedding {len(chunks)} chunks into ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print(f"   Database created")
    return vector_store


def _load_documents():
    print("Step 1: Loading documents...")
    loader = DirectoryLoader(
        KNOWLEDGE_BASE_DIR, glob="**/*.txt",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    # Tag all internal company docs as public so the user_id filter finds them
    for doc in docs:
        doc.metadata["user_id"] = "public"
    print(f"   {len(docs)} documents loaded")
    return docs


def _chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    # Safeguard: Remove any docs with None content to satisfy Pydantic/Chroma
    safe_docs = [d for d in docs if d.page_content is not None]
    chunks = splitter.split_documents(safe_docs)
    return chunks


# ==============================================================================
# RAG CHAIN
# ==============================================================================

# Locally indexed proteins (highly optimized)
INDEXED_PDB_IDS = [
    "4HHB", "1FJS", "7HHB", "6LU7", "1BOM", "1AIE", "1TQN", "2V00", "3WPG", "4GQR",
    "5T8L", "6XDG", "7K7W", "1MBY", "1A02", "1B6W", "1CFC", "1D66", "1E6Y", "1F6R",
    "1G6U", "1H6V", "1I6W", "1J6X", "1K6Y", "1L6Z", "1M70", "1N71", "1O72", "1P73",
    "1Q74", "1R75", "1S76", "1T77", "1U78", "1V79", "1W7A", "1X7B", "1Y7C", "1Z7D",
    "2A7E", "2B7F", "2C7G", "2D7H", "2E7I", "2F7J", "2G7K", "2H7L", "2I7M", "2J7N"
]

def build_rag_chain(vector_store, llm):
    """Build the retriever → prompt → LLM → parser pipeline."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template("""You are a scientific AI assistant — a premium Scientific Discovery Platform.

Your goal is to provide high-fidelity reports. You have access to:
1. **Bioinformatics Data**: Structural data from RCSB PDB and functional data from UniProt.
2. **Internal Corporate Data**: Documents about the scientific platform (certifications, ETL processes, security, etc.).

### KNOWLEDGE CONTEXT PRIORITY:
1. **Tool Results**: If you see "SYSTEM: tool results", prioritize that for real-time protein data.
2. **Context**: Use the provided {context} for internal platform info and optimized proteins: {indexed_ids}.

### RESPONSE POLICY:
1. **Read the question carefully and respond ONLY to what was asked.**
   - Asked about "structure" or "overview"    → include: title, method, resolution, chains
   - Asked about "sequence"                   → include: polymer sequences only
   - Asked about "ligand" or "compound"       → include: ligand details only
   - Asked about "crystallography" or "cell"  → include: space group, unit cell dimensions
   - Asked about "quality" or "refinement"    → include: R-work, R-free only
   - Asked about "citation" or "paper"        → include: DOI, PubMed, journal only
   - Asked about "full report" or "all"       → include: all sections in clinical-grade layout
2. If the query is about the internal platform internally, provide a detailed, factual response based on the CONTEXT.
3. **Absolute Suppression**: DO NOT use "Not documented" or "N/A". For internal platform queries, if data is missing from context, omit it silently. For general scientific or biological queries (e.g. general questions about proteins, DNA, or general science that are not platform-specific), you may answer them using your own general knowledge.
4. **NEVER write image URLs in your text response.** The UI automatically displays the protein image. Do not include any URL, link, or reference to an image file. If the user asks for an image, diagram, structure, or visualization, you must confirm that the image/structure is being displayed in the panel (e.g., "I have retrieved the structure image for 2ACO, which is displayed in the panel.") and provide a brief description of the protein's title, classification, or resolution from the metadata so the text response is not empty.

---

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")
    
    # Fill in the indexed IDs once at build time
    prompt = prompt.partial(indexed_ids=", ".join(INDEXED_PDB_IDS))

    # The chain now takes a dictionary with "context" and "question"
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# ==============================================================================
# ANSWER + SOURCE RETRIEVAL
# ==============================================================================

def format_raw_tool_results(tool_results: list[str]) -> str:
    """Format raw JSON tool results into beautiful markdown tables/lists."""
    import json
    formatted_parts = []
    
    for record in tool_results:
        lines = record.split("\n", 1)
        if len(lines) < 2:
            formatted_parts.append(record)
            continue
            
        header, json_part = lines[0], lines[1]
        try:
            data = json.loads(json_part)
        except Exception:
            formatted_parts.append(record)
            continue
            
        md = []
        md.append(f"### {header.strip(':')}")
        
        if "UNIPROT" in header.upper():
            md.append(f"- **Accession**: {data.get('accession', 'N/A')}")
            md.append(f"- **Name**: {data.get('full_name', 'N/A')}")
            md.append(f"- **Gene**: {data.get('gene', 'N/A')}")
            md.append(f"- **Organism**: {data.get('organism', 'N/A')}")
            md.append(f"- **Sequence Length**: {data.get('sequence_length', 'N/A')} residues")
            md.append(f"\n**Function**:\n{data.get('function', 'N/A')}")
        else:
            # PDB Record
            md.append(f"- **PDB ID**: {data.get('pdb_id', 'N/A')}")
            md.append(f"- **Title**: {data.get('title', 'N/A')}")
            
            res = data.get('resolution_angstrom')
            if res: md.append(f"- **Resolution**: {res} Å")
            
            method = data.get('experimental_method')
            if method: md.append(f"- **Method**: {method}")
            
            classification = data.get('classification')
            if classification: md.append(f"- **Classification**: {classification}")
            
            organism = data.get('organism_common')
            if organism: md.append(f"- **Organism**: {organism}")
            
            weight = data.get('macromolecule_weight_kda')
            if weight: md.append(f"- **Weight**: {weight} kDa")
            
            bioactive = data.get('bioactive_components_summary', [])
            if bioactive:
                md.append("\n**Bioactive Components / Ligand Bindings**:")
                for comp in bioactive:
                    c_type = comp.get('type', 'Component')
                    chains = ", ".join(comp.get('chains', []))
                    md.append(f"  - **{comp.get('id')}** ({comp.get('name')}): {c_type} (Chains: {chains})")
                    
            citation = data.get('citation')
            if citation and isinstance(citation, dict):
                journal = citation.get('journal')
                doi = citation.get('doi')
                pubmed = citation.get('pubmed_id')
                md.append("\n**Citation**:")
                if journal: md.append(f"  - *{journal}*")
                if doi: md.append(f"  - DOI: [{doi}](https://doi.org/{doi})")
                if pubmed: md.append(f"  - PubMed ID: {pubmed}")
                
        formatted_parts.append("\n".join(md))
        
    return "I've retrieved the following scientific data for you:\n\n" + "\n\n---\n\n".join(formatted_parts)

def ask(chain, retriever, question: str, user_id: str = "public") -> dict:
    import re
    # Check if the question is asking about InnovLabs
    if re.search(r'\binnov\s*labs\b', question, re.IGNORECASE):
        return {
            "answer": "I am a scientific AI assistant. I am not authorized to answer questions about InnovLabs.AI or its internal operations.",
            "sources": [],
            "image_paths": [],
            "provenance_type": "Restricted Query"
        }

    iteration = 0
    max_iterations = 3
    all_sources = []
    image_urls = []          # collect one image per protein
    tool_results = []
    
    # 1. Proactive ID Detection
    import re
    # Match standard 4-char PDB IDs (e.g., 4HHB)
    pdb_ids = re.findall(r'\b[0-9][a-zA-Z0-9]{3}\b', question)
    
    # Match UniProt Accessions (e.g., P04637, A0A5Y1I583)
    uniprot_ids = re.findall(r'\b(?:[O,P,Q][0-9][A-Z0-9]{3}[0-9]|[A-N,R-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b', question)
    
    if pdb_ids:
        print(f"PDB IDs detected: {pdb_ids}. Fetching live data...")
        for pdb_id in pdb_ids:
            pdb_id = pdb_id.upper()
            try:
                raw_res = protein_tools.get_rcsb_pdb_metadata(pdb_id)
                # Truncate ONLY for LLM context (token limit) — extract image FIRST from full response
                import json
                try:
                    meta = json.loads(raw_res)        # full parse for image URL
                    if meta.get("image_url"):
                        image_urls.append(meta["image_url"])
                except Exception:
                    pass  # non-JSON error string (e.g. "Error: not found") — skip image
                chunk_res = raw_res[:7000] if len(raw_res) > 7000 else raw_res
                tool_results.append(f"PROTEIN DATA BANK RECORD FOR {pdb_id}:\n{chunk_res}")
                all_sources.append("Global Protein Data Banks (RCSB/UniProt)")
            except Exception as e:
                print(f"Error fetching tool data for PDB {pdb_id}: {e}")

    if uniprot_ids:
        print(f"UniProt IDs detected: {uniprot_ids}. Fetching live data...")
        for uni_id in uniprot_ids:
            uni_id = uni_id.upper()
            try:
                raw_res = protein_tools.get_uniprot_metadata(uni_id)
                chunk_res = raw_res[:7000] if len(raw_res) > 7000 else raw_res
                # UniProt provides detailed functional text, so we emphasize it to the LLM
                tool_results.append(f"UNIPROT FUNCTIONAL DB RECORD FOR {uni_id}:\n{chunk_res}")
                all_sources.append("Global Protein Data Banks (RCSB/UniProt)")
            except Exception as e:
                print(f"Error fetching tool data for UniProt {uni_id}: {e}")

    # 2. Context Retrieval (Retriever with Multi-Tenant Filter)
    comparison_keywords = ["compare", "difference", "versus", "vs", "similar", "other", "better", "higher", "lower", "another"]
    is_comparison = any(k in question.lower() for k in comparison_keywords)
    
    k_boost = 20 if is_comparison else 12
    
    # LOGICAL ISOLATION: Apply filter to ensure user only sees 'public' data + their own data.
    # Note: Using $in to support multiple allowed IDs (public + user)
    # If using newer Chroma/LangChain, ensure terminology matches.
    filter_expr = {"user_id": {"$in": ["public", user_id]}} if user_id and user_id != "admin" else None
    
    print(f"RETRIEVAL with filter: {filter_expr}")
    retrieved_docs = retriever.vectorstore.similarity_search(
        question, 
        k=k_boost,
        filter=filter_expr
    )
    
    # If comparison is requested and we have tool results, find relative proteins by Name/Classification
    if is_comparison and tool_results:
        print(" Comparison query detected. Searching for relative proteins...")
        # Common protein families to look for in title/classification
        families = ["HEMOGLOBIN", "HAEMOGLOBIN", "PROTEASE", "INSULIN", "KINASE", "COLLAGEN", "MYOGLOBIN", "ENZYME", "POLYMERASE", "GLOBIN", "TRYPSIN", "THROMBIN"]
        
        for tr in tool_results:
            tr_upper = tr.upper()
            target_keywords = []
            
            # 1. Search for known families in the whole string
            for fam in families:
                if fam in tr_upper:
                    target_keywords.append(fam)
            
            # 2. Extract specific classification word
            m = re.search(r'"classification":\s*"(.*?)"', tr)
            if m:
                class_word = m.group(1).split()[-1].strip().upper()
                if len(class_word) > 4: # Avoid short noise
                    target_keywords.append(class_word)

            # Unique keywords to search
            unique_keywords = list(set(target_keywords))
            for keyword in unique_keywords:
                print(f"   Found category: {keyword}. Searching relatives in index...")
                rel_docs = retriever.vectorstore.similarity_search(keyword, k=10)
                retrieved_docs.extend(rel_docs)
    
    # Smarter context filtering: deduplicate and skip strict filters for comparisons
    filtered_chunks = []
    seen_contents = set()
    for d in retrieved_docs:
        content_id = d.page_content[:100]
        if content_id in seen_contents: continue
        seen_contents.add(content_id)
        
        content = d.page_content.upper()
        # 1. If it's a comparison query, keep EVERYTHING retrieved (related proteins)
        if is_comparison:
            filtered_chunks.append(d)
        # 2. If it's NOT a comparison, filter strictly by present IDs if any
        elif (pdb_ids or uniprot_ids):
            content_match = False
            for pid in pdb_ids:
                if pid.upper() in content: content_match = True
            for uid in uniprot_ids:
                if uid.upper() in content: content_match = True
            if content_match:
                filtered_chunks.append(d)
        # 3. Handle cases with no specific IDs or general context (Ensure internal documents are included)
        else:
            # If it's from our internal directory or doesn't look like a PDB record, keep it
            source = d.metadata.get('source', '').lower()
            if "internal" in source or "knowledge_base" in source or not any(x in d.page_content for x in ["PDB ID", "RCSB", "UniProt"]):
                filtered_chunks.append(d)

    context_text = "\n\n".join([f"[Source: {d.metadata.get('source', 'Internal')}] {d.page_content}" for d in filtered_chunks])

    # 3. Agentic Loop
    current_question = question
    answer = ""
    
    while iteration < max_iterations:
        iteration += 1
        
        # Consolidate ALL context to prevent redundancy
        unified_context = context_text
        if tool_results:
            unified_context = "### REAL-TIME MOLECULAR DATABASE UPDATES:\n" + "\n".join(tool_results) + "\n\n" + context_text
        
        # Build clean input for the chain
        input_data = {
            "context": unified_context,
            "question": question
        }

        response = chain.invoke(input_data)
        answer = response.strip()

        if not answer:
            print(f"Iteration {iteration}: Empty LLM response. Retrying...")
            continue

        # Check for tool calls
        has_tool_call = any(tool in answer for tool in ["SEARCH_PDB", "GET_UNIPROT_INFO", "GET_PDB_INFO"])
        
        if has_tool_call:
            print(f"Iteration {iteration}: Tool requested: {answer}")
            # Process tool calls
            new_tool_results = []
            
            # GET_PDB_INFO (In case it wasn't caught proactively)
            pdb_m = re.findall(r'GET_PDB_INFO\s*\(\s*(?:pdb_id\s*=\s*)?["\'](.*?)["\']\s*\)', answer)
            for pid in pdb_m:
                res = protein_tools.get_rcsb_pdb_metadata(pid)
                new_tool_results.append(f"GET_PDB_INFO Result for {pid}: {res[:10000]}")
                all_sources.append("Global Protein Data Banks (RCSB/UniProt)")
            
            # SEARCH_PDB
            search_m = re.findall(r'SEARCH_PDB\s*\(\s*(?:query\s*=\s*)?["\'](.*?)["\']\s*\)', answer)
            for q in search_m:
                res = protein_tools.search_rcsb_pdb(q)
                new_tool_results.append(f"SEARCH_PDB Result for '{q}': {res}")
                all_sources.append("Global Protein Data Banks (RCSB/UniProt)")

            # UNIPROT
            uni_m = re.findall(r'GET_UNIPROT_INFO\s*\(\s*(?:accession\s*=\s*)?["\'](.*?)["\']\s*\)', answer)
            for acc in uni_m:
                res = protein_tools.get_uniprot_metadata(acc)
                new_tool_results.append(f"GET_UNIPROT_INFO Result for {acc}: {res}")
                all_sources.append("Global Protein Data Banks (RCSB/UniProt)")

            if new_tool_results:
                tool_results.extend(new_tool_results)
                continue # Retry with new data
        
        # Strip all image references from LLM text — UI renders the image separately via image_url
        answer = re.sub(r'!\[.*?\]\(.*?\)', '', answer)                          # markdown images
        answer = re.sub(r'<img.*?>', '', answer, flags=re.IGNORECASE)              # html img tags
        answer = re.sub(r'https?://cdn\.rcsb\.org/\S+\.jpe?g', '', answer)       # bare RCSB CDN URLs
        answer = re.sub(r'(?i)(image|image url|structure image)\s*:\s*https?://\S+', '', answer)  # "Image: https://..." lines
        answer = re.sub(r'\n{3,}', '\n\n', answer)   # collapse blank lines left by stripping
        answer = answer.strip()
        break

    # 4. Final Polish
    if not answer and tool_results:
        answer = format_raw_tool_results(tool_results)

    # 5. Source Attribution & Image logic
    # Fallback: only pull images from Vector DB docs if a protein was actually in the question
    # Prevents unrelated queries (e.g. "Show me RAG diagram") from showing random protein images
    if not image_urls and (pdb_ids or uniprot_ids):
        fallback = next((d.metadata.get("image_path") for d in retrieved_docs if d.metadata.get("image_path")), None)
        if fallback:
            image_urls.append(fallback)

    # Only include RCSB/UniProt as a source if tool_results were actually fetched (live API was called)
    # Do NOT include it just because a retrieved doc happened to mention PDB data
    if tool_results:
        # all_sources already has "Global Protein Data Banks (RCSB/UniProt)" from live fetches
        final_sources = list(set(all_sources))
    else:
        # Source = only the internal docs that contributed context, strip PDB tags
        final_sources = sorted(set(
            os.path.basename(d.metadata.get("source", "Internal Knowledge Base"))
            for d in filtered_chunks
            if d.metadata.get("source")
        ))
        # Remove any accidental RCSB/UniProt tags that leaked in from doc metadata
        final_sources = [s for s in final_sources if "rcsb" not in s.lower() and "uniprot" not in s.lower()]

    if not final_sources:
        final_sources = ["Internal Scientific Knowledge Base"]

    # 6. Final check for visual intent — clear image list if not requested
    visual_keywords = ["structure", "diagram", "image", "picture", "show me", "visual", "draw", "render", "struture", "sturcture", "sitructure", "strcture"]
    wants_visual = any(k in question.lower() for k in visual_keywords)
    if any(k in question.lower() for k in ["sequence", "sequnce"]) and not any(k in question.lower() for k in ["structure", "diagram", "image", "strcture"]):
        wants_visual = False
    
    # If the question explicitly contains a PDB ID or UniProt ID, always preserve the image
    # so the user can visualize it along with the metadata.
    has_specific_proteins = bool(pdb_ids or uniprot_ids)
    if not wants_visual and not has_specific_proteins:
        image_urls = []

    # 7. Provenance Determination
    # Honest labeling: if no filtered_chunks contributed, the LLM answered from training data
    print(f"DEBUG: tool_results={len(tool_results)}, filtered_chunks={len(filtered_chunks)}")
    
    if tool_results and not filtered_chunks:
        provenance = "Live API (Real-time Synthesis)"
    elif tool_results and filtered_chunks:
        provenance = "Hybrid"
    elif filtered_chunks:
        # Verify if the question keywords actually overlap with the retrieved chunks.
        # If the user asks a general science query (e.g. "what is insulin") and we only
        # retrieve company docs, the query subject won't overlap with the documents.
        import re
        stop_words = {"what", "how", "why", "who", "where", "when", "which", "show", "tell", "explain", "about", "does", "mean", "stands", "stand", "the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for", "and", "or"}
        question_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', question) if w.lower() not in stop_words]
        
        chunks_text = " ".join([c.page_content.lower() for c in filtered_chunks])
        has_overlap = any(word in chunks_text for word in question_words) if question_words else True
        
        if has_overlap:
            provenance = "Vector DB"
        else:
            provenance = "LLM General Knowledge"
            final_sources = []
    else:
        # LLM answered but no context was provided — it used its own training knowledge
        provenance = "LLM General Knowledge"

    return {
        "answer": answer,
        "sources": sorted(final_sources),
        "image_paths": image_urls,      # list of all protein images
        "provenance_type": provenance
    }


# ==============================================================================
# SINGLETON: initialize once at module level (shared by API workers)
# ==============================================================================

_vector_store = None
_retriever = None

def get_rag_chain(provider_id=None, model_id=None):
    """Return a RAG chain for the specified provider/model."""
    global _vector_store, _retriever
    
    # 1. Initialize Vector Store (shared singleton)
    if _vector_store is None:
        print(" Loading RAG knowledge base...")
        embeddings = init_embeddings()
        _vector_store = get_vector_store(embeddings)
        _retriever = _vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # 2. Initialize LLM (dynamic per request)
    llm, provider_info = init_llm(provider_id, model_id)
    
    # 3. Build a fresh chain with the requested LLM
    chain, _ = build_rag_chain(_vector_store, llm)
    
    return chain, _retriever, provider_info
