# Core
import os
import io
import re
import json
import urllib.parse
from datetime import datetime

# Numerical & Datascience
import requests
import pandas as pd

# 1. Define the UniProt API endpoint and query parameters
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/stream"
BASE_QUERY = "(reviewed:true) AND (length:[30 TO 150])"

FAMILIES = {
    "Ribosomal": 'family:"ribosomal protein"',
    "Cytochrome": 'family:"cytochrome c"',
    "Ferredoxin": 'family:"ferredoxin"',
    "Thioredoxin": 'family:"thioredoxin"',
    "Histone": 'family:"histone"',
    "Ubiquitin": 'family:"ubiquitin"',
    "Cystatin": 'family:"cystatin"'
}

FIELDS = "accession,protein_name,sequence,length,mass,organism_name,lineage,xref_pfam,go,cc_subcellular_location,xref_pdb"
df_list = []

print("Initiating UniProt REST API queries...")

# 2. Fetch the data for each family
for fam_name, fam_query in FAMILIES.items():
    print(f"Fetching {fam_name} proteins...")
    
    full_query = f"{BASE_QUERY} AND ({fam_query})"
    encoded_query = urllib.parse.quote(full_query)
    request_url = f"{UNIPROT_URL}?query={encoded_query}&format=tsv&fields={FIELDS}"
    
    response = requests.get(request_url)
    response.raise_for_status()
    
    df_fam = pd.read_csv(io.StringIO(response.text), sep="\t")
    df_fam["Protein_Family"] = fam_name
    df_list.append(df_fam)

# Combine all fetched data
df_proteins = pd.concat(df_list, ignore_index=True)

# 3. Defensive Column Standardization
df_proteins.columns = (
    df_proteins.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# 4. Taxonomic Lineage Parsing
def extract_domain(lineage_string):
    if not isinstance(lineage_string, str):
        return ""
    if "Archaea" in lineage_string:
        return "Archaea"
    elif "Bacteria" in lineage_string:
        return "Bacteria"
    elif "Eukaryota" in lineage_string:
        return "Eukarya"
    return ""

if 'taxonomic_lineage' in df_proteins.columns:
    df_proteins["taxonomic_domain"] = df_proteins["taxonomic_lineage"].apply(extract_domain)
    
df_proteins = df_proteins[df_proteins["taxonomic_domain"].isin(["Archaea", "Bacteria", "Eukarya"])]

# 5. Metadata Extraction and Column Cleaning
go_mapping = {}
eco_mapping = {}

def process_go(text):
    if pd.isna(text) or text == "": 
        return ""
    matches = re.findall(r'(.*?)\s*\[GO:(\d+)\]', str(text))
    ids = []
    for desc, go_id in matches:
        clean_desc = desc.strip(' ;')
        go_mapping[go_id] = clean_desc
        ids.append(go_id)
    return ";".join(ids)

def process_subcell(text):
    if pd.isna(text) or text == "": 
        return ""
    eco_ids = re.findall(r'ECO:(\d+)', str(text))
    mapping_matches = re.findall(r'([^.]+?)\s*\{[^}]*ECO:(\d+)', str(text))
    for loc_desc, eco_id in mapping_matches:
        clean_loc = loc_desc.replace("SUBCELLULAR LOCATION:", "").strip()
        eco_mapping[eco_id] = clean_loc
    return ";".join(eco_ids)

def process_pfam(text):
    if pd.isna(text) or text == "": 
        return ""
    parts = [p.strip() for p in str(text).split(';') if p.strip()]
    return ";".join(parts)

# Apply processing to extract IDs and build metadata maps
if 'gene_ontology_(go)' in df_proteins.columns:
    df_proteins['go_id'] = df_proteins['gene_ontology_(go)'].apply(process_go)
if 'subcellular_location_[cc]' in df_proteins.columns:
    df_proteins['eco_id'] = df_proteins['subcellular_location_[cc]'].apply(process_subcell)
    
pfam_raw_col = 'pfam' if 'pfam' in df_proteins.columns else 'cross_reference_(pfam)'
if pfam_raw_col in df_proteins.columns:
    df_proteins['pfam_id'] = df_proteins[pfam_raw_col].apply(process_pfam)
    
pdb_raw_col = 'pdb' if 'pdb' in df_proteins.columns else 'cross_reference_(pdb)'
if pdb_raw_col in df_proteins.columns:
    df_proteins['pdb_id'] = df_proteins[pdb_raw_col].apply(process_pfam)

# Fix the raw UniProt naming conventions before final selection
rename_map = {
    "entry": "accession",
    "protein_names": "protein_name",
    "organism": "organism_name"
}
df_proteins = df_proteins.rename(columns=rename_map)

# Select final columns safely
target_columns = [
    "accession", "protein_name", "sequence", "length", "mass", "organism_name", 
    "taxonomic_domain", "protein_family", "pfam_id", "go_id", "eco_id", "pdb_id"
]

final_cols = [col for col in target_columns if col in df_proteins.columns]
df_final = df_proteins[final_cols].copy()

# Final safety sweep: replace all NaN and "Unknown" with blank strings
df_final = df_final.fillna("").replace("Unknown", "")

# 6. Export the Parquet DataFrame
OUTPUT_FILE = "proteins_tax_pfam_enriched.parquet"
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)
df_final.to_parquet(OUTPUT_FILE, index=False)

# 7. Export the Metadata JSON
metadata = {
    "description": "Curated dataset of diverse protein families from the UniProtKB (Swiss-Prot) database. Designed for Exploratory Data Analysis of high-dimensional protein language model latent spaces, capturing evolutionary taxonomy, structural motifs, and functional GO terms.",
    "date_accessed": datetime.now().isoformat(),
    "source": "UniProt Swiss-Prot",
    "query": BASE_QUERY,
    "go_mapping": go_mapping,
    "eco_mapping": eco_mapping
}

META_FILE = "proteins_metadata.json"
if os.path.exists(META_FILE):
    os.remove(META_FILE)
with open(META_FILE, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"\nSuccess! Enriched Dataset constructed with shape: {df_final.shape}")
print(f"Data saved to: {OUTPUT_FILE}")
print(f"Metadata saved to: {META_FILE}")
print(f"Columns ready: {df_final.columns.tolist()}")

