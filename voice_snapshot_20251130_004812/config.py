from pathlib import Path
import os

# Root of your offline data on the T7
OFFLINE_ROOT = Path(os.environ.get("OFFLINE_AI_ROOT", "/T7_OFFLINE_AI")).resolve()

# Data layout
WIKI_DIR = OFFLINE_ROOT / "01_wiki_zims"
SCIENCE_MED_DIR = OFFLINE_ROOT / "02_science_med"
DEV_DOCS_DIR = OFFLINE_ROOT / "03_dev_docs"
MAPS_DIR = OFFLINE_ROOT / "04_maps_osm"
PSYCHOACTIVE_DIR = OFFLINE_ROOT / "05_psychoactive"
INDEX_DIR = OFFLINE_ROOT / "06_indexes"

# Additional corpora
NOTES_DIR = OFFLINE_ROOT / "09_notes"
GOLDEN_SNIPPETS_DIR = OFFLINE_ROOT / "10_golden_snippets"

# Raw and index locations
RAW_CORPUS_DIR = INDEX_DIR / "raw"
RAW_NOTES_DIR = RAW_CORPUS_DIR / "notes"
RAW_GOLDEN_DIR = RAW_CORPUS_DIR / "golden"
FAISS_STORE_DIR = INDEX_DIR / "faiss"

# Embeddings
EMBEDDING_MODEL_NAME = os.environ.get(
    "OFFLINE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS index filenames
FAISS_INDEX_FILE = FAISS_STORE_DIR / "corpus.index"
FAISS_META_FILE = FAISS_STORE_DIR / "corpus_meta.jsonl"
FAISS_NOTES_INDEX_FILE = FAISS_STORE_DIR / "notes.index"
FAISS_NOTES_META_FILE = FAISS_STORE_DIR / "notes_meta.jsonl"
FAISS_GOLDEN_INDEX_FILE = FAISS_STORE_DIR / "golden.index"
FAISS_GOLDEN_META_FILE = FAISS_STORE_DIR / "golden_meta.jsonl"

# GraphHopper routing server
GRAPHHOPPER_URL = os.environ.get("GRAPHHOPPER_URL", "http://localhost:8989")

# Mode toggle
SHAI_MODE = os.environ.get("SHAI_MODE", "normal")  # normal | emergency
