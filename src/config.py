"""
config.py – Central configuration for the Trademarkia semantic-search system.

All tuneable hyper-parameters live here so that nothing is buried in
business-logic files.  Changing a value here propagates everywhere.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


BASE_DIR       = Path(__file__).resolve().parent.parent
DATA_DIR       = BASE_DIR / "data"
CHROMA_DIR     = DATA_DIR / "chroma_db"
CLUSTER_DIR    = DATA_DIR / "clusters"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_DIR.mkdir(parents=True, exist_ok=True)


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM   = 384


CHROMA_COLLECTION = "newsgroups"


MAX_CHARS_PER_DOC = 512


MIN_CHARS_PER_DOC = 50


N_CLUSTERS = 15


FCM_M = 2.0

FCM_MAX_ITER  = 150
FCM_TOL       = 1e-4

CLUSTER_SAMPLE = 5000


CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90"))

CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
