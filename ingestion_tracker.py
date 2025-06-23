import os
import json
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger("ingestion_tracker")

TRACKER_FILE = "ingestion_tracker.json"

def load_ingestion_tracker() -> Dict:
    """Load the ingestion tracker JSON file."""
    try:
        if os.path.exists(TRACKER_FILE):
            with open(TRACKER_FILE, 'r') as f:
                return json.load(f)
        else:
            # Create new tracker file with default structure
            default_tracker = {
                "ingested_files": {
                    "ICICI_2023-24.pdf": {
                        "COLLECTION_NAME": "AR_EMBEDDINGS",
                        "COMPANY_NAME": "ICICI",
                        "VECTOR_NAME": "icici_pagewise_embedding"
                    },
                    "ASI_2023-24.pdf": {
                        "COLLECTION_NAME": "ASI_AR_EMBEDDINGS",
                        "COMPANY_NAME": "ASI",
                        "VECTOR_NAME": "asi_pagewise_embedding"
                    },
                    "ZOMATO_2023-24.pdf": {
                        "COLLECTION_NAME": "ZOMATO_AR_EMBEDDINGS",
                        "COMPANY_NAME": "ZOMATO",
                        "VECTOR_NAME": "zomato_pagewise_embedding"
                    },
                    "COAL INDIA_2023-24.pdf": {
                        "COLLECTION_NAME": "COAL INDIA_AR_EMBEDDINGS",
                        "COMPANY_NAME": "COAL INDIA",
                        "VECTOR_NAME": "coal india_pagewise_embedding"
                    }
                }
            }
            save_ingestion_tracker(default_tracker)
            return default_tracker
    except Exception as e:
        logger.error(f"Failed to load ingestion tracker: {e}")
        return {"ingested_files": {}}

def save_ingestion_tracker(tracker_data: Dict):
    """Save the ingestion tracker JSON file."""
    try:
        with open(TRACKER_FILE, 'w') as f:
            json.dump(tracker_data, f, indent=2)
        logger.info(f"Saved ingestion tracker to {TRACKER_FILE}")
    except Exception as e:
        logger.error(f"Failed to save ingestion tracker: {e}")

def is_file_ingested(filename: str) -> bool:
    """Check if a file has already been ingested."""
    tracker = load_ingestion_tracker()
    return filename in tracker.get("ingested_files", {})

def get_ingestion_info(filename: str) -> Optional[Dict]:
    """Get ingestion information for a specific file."""
    tracker = load_ingestion_tracker()
    return tracker.get("ingested_files", {}).get(filename)

def add_ingested_file(filename: str, collection_name: str, company_name: str, vector_name: str):
    """Add a newly ingested file to the tracker."""
    tracker = load_ingestion_tracker()
    tracker["ingested_files"][filename] = {
        "COLLECTION_NAME": collection_name,
        "COMPANY_NAME": company_name,
        "VECTOR_NAME": vector_name
    }
    save_ingestion_tracker(tracker)
    logger.info(f"Added {filename} to ingestion tracker")

def extract_company_and_fy_from_pdf_path(pdf_path: str) -> Tuple[str, str]:
    """Extract company name and financial year from PDF filename."""
    filename = os.path.basename(pdf_path)
    name_without_ext = filename.replace(".pdf", "").replace(".PDF", "")
    parts = name_without_ext.split("_")

    if len(parts) >= 2:
        company_name = parts[0].upper()
        financial_year = parts[1]
        return company_name, financial_year
    else:
        return "COMPANY", "FY2024"

def get_collection_name(pdf_path: str) -> str:
    """Get collection name based on PDF filename."""
    filename = os.path.basename(pdf_path)
    name_without_ext = filename.replace(".pdf", "").replace(".PDF", "")
    
    # Special case for ICICI_2023-24.pdf
    if name_without_ext == "ICICI_2023-24":
        return "AR_EMBEDDINGS"
    else:
        # For all other files, use the filename as collection name
        return f"{name_without_ext}_AR_EMBEDDINGS" 