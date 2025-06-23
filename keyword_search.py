import os
import json
import logging
import re
from typing import List, Dict, Any, Set, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# Import ingestion tracker
from ingestion_tracker import (
    is_file_ingested,
    get_ingestion_info,
    extract_company_and_fy_from_pdf_path,
    get_collection_name,
)

# Import configuration
from config import QDRANT_URL, QDRANT_API_KEY, LOG_LEVEL

# ---------------------------------------------------------------------------
# ENV & CONSTANTS
# ---------------------------------------------------------------------------
load_dotenv()

# Get PDF path from environment or use default
PDF_PATH: str = os.getenv(
    "PDF_PATH", r"C:\Users\himan\Downloads\Documents\ICICI_2023-24.pdf"
)

# Extract company and FY from PDF path
COMPANY_NAME, FINANCIAL_YEAR = extract_company_and_fy_from_pdf_path(PDF_PATH)

# Get collection name - check if file is already ingested first
filename = os.path.basename(PDF_PATH)
if is_file_ingested(filename):
    ingestion_info = get_ingestion_info(filename)
    COLLECTION_NAME = ingestion_info["COLLECTION_NAME"]
    COMPANY_NAME = ingestion_info["COMPANY_NAME"]
    logger = logging.getLogger("keyword_search")
    logger.info(
        f"Using existing ingestion info for {filename}: Collection={COLLECTION_NAME}, Company={COMPANY_NAME}"
    )
else:
    # Use default collection naming logic for new files
    COLLECTION_NAME = get_collection_name(PDF_PATH)

# Exact phrases to search for
FINANCIAL_TABLE_PHRASES = [
    "Standalone Balance Sheet",
    "Balance Sheet",
    "Standalone Statement of Balance Sheet",
    "Statement of Balance Sheet",
    "Balance Sheet - Standalone",
    "Standalone Profit and Loss Account",
    "Profit and Loss Account",
    "Profit and Loss",
    "Profit and Loss - Standalone",
    "Profit and Loss Account",
    "Statement of Profit and Loss",
    "Standalone Statement of Profit and Loss",
    "Statement of Profit and Loss",
    "Profit and Loss - Standalone",
    "Statement of Profit and Loss - Standalone",
    "Standalone Cash Flow Statement",
    "Cash Flow Statement",
    "Standalone Statement of Cash Flow",
    "Statement of Cash Flow",
    "Statement of Cash Flow - Standalone",
    "Cash Flow - Standalone",
    "Standalone Statement of Cash Flows",
    "Statement of Cash Flows",
    "Statement of Cash Flow",
    "Cash Flow - Consolidated",
    "Consolidated Balance Sheet",
    "Consolidated Profit and Loss Account",
    "Consolidated Statement of Profit and Loss",
    "Profit and Loss - Consolidated",
    "Consolidated Cash Flow Statement",
    "Consolidated Statement of Cash Flows",
    "Consolidated Statement of Cash Flow",
    "Cash Flow - Consolidated",
    "Consolidated Statement of Profit and Loss",
    "Consolidated Statement of Profit and Loss Account",
    "Consolidated Statement of Cash Flows",
    "Consolidated Statement of Cash Flow",
    "Consolidated Statement of Profit and Loss",
    "Consolidated Statement of Profit and Loss Account",
    "Statement of Consolidated Profit and Loss",
    "Statement of Consolidated Cash Flow",
    "Statement of Consolidated Cash Flows",
]

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
)
logger = logging.getLogger("keyword_search")

# ---------------------------------------------------------------------------
# QDRANT CLIENT
# ---------------------------------------------------------------------------
if not QDRANT_URL:
    raise ValueError("QDRANT_URL must be set in .env file")

_qdrant_kwargs: Dict[str, Any] = {"url": QDRANT_URL, "prefer_grpc": False}
if QDRANT_API_KEY:
    _qdrant_kwargs["api_key"] = QDRANT_API_KEY

qdrant_client = QdrantClient(**_qdrant_kwargs, check_compatibility=False)
logger.info("Initialised Qdrant client with URL: %s", QDRANT_URL)

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for better keyword matching."""
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    return text


def extract_table_info(summary: str, pdf_filename: str) -> str:
    """
    Extract the appropriate part of the summary based on PDF filename.
    For ICICI_2023-34.pdf: use after first full stop to last
    For all other files: use before first full stop (main heading part)
    """
    # Get just the filename without extension
    filename = os.path.basename(pdf_filename).replace(".pdf", "").replace(".PDF", "")

    # Special case for ICICI_2023-34.pdf
    if filename == "ICICI_2023-34":
        # Find the first full stop
        first_period = summary.find(".")
        if first_period != -1:
            # Return everything after the first full stop (table description part)
            return summary[first_period + 1 :].strip()
        # If no full stop found, return the whole summary
        return summary
    else:
        # For all other files: use before first full stop (main heading part)
        first_period = summary.find(".")
        if first_period != -1:
            # Return everything before the first full stop (main heading part)
            return summary[:first_period].strip()
        # If no full stop found, return the whole summary
        return summary


def phrase_search_in_text(text: str, phrase: str) -> Tuple[bool, int]:
    """
    Check if a phrase exists in the text (case-insensitive).
    Returns (found, position) where position is the index where phrase was found (-1 if not found).
    """
    # Convert both text and phrase to lowercase for case-insensitive search
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)

    position = normalized_text.find(normalized_phrase)
    found = position != -1

    return found, position


def get_all_points_from_collection() -> List[qm.Record]:
    """Retrieve all points from the collection."""
    all_points = []
    offset = None
    batch_size = 100

    while True:
        result = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # We don't need vectors for keyword search
        )

        points, next_offset = result
        all_points.extend(points)

        if next_offset is None:
            break
        offset = next_offset

    return all_points


def search_phrases_in_collection(phrases: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Search for exact phrases in all summaries in the collection.
    Returns results grouped by phrase.
    """
    logger.info(f"Searching for {len(phrases)} phrases in collection")

    # Get all points from collection
    all_points = get_all_points_from_collection()
    logger.info(f"Retrieved {len(all_points)} points from collection")

    results = {phrase: [] for phrase in phrases}

    for point in all_points:
        payload = point.payload
        summary = payload.get("summary", "")

        if not summary:
            continue

        # Extract appropriate part based on PDF filename
        search_text = extract_table_info(summary, PDF_PATH)

        # Check each phrase in the extracted text
        for phrase in phrases:
            found, position = phrase_search_in_text(search_text, phrase)
            if found:
                results[phrase].append(
                    {
                        "page_num": payload.get("page_num"),
                        "position": position,
                        "summary": summary,
                        "search_text": search_text,  # Add the extracted search text
                        "company_name": payload.get("company_name", ""),
                        "phrase": phrase,
                    }
                )

    # Sort each phrase's results by position (earlier in text = higher priority)
    for phrase in results:
        results[phrase].sort(key=lambda x: x["position"])

    return results


def main():
    """Main function to search for financial table phrases."""
    logger.info("Starting financial table phrase search...")
    logger.info(f"Using collection: {COLLECTION_NAME}")
    logger.info(f"PDF path: {PDF_PATH}")

    # Check if collection exists
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        logger.error(f"Collection '{COLLECTION_NAME}' does not exist!")
        return

    # Check if PDF file exists
    if not os.path.exists(PDF_PATH):
        logger.error(f"PDF file not found: {PDF_PATH}")
        return

    # Get collection info
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    logger.info(
        f"Collection '{COLLECTION_NAME}' has {collection_info.points_count} points"
    )

    # Search for all phrases
    results = search_phrases_in_collection(FINANCIAL_TABLE_PHRASES)

    # Display results
    logger.info(f"\n{'='*60}")
    logger.info("PHRASE SEARCH RESULTS")
    logger.info(f"{'='*60}")

    all_results = []

    for phrase, matches in results.items():
        if matches:
            logger.info(f"\n'{phrase}' found in {len(matches)} pages:")
            for i, match in enumerate(matches[:3], 1):  # Show top 3 for each phrase
                logger.info(
                    f"  {i}. Page {match['page_num']} (position: {match['position']})"
                )
                logger.info(f"     Search text: {match['search_text'][:100]}...")

            # Add to all results
            for match in matches:
                match["searched_phrase"] = phrase
                all_results.append(match)
        else:
            logger.info(f"\n'{phrase}' - No matches found")

    # Summary of all results
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY OF ALL MATCHES")
    logger.info(f"{'='*60}")

    # Group by page number
    page_results = {}
    for result in all_results:
        page_num = result["page_num"]
        if page_num not in page_results:
            page_results[page_num] = []
        page_results[page_num].append(result["searched_phrase"])

    # Display pages and what was found on each
    for page_num, phrases_found in sorted(page_results.items()):
        logger.info(f"Page {page_num}: {', '.join(phrases_found)}")

    # Save detailed results to JSON file
    output_data = {
        "pdf_path": PDF_PATH,
        "collection_name": COLLECTION_NAME,
        "search_phrases": FINANCIAL_TABLE_PHRASES,
        "results_by_phrase": results,
        "pages_summary": {page: phrases for page, phrases in page_results.items()},
    }

    output_file = f"{COMPANY_NAME}_financial_tables_phrase_search_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"\nDetailed results saved to: {output_file}")

    # Create a simple summary file with just page numbers
    summary_output = {
        "pdf_path": PDF_PATH,
        "collection_name": COLLECTION_NAME,
        "page_numbers": {
            phrase: [match["page_num"] for match in matches]
            for phrase, matches in results.items()
        },
    }

    summary_file = f"{COMPANY_NAME}_phrase_search_page_numbers.json"
    with open(summary_file, "w") as f:
        json.dump(summary_output, f, indent=2)
    logger.info(f"Page numbers summary saved to: {summary_file}")

    # Display next steps
    logger.info(f"\n{'='*60}")
    logger.info("NEXT STEPS")
    logger.info(f"{'='*60}")
    logger.info(
        f"1. Use the page numbers from the results to extract tables from: {PDF_PATH}"
    )
    logger.info("2. The page numbers are saved in 'phrase_search_page_numbers.json'")
    logger.info(
        "3. You can now run your table extraction algorithm on the identified pages"
    )


if __name__ == "__main__":
    main()
