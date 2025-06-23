import os
import json
import uuid
import logging
import concurrent.futures as cf
from typing import List, Tuple, Dict, Any, Optional

import PyPDF2
import pdfplumber
import pandas as pd
import requests
import boto3
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import fitz  # PyMuPDF for page analysis
from PIL import Image
import io

# Import ingestion tracker
from ingestion_tracker import (
    is_file_ingested, 
    get_ingestion_info, 
    add_ingested_file,
    extract_company_and_fy_from_pdf_path,
    get_collection_name
)

# ---------------------------------------------------------------------------
# ENV & CONSTANTS
# ---------------------------------------------------------------------------
load_dotenv()

PDF_PATH: str = os.getenv(
    "PDF_PATH", r"C:\Users\himan\Downloads\Documents\ICICI_2023-24.pdf"
)

# Extract company and FY from PDF path
COMPANY_NAME, FINANCIAL_YEAR = extract_company_and_fy_from_pdf_path(PDF_PATH)

# Get collection name and vector name
COLLECTION_NAME = get_collection_name(PDF_PATH)
VECTOR_NAME: str = f"{COMPANY_NAME.lower()}_pagewise_embedding"

SPARSE_EMBEDDING_URL: str = os.getenv(
    "SPARSE_EMBEDDING_URL", "http://52.7.81.94:8010/embed"
)
QDRANT_URL: str = os.getenv("QDRANT_URL")
QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

AWS_REGION_NAME: str = os.getenv("AWS_REGION_NAME", "us-east-1")

# ---------- Bedrock Claude ----------
CLAUDE_MODEL_ID: str = os.getenv(
    "CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
)
CLAUDE_MAX_TOKENS: int = 1024  # per user request
ANTHROPIC_VERSION: str = os.getenv("ANTHROPIC_VERSION", "bedrock-2023-05-31")

# ---------- Processing Configuration ----------
PAGES_PER_BATCH: int = 5  # Process 5 pages at a time
MAX_WORKERS: int = 5  # Number of parallel workers

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("ingest")
logger.info("Logger initialised at level %s", LOG_LEVEL)

# ---------------------------------------------------------------------------
# QDRANT SETUP (sparse‑only collection)
# ---------------------------------------------------------------------------
if not QDRANT_URL:
    raise ValueError("QDRANT_URL must be set in .env file")

_qdrant_kwargs: Dict[str, Any] = {"url": QDRANT_URL, "prefer_grpc": False}
if QDRANT_API_KEY:
    _qdrant_kwargs["api_key"] = QDRANT_API_KEY

qdrant_client = QdrantClient(**_qdrant_kwargs, check_compatibility=False)
logger.info("Initialised Qdrant client with URL: %s", QDRANT_URL)

sparse_cfg = {VECTOR_NAME: qm.SparseVectorParams(index=qm.SparseIndexParams())}

if not qdrant_client.collection_exists(COLLECTION_NAME):
    logger.info("Creating sparse‑only collection %s", COLLECTION_NAME)
    qdrant_client.create_collection(
        COLLECTION_NAME,
        vectors_config={},  # no dense spaces
        sparse_vectors_config=sparse_cfg,
    )
else:
    # Ensure sparse space exists (idempotent)
    try:
        qdrant_client.update_collection(
            COLLECTION_NAME, sparse_vectors_config=sparse_cfg
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            logger.warning("Collection update warning: %s", e)

# ---------------------------------------------------------------------------
# BEDROCK RUNTIME CLIENT
# ---------------------------------------------------------------------------
try:
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION_NAME)
    logger.info("Bedrock client initialised in %s", AWS_REGION_NAME)
except Exception as e:
    logger.error("Failed to init Bedrock client: %s", e)
    raise

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def detect_page_orientation(pdf_path: str, page_idx: int) -> Tuple[str, float, float]:
    """Detect if a page is portrait or landscape and return dimensions."""
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_idx]

        # Get page dimensions
        rect = page.rect
        width = rect.width
        height = rect.height

        # Determine orientation
        if width > height:
            orientation = "landscape"
        else:
            orientation = "portrait"

        pdf_document.close()
        return orientation, width, height

    except Exception as e:
        logger.error(f"Failed to detect orientation for page {page_idx + 1}: {e}")
        return "portrait", 0, 0


def split_landscape_page_to_images(
    pdf_path: str, page_idx: int
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Split a landscape page into left and right images."""
    try:
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[page_idx]

        # Get page dimensions
        rect = page.rect
        width = rect.width
        height = rect.height

        # Create transformation matrix for high DPI
        mat = fitz.Matrix(2, 2)  # 2x zoom for better quality

        # Render the full page
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")

        # Convert to PIL Image
        full_image = Image.open(io.BytesIO(img_data))

        # Split into left and right halves
        img_width, img_height = full_image.size
        left_half = full_image.crop((0, 0, img_width // 2, img_height))
        right_half = full_image.crop((img_width // 2, 0, img_width, img_height))

        pdf_document.close()
        return left_half, right_half

    except Exception as e:
        logger.error(f"Failed to split landscape page {page_idx + 1}: {e}")
        return None, None


def extract_text_from_image_ocr(image: Image.Image) -> str:
    """Extract text from image using OCR service."""
    try:
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Send to OCR service
        ocr_url = os.getenv("OCR_SERVICE_URL", "http://52.7.81.94:8000/ocr_image")

        files = {"file": ("image.png", img_byte_arr, "image/png")}
        headers = {"accept": "application/json"}

        response = requests.post(ocr_url, files=files, headers=headers, timeout=60)

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                return result.get("markdown", "")
            else:
                logger.error(f"OCR failed: {result}")
                return ""
        else:
            logger.error(f"OCR service error: {response.status_code}")
            return ""

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return ""


def extract_page_to_markdown(pdf_path: str, page_idx: int) -> List[Tuple[int, str]]:
    """Extract text and tables from a PDF page and convert to markdown. Returns list of (page_num, markdown) tuples."""
    page_num = page_idx + 1
    results = []

    try:
        # Detect page orientation
        orientation, width, height = detect_page_orientation(pdf_path, page_idx)

        if orientation == "landscape":
            logger.info(
                f"Page {page_num} is landscape ({width:.1f}x{height:.1f}), splitting into two pages"
            )

            # Split landscape page into left and right images
            left_image, right_image = split_landscape_page_to_images(pdf_path, page_idx)

            if left_image and right_image:
                # Extract text from left half using OCR
                left_markdown = extract_text_from_image_ocr(left_image)
                if left_markdown:
                    results.append((page_num, left_markdown))
                    logger.info(
                        f"Extracted left half of page {page_num} using OCR ({len(left_markdown)} chars)"
                    )

                # Extract text from right half using OCR
                right_markdown = extract_text_from_image_ocr(right_image)
                if right_markdown:
                    results.append(
                        (page_num + 0.5, right_markdown)
                    )  # Use decimal to indicate right half
                    logger.info(
                        f"Extracted right half of page {page_num} using OCR ({len(right_markdown)} chars)"
                    )

        else:
            # Portrait page - use existing method
            logger.info(
                f"Page {page_num} is portrait ({width:.1f}x{height:.1f}), using standard extraction"
            )

            markdown_content = ""

            # Extract text using PyPDF2
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if page_idx < len(pdf_reader.pages):
                    page = pdf_reader.pages[page_idx]
                    text = page.extract_text()
                    if text:
                        markdown_content = text

            # Extract tables using pdfplumber (PyPDF2 doesn't handle tables well)
            with pdfplumber.open(pdf_path) as pdf:
                if page_idx < len(pdf.pages):
                    page = pdf.pages[page_idx]
                    tables = page.extract_tables()

                    markdown_tables = []
                    for table in tables:
                        if not table:
                            continue
                        # Convert table to markdown
                        markdown_table = []
                        for row in table:
                            cleaned_row = [
                                cell if cell is not None else "" for cell in row
                            ]
                            markdown_table.append("| " + " | ".join(cleaned_row) + " |")
                        # Add header separator
                        if markdown_table:
                            header = markdown_table[0]
                            separator = (
                                "| "
                                + " | ".join(["---"] * (header.count("|") - 1))
                                + " |"
                            )
                            markdown_table.insert(1, separator)
                            markdown_tables.append("\n".join(markdown_table))

                    # Append tables to markdown content
                    if markdown_tables:
                        markdown_content += "\n\n" + "\n\n".join(markdown_tables)

            if markdown_content:
                results.append((page_num, markdown_content))
                logger.info(
                    f"Extracted markdown for page {page_num} ({len(markdown_content)} chars)"
                )

        return results

    except Exception as e:
        logger.error(f"Failed to extract markdown for page {page_num}: {e}")
        return []


def generate_summary_with_claude(
    page_num: int, markdown: str
) -> Tuple[int, Optional[str]]:
    """Generate a summary using Claude based on the markdown content."""
    if not markdown:
        return page_num, None

    prompt = f"""
You are given markdown extracted from page {page_num} of an annual report.

Your task is to read the markdown content and respond in the following exact sentence structure:

Page {page_num} has main heading as HEADING. Page {page_num} has subheadings as SUBHEADING_1, SUBHEADING_2, ... Page {page_num} contains table(s) that has {{content}} (in few words).

Rules:
- If a main heading is present (typically in bold or top-level format), replace HEADING with the actual main heading text.
- If no main heading is found, write:
  Page {page_num} has no main heading.
- If no subheadings are present, say:
  Page {page_num} has no subheadings.
- If subheadings are present, list them separated by commas.
- Replace {{content}} with a few words describing what the table(s) is/are about.
- If no tables are present, say:
  Page {page_num} contains no tables.

Output only this one-line sentence. No extra explanation or formatting. If nothing relevant is found on the page, return: `null`.

Markdown:
{markdown}

"""

    body = {
        "anthropic_version": ANTHROPIC_VERSION,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": CLAUDE_MAX_TOKENS,
        "temperature": 0.0,
    }

    try:
        resp = bedrock_runtime.invoke_model(
            modelId=CLAUDE_MODEL_ID,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        raw = resp["body"].read().decode("utf-8").strip()
        result = json.loads(raw)

        # Extract the summary from Claude's response
        if "content" in result and len(result["content"]) > 0:
            summary = result["content"][0].get("text", "")
            if summary:
                logger.info(
                    f"Generated summary for page {page_num}: {summary[:100]}..."
                )
                return page_num, summary

        logger.warning(f"No summary generated for page {page_num}")
        return page_num, None

    except Exception as e:
        logger.error(f"Claude failure for page {page_num}: {e}")
        return page_num, None


def embed_sparse(
    page_num: int, text: str
) -> Tuple[int, Optional[List[int]], Optional[List[float]]]:
    """Generate sparse embeddings for the given text."""
    if not text:
        return page_num, None, None
    try:
        r = requests.post(SPARSE_EMBEDDING_URL, json={"text": text}, timeout=10)
        r.raise_for_status()
        d = r.json() or {}
        indices = [int(k) for k in d.keys()]
        values = [float(v) for v in d.values()]
        logger.debug(
            f"Generated sparse embedding for page {page_num}: {len(indices)} non-zero tokens"
        )
        return page_num, indices, values
    except Exception as e:
        logger.error(f"Sparse embed error for page {page_num}: {e}")
        return page_num, None, None


# ---------------------------------------------------------------------------
# MAIN INGEST
# ---------------------------------------------------------------------------


def ingest_pdf(pdf_path: str) -> bool:
    try:
        # Check if file is already ingested
        filename = os.path.basename(pdf_path)
        if is_file_ingested(filename):
            ingestion_info = get_ingestion_info(filename)
            logger.info(f"File {filename} is already ingested in collection '{ingestion_info['COLLECTION_NAME']}'")
            logger.info(f"Company: {ingestion_info['COMPANY_NAME']}, Vector: {ingestion_info['VECTOR_NAME']}")
            return True

        logger.info(f"File {filename} not found in ingestion tracker. Proceeding with ingestion...")

        # Get total number of pages using PyPDF2
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

        logger.info(f"PDF opened: {total_pages} pages found")

        # Process pages in batches of 5
        for batch_start in range(0, total_pages, PAGES_PER_BATCH):
            batch_end = min(batch_start + PAGES_PER_BATCH, total_pages)
            page_indices = list(range(batch_start, batch_end))

            logger.info(f"Processing batch: pages {batch_start + 1} to {batch_end}")

            # Stage 1: Extract markdown from all pages in parallel
            logger.info("Stage 1: Extracting markdown in parallel...")
            page_markdown_map = {}
            with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_page = {
                    executor.submit(extract_page_to_markdown, pdf_path, idx): idx
                    for idx in page_indices
                }

                for future in cf.as_completed(future_to_page):
                    try:
                        page_results = future.result()
                        # Handle multiple results from landscape pages
                        for page_num, markdown in page_results:
                            page_markdown_map[page_num] = markdown
                    except Exception as e:
                        page_idx = future_to_page[future]
                        logger.error(f"Error extracting page {page_idx + 1}: {e}")

            # Stage 2: Generate summaries in parallel
            logger.info("Stage 2: Generating summaries in parallel...")
            page_summary_map = {}
            with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_page = {
                    executor.submit(
                        generate_summary_with_claude, page_num, markdown
                    ): page_num
                    for page_num, markdown in page_markdown_map.items()
                }

                for future in cf.as_completed(future_to_page):
                    try:
                        page_num, summary = future.result()
                        if summary:
                            page_summary_map[page_num] = summary
                    except Exception as e:
                        page_num = future_to_page[future]
                        logger.error(
                            f"Error generating summary for page {page_num}: {e}"
                        )

            # Stage 3: Generate embeddings in parallel
            logger.info("Stage 3: Generating embeddings in parallel...")
            embedding_results = []
            with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_data = {
                    executor.submit(embed_sparse, page_num, summary): (
                        page_num,
                        summary,
                    )
                    for page_num, summary in page_summary_map.items()
                }

                for future in cf.as_completed(future_to_data):
                    try:
                        page_num, indices, values = future.result()
                        if indices is not None and values is not None:
                            page_num_from_data, summary = future_to_data[future]
                            markdown = page_markdown_map.get(page_num_from_data, "")
                            embedding_results.append(
                                (page_num, summary, markdown, indices, values)
                            )
                    except Exception as e:
                        page_num, _ = future_to_data[future]
                        logger.error(
                            f"Error generating embedding for page {page_num}: {e}"
                        )

            # Create points and upsert to Qdrant
            if embedding_results:
                points = []
                for page_num, summary, markdown, indices, values in embedding_results:
                    point = qm.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            VECTOR_NAME: qm.SparseVector(indices=indices, values=values)
                        },
                        payload={
                            "page_num": page_num,
                            "company_name": COMPANY_NAME,
                            "markdown": markdown,
                            "summary": summary,
                        },
                    )
                    points.append(point)

                try:
                    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                    logger.info(
                        f"Upserted {len(points)} points for pages {batch_start + 1}-{batch_end}"
                    )
                except Exception as e:
                    logger.error(f"Upsert error: {e}")

        # Update ingestion tracker after successful ingestion
        add_ingested_file(filename, COLLECTION_NAME, COMPANY_NAME, VECTOR_NAME)
        
        logger.info(
            "SUCCESS: Ingestion complete for %s. Stored sparse vectors in '%s'",
            COMPANY_NAME,
            COLLECTION_NAME,
        )
        return True

    except Exception as e:
        logger.exception("Fatal ingest error: %s", e)
        return False


if __name__ == "__main__":
    if not ingest_pdf(PDF_PATH):
        logger.error("Ingestion failed for PDF: %s", PDF_PATH)
        exit(1)
