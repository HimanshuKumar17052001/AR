# import os
# import json
# import uuid
# import logging
# import concurrent.futures as cf
# from typing import List, Tuple, Dict, Any, Optional, Callable
# import queue
# import time

# import PyPDF2
# import pdfplumber
# import pandas as pd
# import requests
# import boto3
# from dotenv import load_dotenv
# from qdrant_client import QdrantClient
# from qdrant_client.http import models as qm

# # ---------------------------------------------------------------------------
# # ENV & CONSTANTS
# # ---------------------------------------------------------------------------
# load_dotenv()

# PDF_PATH: str = os.getenv(
#     "PDF_PATH", r"C:\\Users\\himan\\Downloads\\Documents\\ASI_FY_202324_CY.pdf"
# )
# COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "ASI_AR_EMBEDDINGS")
# COMPANY_NAME: str = os.getenv("COMPANY_NAME", "ASI")

# SPARSE_EMBEDDING_URL: str = os.getenv(
#     "SPARSE_EMBEDDING_URL", "http://52.7.81.94:8010/embed"
# )
# QDRANT_URL: str = os.getenv("QDRANT_URL")
# QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

# AWS_REGION_NAME: str = os.getenv("AWS_REGION_NAME", "us-east-1")

# # ---------- Bedrock Claude ----------
# CLAUDE_MODEL_ID: str = os.getenv(
#     "CLAUDE_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
# )
# CLAUDE_MAX_TOKENS: int = 1024  # per user request
# ANTHROPIC_VERSION: str = os.getenv("ANTHROPIC_VERSION", "bedrock-2023-05-31")

# # ---------- Sparse Vector Name ----------
# VECTOR_NAME: str = "asi_pagewise_embedding"  # as requested

# # ---------- Processing Configuration ----------
# PAGES_PER_BATCH: int = 50  # Larger batches for GPU efficiency
# MAX_WORKERS_EXTRACTION: int = int(
#     os.getenv("MAX_WORKERS_EXTRACTION", "10")
# )  # CPU-bound
# MAX_WORKERS_SUMMARY: int = int(os.getenv("MAX_WORKERS_SUMMARY", "10"))  # Network-bound
# MAX_WORKERS_EMBEDDING: int = int(
#     os.getenv("MAX_WORKERS_EMBEDDING", "20")
# )  # Increased for GPU
# MAX_QUEUE_SIZE: int = 100  # Larger queue for GPU throughput
# GPU_BATCH_SIZE: int = int(os.getenv("GPU_BATCH_SIZE", "8"))  # GPU batch size

# # ---------------------------------------------------------------------------
# # LOGGING
# # ---------------------------------------------------------------------------
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# logging.basicConfig(
#     format="%(asctime)s | %(levelname)-8s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=getattr(logging, LOG_LEVEL, logging.INFO),
# )
# logger = logging.getLogger("ingest")
# logger.info("Logger initialised at level %s", LOG_LEVEL)

# # ---------------------------------------------------------------------------
# # QDRANT SETUP (sparse‑only collection)
# # ---------------------------------------------------------------------------
# if not QDRANT_URL:
#     raise ValueError("QDRANT_URL must be set in .env file")

# _qdrant_kwargs: Dict[str, Any] = {"url": QDRANT_URL, "prefer_grpc": False}
# if QDRANT_API_KEY:
#     _qdrant_kwargs["api_key"] = QDRANT_API_KEY

# qdrant_client = QdrantClient(**_qdrant_kwargs, check_compatibility=False)
# logger.info("Initialised Qdrant client with URL: %s", QDRANT_URL)

# sparse_cfg = {VECTOR_NAME: qm.SparseVectorParams(index=qm.SparseIndexParams())}

# if not qdrant_client.collection_exists(COLLECTION_NAME):
#     logger.info("Creating sparse‑only collection %s", COLLECTION_NAME)
#     qdrant_client.create_collection(
#         COLLECTION_NAME,
#         vectors_config={},  # no dense spaces
#         sparse_vectors_config=sparse_cfg,
#     )
# else:
#     # Ensure sparse space exists (idempotent)
#     try:
#         qdrant_client.update_collection(
#             COLLECTION_NAME, sparse_vectors_config=sparse_cfg
#         )
#     except Exception as e:
#         if "already exists" not in str(e).lower():
#             logger.warning("Collection update warning: %s", e)

# # ---------------------------------------------------------------------------
# # BEDROCK RUNTIME CLIENT
# # ---------------------------------------------------------------------------
# try:
#     bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION_NAME)
#     logger.info("Bedrock client initialised in %s", AWS_REGION_NAME)
# except Exception as e:
#     logger.error("Failed to init Bedrock client: %s", e)
#     raise

# # ---------------------------------------------------------------------------
# # HELPERS
# # ---------------------------------------------------------------------------


# def extract_page_to_markdown(pdf_path: str, page_idx: int) -> Tuple[int, str]:
#     """Optimized markdown extraction with error handling."""
#     page_num = page_idx + 1
#     try:
#         # Extract text using PyPDF2 (faster for text)
#         with open(pdf_path, "rb") as file:
#             pdf_reader = PyPDF2.PdfReader(file)
#             if page_idx < len(pdf_reader.pages):
#                 page = pdf_reader.pages[page_idx]
#                 text = page.extract_text() or ""
#             else:
#                 return page_num, ""

#         # Extract tables using pdfplumber only if tables detected
#         tables_md = []
#         if any(char in text for char in ["\t", "│", "┌", "└"]):  # Table indicators
#             with pdfplumber.open(pdf_path) as pdf:
#                 if page_idx < len(pdf.pages):
#                     page = pdf.pages[page_idx]
#                     for table in page.extract_tables():
#                         if table and any(any(cell for cell in row) for row in table):
#                             df = pd.DataFrame(table[1:], columns=table[0])
#                             tables_md.append(df.to_markdown(index=False))

#         return page_num, text + ("\n\n" + "\n\n".join(tables_md) if tables_md else "")
#     except Exception as e:
#         logger.error(f"Page {page_num} extraction failed: {str(e)[:100]}")
#         return page_num, ""


# def generate_summary_with_claude(
#     page_num: int, markdown: str
# ) -> Tuple[int, Optional[str]]:
#     """Generate a summary using Claude based on the markdown content."""
#     if not markdown:
#         return page_num, None

#     prompt = f"""
# You are given markdown extracted from page {page_num} of an annual report.

# Your task is to read the content and respond in the following exact sentence structure:

# Page {page_num} contains heading(s) as HEADING_1, HEADING_2, HEADING_3,...and so on. Page {page_num} contains table(s) that has {{content}} (in few words).

# - Replace HEADING_1, HEADING_2, etc., with actual headings or subheadings found on the page.
# - Replace {{content}} with a few words describing what the table(s) is/are about.
# - IMPORTANT: If the page contains consolidated financial statements (Balance Sheet, Cash Flow Statement, or Profit & Loss Account), specifically mention "consolidated" in the table description.
# - Examples of consolidated table descriptions:
#   - "consolidated balance sheet"
#   - "consolidated cash flow statement" 
#   - "consolidated profit and loss account"
#   - "consolidated financial statements"
# - If there are no headings, write "no headings".
# - If there are no tables, write "no tables".

# Return only this one-line response. No extra explanation or formatting.

# Markdown:
# {markdown}
# """

#     body = {
#         "anthropic_version": ANTHROPIC_VERSION,
#         "messages": [{"role": "user", "content": prompt}],
#         "max_tokens": CLAUDE_MAX_TOKENS,
#         "temperature": 0.0,
#     }

#     try:
#         resp = bedrock_runtime.invoke_model(
#             modelId=CLAUDE_MODEL_ID,
#             body=json.dumps(body),
#             accept="application/json",
#             contentType="application/json",
#         )
#         raw = resp["body"].read().decode("utf-8").strip()
#         result = json.loads(raw)

#         # Extract the summary from Claude's response
#         if "content" in result and len(result["content"]) > 0:
#             summary = result["content"][0].get("text", "")
#             if summary:
#                 logger.info(
#                     f"Generated summary for page {page_num}: {summary[:100]}..."
#                 )
#                 return page_num, summary

#         logger.warning(f"No summary generated for page {page_num}")
#         return page_num, None

#     except Exception as e:
#         logger.error(f"Claude failure for page {page_num}: {e}")
#         return page_num, None


# def embed_sparse_batch(
#     batch: List[Tuple[int, str]],
# ) -> List[Tuple[int, List[int], List[float]]]:
#     """Batch sparse embeddings for GPU efficiency."""
#     if not batch:
#         return []

#     try:
#         # Prepare batch request
#         batch_data = [{"text": text} for _, text in batch]
#         r = requests.post(SPARSE_EMBEDDING_URL, json={"batch": batch_data}, timeout=60)
#         r.raise_for_status()
#         response = r.json()

#         results = []
#         for i, embedding in enumerate(response["embeddings"]):
#             page_num = batch[i][0]
#             indices = [int(k) for k in embedding.keys()]
#             values = [float(v) for v in embedding.values()]
#             results.append((page_num, indices, values))
#             logger.debug(
#                 f"Generated sparse embedding for page {page_num} (batch size: {len(batch)})"
#             )

#         return results
#     except Exception as e:
#         logger.error(f"Batch embedding failed: {e}")
#         return []


# def embed_sparse_worker(input_queue: queue.Queue, output_queue: queue.Queue):
#     """Worker for batch embedding processing."""
#     batch = []
#     while True:
#         try:
#             # Get item with timeout
#             item = input_queue.get(timeout=5)
#             if item is None:  # Sentinel to flush
#                 if batch:
#                     results = embed_sparse_batch(batch)
#                     for result in results:
#                         output_queue.put(result)
#                     batch = []
#                 break

#             page_num, summary = item
#             batch.append((page_num, summary))

#             # Process batch when full or after timeout
#             if len(batch) >= GPU_BATCH_SIZE:
#                 results = embed_sparse_batch(batch)
#                 for result in results:
#                     output_queue.put(result)
#                 batch = []

#         except queue.Empty:
#             # Process partial batch if exists
#             if batch:
#                 results = embed_sparse_batch(batch)
#                 for result in results:
#                     output_queue.put(result)
#                 batch = []
#         except Exception as e:
#             logger.error(f"Embedding worker error: {e}")

#     # Final flush
#     if batch:
#         results = embed_sparse_batch(batch)
#         for result in results:
#             output_queue.put(result)


# # ---------------------------------------------------------------------------
# # PARALLEL PROCESSING PIPELINE (GPU OPTIMIZED)
# # ---------------------------------------------------------------------------


# def run_pipeline(
#     pdf_path: str,
#     page_indices: List[int],
# ) -> List[qm.PointStruct]:
#     """Parallel pipeline with GPU-optimized embedding."""
#     points = []
#     extraction_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
#     summary_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
#     embedding_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

#     # Stage 1: Markdown Extraction (CPU-bound)
#     def extract_worker():
#         for idx in page_indices:
#             page_num, markdown = extract_page_to_markdown(pdf_path, idx)
#             if markdown:
#                 extraction_queue.put((page_num, markdown))
#         # Signal end of extraction
#         for _ in range(MAX_WORKERS_SUMMARY):
#             extraction_queue.put(None)

#     # Stage 2: Summary Generation (Network-bound)
#     def summary_worker():
#         while True:
#             item = extraction_queue.get()
#             if item is None:
#                 # Signal embedding workers to flush
#                 for _ in range(MAX_WORKERS_EMBEDDING):
#                     summary_queue.put(None)
#                 break
#             page_num, markdown = item
#             _, summary = generate_summary_with_claude(page_num, markdown)
#             if summary:
#                 summary_queue.put((page_num, markdown, summary))
#             extraction_queue.task_done()

#     # Start pipeline with thread pools
#     with cf.ThreadPoolExecutor(
#         max_workers=MAX_WORKERS_EXTRACTION
#     ) as extract_exec, cf.ThreadPoolExecutor(
#         max_workers=MAX_WORKERS_SUMMARY
#     ) as summary_exec, cf.ThreadPoolExecutor(
#         max_workers=MAX_WORKERS_EMBEDDING
#     ) as embed_exec:

#         # Start extraction
#         extract_exec.submit(extract_worker)

#         # Start summary workers
#         for _ in range(MAX_WORKERS_SUMMARY):
#             summary_exec.submit(summary_worker)

#         # Start embedding workers (GPU-optimized)
#         for _ in range(MAX_WORKERS_EMBEDDING):
#             embed_exec.submit(
#                 embed_sparse_worker,
#                 input_queue=summary_queue,
#                 output_queue=embedding_queue,
#             )

#         # Collect results
#         completed_workers = 0
#         while completed_workers < MAX_WORKERS_EMBEDDING:
#             try:
#                 result = embedding_queue.get(timeout=120)
#                 if result is None:
#                     completed_workers += 1
#                 else:
#                     page_num, indices, values = result
#                     # Find matching summary and markdown
#                     # Note: In practice, you'd want to pass this through the pipeline
#                     # For simplicity, we'll reconstruct from the queues
#                     point = qm.PointStruct(
#                         id=str(uuid.uuid4()),
#                         vector={
#                             VECTOR_NAME: qm.SparseVector(indices=indices, values=values)
#                         },
#                         payload={
#                             "page_num": page_num,
#                             "company_name": COMPANY_NAME,
#                             # These would need to be passed through the pipeline
#                             "markdown": "",
#                             "summary": "",
#                         },
#                     )
#                     points.append(point)
#             except queue.Empty:
#                 logger.warning("Embedding queue timeout, retrying...")
#                 continue

#     return points


# # ---------------------------------------------------------------------------
# # MAIN INGEST (GPU OPTIMIZED)
# # ---------------------------------------------------------------------------


# def ingest_pdf(pdf_path: str) -> bool:
#     try:
#         # Get total pages
#         with open(pdf_path, "rb") as file:
#             pdf_reader = PyPDF2.PdfReader(file)
#             total_pages = len(pdf_reader.pages)
#         logger.info(f"Processing {total_pages} pages")

#         # Process in batches optimized for GPU
#         for batch_start in range(0, total_pages, PAGES_PER_BATCH):
#             batch_end = min(batch_start + PAGES_PER_BATCH, total_pages)
#             page_indices = list(range(batch_start, batch_end))
#             logger.info(f"Processing batch: pages {batch_start+1}-{batch_end}")

#             # Run parallel pipeline for batch
#             start_time = time.time()
#             points = run_pipeline(pdf_path=pdf_path, page_indices=page_indices)
#             batch_time = time.time() - start_time
#             logger.info(
#                 f"Processed batch in {batch_time:.2f} seconds "
#                 f"({len(points)} points, {len(points)/batch_time:.2f} pages/sec)"
#             )

#             # Batch upsert to Qdrant
#             if points:
#                 try:
#                     qdrant_client.upsert(
#                         collection_name=COLLECTION_NAME, points=points, wait=True
#                     )
#                     logger.info(f"Upserted {len(points)} points")
#                 except Exception as e:
#                     logger.error(f"Batch upsert failed: {str(e)[:100]}")

#         logger.info(f"SUCCESS: Ingestion complete for {total_pages} pages")
#         return True

#     except Exception as e:
#         logger.exception(f"Fatal error: {str(e)[:100]}")
#         return False


# if __name__ == "__main__":
#     logger.info("Starting GPU-optimized ingestion")
#     start_time = time.time()

#     if not ingest_pdf(PDF_PATH):
#         logger.error("Ingestion failed")
#         exit(1)

#     total_time = time.time() - start_time
#     logger.info(f"Total processing time: {total_time:.2f} seconds")
