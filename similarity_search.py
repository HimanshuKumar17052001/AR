# import os
# import json
# import logging
# import requests
# from typing import List, Dict, Any, Tuple
# from dotenv import load_dotenv
# from qdrant_client import QdrantClient
# from qdrant_client.http import models as qm

# # ---------------------------------------------------------------------------
# # ENV & CONSTANTS
# # ---------------------------------------------------------------------------
# load_dotenv()

# COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "AR_EMBEDDINGS")
# QDRANT_URL: str = os.getenv("QDRANT_URL")
# QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
# SPARSE_EMBEDDING_URL: str = os.getenv(
#     "SPARSE_EMBEDDING_URL", "http://52.7.81.94:8010/embed"
# )
# VECTOR_NAME: str = "icici_pagewise_embedding"

# # Financial table queries
# FINANCIAL_TABLE_QUERIES = [
#     "Standalone Balance Sheet table",
#     "Balance Sheet table",
#     "Standalone Profit & Loss Account table",
#     "Profit & Loss Account table",
#     "Statement of Profit and Loss table",
#     "Standalone Cash Flow Statement table",
#     "Cash Flow Statement table",
#     "Consolidated Balance Sheet table",
#     "Consolidated Profit & Loss Account table",
#     "Consolidated Statement of Profit and Loss table",
#     "Consolidated Cash Flow Statement table",
# ]

# # ---------------------------------------------------------------------------
# # LOGGING
# # ---------------------------------------------------------------------------
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# logging.basicConfig(
#     format="%(asctime)s | %(levelname)-8s | %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     level=getattr(logging, LOG_LEVEL, logging.INFO),
# )
# logger = logging.getLogger("search")

# # ---------------------------------------------------------------------------
# # QDRANT CLIENT
# # ---------------------------------------------------------------------------
# if not QDRANT_URL:
#     raise ValueError("QDRANT_URL must be set in .env file")

# _qdrant_kwargs: Dict[str, Any] = {"url": QDRANT_URL, "prefer_grpc": False}
# if QDRANT_API_KEY:
#     _qdrant_kwargs["api_key"] = QDRANT_API_KEY

# qdrant_client = QdrantClient(**_qdrant_kwargs, check_compatibility=False)
# logger.info("Initialised Qdrant client with URL: %s", QDRANT_URL)

# # ---------------------------------------------------------------------------
# # HELPER FUNCTIONS
# # ---------------------------------------------------------------------------


# def generate_sparse_embedding(text: str) -> qm.SparseVector:
#     """Generate sparse embedding for the given text."""
#     try:
#         response = requests.post(SPARSE_EMBEDDING_URL, json={"text": text}, timeout=10)
#         response.raise_for_status()
#         data = response.json() or {}

#         indices = [int(k) for k in data.keys()]
#         values = [float(v) for v in data.values()]

#         logger.debug(f"Generated sparse embedding: {len(indices)} non-zero tokens")
#         return qm.SparseVector(indices=indices, values=values)

#     except Exception as e:
#         logger.error(f"Failed to generate sparse embedding: {e}")
#         raise


# def search_financial_table(query: str, limit: int = 3) -> List[Dict[str, Any]]:
#     """
#     Search for a financial table using sparse embeddings.
#     Returns top N results with page numbers.
#     """
#     logger.info(f"Searching for: '{query}' (limit={limit})")

#     try:
#         # Generate sparse embedding for the query
#         sparse_vec = generate_sparse_embedding(query)

#         # Search using sparse vector
#         results = qdrant_client.query_points(
#             collection_name=COLLECTION_NAME,
#             query=sparse_vec,
#             using=VECTOR_NAME,  # Use the sparse vector space name
#             with_payload=True,
#             limit=limit,
#         )

#         # Extract relevant information
#         search_results = []
#         for point in results.points:
#             search_results.append(
#                 {
#                     "page_num": point.payload.get("page_num"),
#                     "score": point.score,
#                     "summary": point.payload.get("summary", ""),
#                     "company_name": point.payload.get("company_name", ""),
#                 }
#             )

#         return search_results

#     except Exception as e:
#         logger.error(f"Search failed for query '{query}': {e}")
#         return []


# def search_all_financial_tables() -> Dict[str, List[Dict[str, Any]]]:
#     """
#     Search for all financial tables and return results grouped by table type.
#     """
#     results = {}

#     # Group similar queries
#     table_groups = {
#         "Standalone Balance Sheet": [
#             "Standalone Balance Sheet table",
#             "Balance Sheet table",
#         ],
#         "Standalone Profit & Loss": [
#             "Standalone Profit & Loss Account table",
#             "Profit & Loss Account table",
#             "Statement of Profit and Loss",
#         ],
#         "Standalone Cash Flow": [
#             "Standalone Cash Flow Statement table",
#             "Cash Flow Statement table",
#         ],
#         "Consolidated Balance Sheet": ["Consolidated Balance Sheet table"],
#         "Consolidated Profit & Loss": [
#             "Consolidated Profit & Loss Account table",
#             "Consolidated Statement of Profit and Loss",
#         ],
#         "Consolidated Cash Flow": ["Consolidated Cash Flow Statement table"],
#     }

#     for table_type, queries in table_groups.items():
#         logger.info(f"\n{'='*60}")
#         logger.info(f"Searching for: {table_type}")
#         logger.info(f"{'='*60}")

#         # Collect results from all related queries
#         all_results = []
#         for query in queries:
#             search_results = search_financial_table(query, limit=5)
#             all_results.extend(search_results)

#         # Remove duplicates based on page number and sort by score
#         unique_results = {}
#         for result in all_results:
#             page_num = result["page_num"]
#             if (
#                 page_num not in unique_results
#                 or result["score"] > unique_results[page_num]["score"]
#             ):
#                 unique_results[page_num] = result

#         # Get top 3 unique results
#         top_results = sorted(
#             unique_results.values(), key=lambda x: x["score"], reverse=True
#         )[:3]

#         results[table_type] = top_results

#         # Print results
#         if top_results:
#             for i, result in enumerate(top_results, 1):
#                 logger.info(
#                     f"  {i}. Page {result['page_num']} (Score: {result['score']:.4f})"
#                 )
#                 logger.info(f"     Summary: {result['summary'][:100]}...")
#         else:
#             logger.info("  No results found")

#     return results


# def main():
#     """Main function to search for financial tables."""
#     logger.info("Starting financial table search...")

#     # Check if collection exists
#     if not qdrant_client.collection_exists(COLLECTION_NAME):
#         logger.error(f"Collection '{COLLECTION_NAME}' does not exist!")
#         return

#     # Get collection info
#     collection_info = qdrant_client.get_collection(COLLECTION_NAME)
#     logger.info(
#         f"Collection '{COLLECTION_NAME}' has {collection_info.points_count} points"
#     )

#     # Search for all financial tables
#     results = search_all_financial_tables()

#     # Summary of results
#     logger.info(f"\n{'='*60}")
#     logger.info("SUMMARY OF SEARCH RESULTS")
#     logger.info(f"{'='*60}")

#     for table_type, top_results in results.items():
#         if top_results:
#             page_nums = [r["page_num"] for r in top_results]
#             logger.info(f"{table_type}: Pages {page_nums}")
#         else:
#             logger.info(f"{table_type}: No pages found")

#     # Save results to JSON file
#     output_file = "financial_tables_search_results.json"
#     with open(output_file, "w") as f:
#         json.dump(results, f, indent=2)
#     logger.info(f"\nResults saved to: {output_file}")


# if __name__ == "__main__":
#     main()
