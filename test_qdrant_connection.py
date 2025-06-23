import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from config import QDRANT_URL, QDRANT_API_KEY

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_qdrant_connection():
    """Test Qdrant connection and collection creation."""

    print("=== Qdrant Connection Test ===")
    print(f"QDRANT_URL: {QDRANT_URL}")
    print(f"QDRANT_API_KEY: {'***' if QDRANT_API_KEY else 'None'}")

    if not QDRANT_URL:
        print("ERROR: QDRANT_URL is not set!")
        return False

    try:
        # Test 1: Create client
        print("\n1. Creating Qdrant client...")
        _qdrant_kwargs = {"url": QDRANT_URL, "prefer_grpc": False}
        if QDRANT_API_KEY:
            _qdrant_kwargs["api_key"] = QDRANT_API_KEY

        qdrant_client = QdrantClient(**_qdrant_kwargs, check_compatibility=False)
        print("✓ Qdrant client created successfully")

        # Test 2: Check connection
        print("\n2. Testing connection...")
        collections = qdrant_client.get_collections()
        print(
            f"✓ Connection successful. Found {len(collections.collections)} collections"
        )

        # Test 3: Test collection creation
        print("\n3. Testing collection creation...")
        test_collection_name = "test_collection_12345"

        # Check if test collection exists
        if qdrant_client.collection_exists(test_collection_name):
            print(f"✓ Test collection '{test_collection_name}' already exists")
        else:
            print(f"Creating test collection '{test_collection_name}'...")

            # Create a simple sparse vector configuration
            sparse_cfg = {
                "test_vector": qm.SparseVectorParams(index=qm.SparseIndexParams())
            }

            qdrant_client.create_collection(
                test_collection_name,
                vectors_config={},  # no dense vectors
                sparse_vectors_config=sparse_cfg,
            )
            print(f"✓ Test collection '{test_collection_name}' created successfully")

        # Test 4: Clean up test collection
        print("\n4. Cleaning up test collection...")
        try:
            qdrant_client.delete_collection(test_collection_name)
            print(f"✓ Test collection '{test_collection_name}' deleted successfully")
        except Exception as e:
            print(f"Warning: Could not delete test collection: {e}")

        print("\n=== All tests passed! ===")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False


if __name__ == "__main__":
    test_qdrant_connection()
