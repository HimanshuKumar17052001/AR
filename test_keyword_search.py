#!/usr/bin/env python3
"""
Test script to verify keyword search logic for different PDF filenames.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from keyword_search import extract_table_info, get_collection_name

def test_collection_naming():
    """Test collection naming logic."""
    print("Testing collection naming logic:")
    print("-" * 50)
    
    test_files = [
        "ICICI_2023-24.pdf",
        "ICICI_2023-34.pdf", 
        "HDFC_2024-25.pdf",
        "AXIS_2023-24.pdf",
        "ZOMATO_2024-25.pdf"
    ]
    
    for filename in test_files:
        collection_name = get_collection_name(filename)
        print(f"{filename:20} -> {collection_name}")
    
    print()

def test_extract_table_info():
    """Test table info extraction logic."""
    print("Testing table info extraction logic:")
    print("-" * 50)
    
    # Sample summary text
    sample_summary = "Standalone Balance Sheet. This table shows the financial position of the company as of March 31, 2024."
    
    test_files = [
        "ICICI_2023-34.pdf",  # Should use after first full stop
        "ICICI_2023-24.pdf",  # Should use before first full stop
        "HDFC_2024-25.pdf",   # Should use before first full stop
    ]
    
    for filename in test_files:
        extracted_text = extract_table_info(sample_summary, filename)
        print(f"{filename:20} -> {extracted_text}")
    
    print()

def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases:")
    print("-" * 50)
    
    # Test with no full stop
    summary_no_period = "Standalone Balance Sheet"
    print(f"No period: {extract_table_info(summary_no_period, 'ICICI_2023-34.pdf')}")
    print(f"No period: {extract_table_info(summary_no_period, 'HDFC_2024-25.pdf')}")
    
    # Test with multiple periods
    summary_multiple = "Standalone Balance Sheet. First sentence. Second sentence. Third sentence."
    print(f"Multiple periods (ICICI_2023-34): {extract_table_info(summary_multiple, 'ICICI_2023-34.pdf')}")
    print(f"Multiple periods (HDFC): {extract_table_info(summary_multiple, 'HDFC_2024-25.pdf')}")

if __name__ == "__main__":
    test_collection_naming()
    test_extract_table_info()
    test_edge_cases()
    
    print("✅ All tests completed!") 