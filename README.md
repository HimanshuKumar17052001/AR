# Financial Table Extractor

A Streamlit application for extracting financial tables (Balance Sheet, Profit & Loss Account, Cash Flow Statement) from annual report PDFs using advanced OCR and AI processing.

## Features

- 📁 **File Upload**: Upload PDF annual reports with proper naming convention
- 🔍 **AI Processing**: Uses Claude AI for content summarization and sparse embeddings
- 📊 **Table Extraction**: Extracts financial tables using OCR technology
- 📋 **Excel Output**: Generates formatted Excel files with multiple worksheets
- 🏷️ **Smart Labeling**: Automatically labels tables as Standalone/Consolidated

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file with the following variables:
   ```env
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   AWS_REGION_NAME=us-east-1
   CLAUDE_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
   ANTHROPIC_VERSION=bedrock-2023-05-31
   SPARSE_EMBEDDING_URL=http://52.7.81.94:8010/embed
   OCR_SERVICE_URL=http://52.7.81.94:8000/ocr_image
   ```

## Usage

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### File Naming Convention

Upload your PDF files with the following naming convention:
- **Format**: `COMPANY_FINANCIAL_YEAR.pdf`
- **Examples**: 
  - `HDFC_FY2024.pdf`
  - `ICICI_FY2023.pdf`
  - `AXIS_FY2024.pdf`

### Process Flow

1. **Upload PDF**: Select your annual report PDF file
2. **Processing**: The app will:
   - Ingest the PDF and generate embeddings
   - Search for financial table phrases
   - Extract tables using OCR
3. **Download**: Get your results as a ZIP file containing:
   - Excel file with formatted worksheets
   - OCR images for verification

### Output Format

The generated Excel file (`COMPANY_financial_statements.xlsx`) contains:

#### Worksheets:
- **Standalone Balance Sheet**
- **Standalone Profit & Loss Account**
- **Standalone Cash Flow Statement**
- **Consolidated Balance Sheet**
- **Consolidated Profit & Loss Account**
- **Consolidated Cash Flow Statement**

#### Formatting:
- Bold headers and first column
- Table borders
- Auto-sized columns
- Professional layout

## Backend Components

The app uses three main backend scripts:

1. **`ingest.py`**: PDF ingestion and embedding generation
2. **`keyword_search.py`**: Financial table phrase search
3. **`extract_tables.py`**: Table extraction and Excel generation

## Technical Details

- **AI Model**: Claude 3 Sonnet for content summarization
- **Vector Database**: Qdrant for sparse embeddings
- **OCR Service**: Hosted OCR service for table extraction
- **PDF Processing**: PyPDF2 and pdfplumber for text extraction
- **Excel Generation**: openpyxl for formatted output

## Troubleshooting

### Common Issues:

1. **Environment Variables**: Ensure all required environment variables are set
2. **File Permissions**: Make sure the app has write permissions for creating output files
3. **Network Access**: Verify access to Qdrant, OCR service, and AWS Bedrock
4. **PDF Quality**: Ensure PDF files are readable and not password-protected

### Error Messages:

- **"Ingestion failed"**: Check PDF file and environment variables
- **"Keyword search failed"**: Verify Qdrant connection
- **"Table extraction failed"**: Check OCR service availability

## Support

For issues or questions, please check:
1. Environment variable configuration
2. Network connectivity to external services
3. PDF file format and quality
4. System requirements and dependencies 