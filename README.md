# Hybrid RAG

A Python FastAPI application that answers questions from uploaded PDF documents using a hybrid retrieval pipeline. It combines dense vector search with sparse BM25 retrieval, reranks the combined results with Cohere, and generates answers with OpenAI.

## Features

- Upload a PDF through a REST API
- Extract and split PDF text into overlapping chunks
- Generate document embeddings with OpenAI
- Store vectors in an in-memory Qdrant collection
- Combine semantic MMR search and BM25 keyword search using an ensemble retriever
- Rerank retrieved documents with Cohere
- Generate grounded answers with OpenAI `gpt-4o-mini`
- Run locally or with Docker

## How It Works

```text
PDF upload
    |
    v
PyPDFLoader -> text chunks -> OpenAI embeddings
                                      |
                                      v
                              In-memory Qdrant
                                      |
                       +--------------+--------------+
                       |                             |
                 Vector search (60%)          BM25 search (40%)
                       +--------------+--------------+
                                      |
                                      v
                              Cohere reranking
                                      |
                                      v
                          Prompt + retrieved context
                                      |
                                      v
                        OpenAI GPT-4o-mini answer
```

## Project Structure

```text
hybrid-rag/
├── app.py             # FastAPI application and /rag endpoint
├── rag_pipeline.py    # PDF loading, chunking, retrieval, reranking, and generation
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container configuration
└── data/
    └── ODI_rules.pdf  # Example PDF document
```

## Requirements

- Python 3.11 or later
- OpenAI API key
- Cohere API key
- Docker optional for containerized execution

## Installation

Clone the repository:

```bash
git clone https://github.com/upskillvikas/hybrid-rag.git
cd hybrid-rag
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Set the API keys required by LangChain:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export COHERE_API_KEY="your-cohere-api-key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-openai-api-key"
$env:COHERE_API_KEY="your-cohere-api-key"
```

The application uses:

| Variable | Used for |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI embeddings and ChatOpenAI generation |
| `COHERE_API_KEY` | Cohere document reranking |
| `PORT` | Optional server port in Docker; defaults to `8000` |

## Running the API

Start the FastAPI server with Uvicorn:

```bash
uvicorn app:api --host 0.0.0.0 --port 8000
```

Open the interactive documentation at:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

## API Usage

### `POST /rag`

Submit a question and PDF using `multipart/form-data`.

**Request fields:**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `query` | string form field | Yes | Question to answer from the PDF |
| `file` | PDF file | Yes | PDF document to search |

**Example with cURL:**

```bash
curl -X POST "http://localhost:8000/rag" \
  -F 'query=What are the main rules described in this document?' \
  -F 'file=@data/ODI_rules.pdf'
```

**Example response:**

```json
{
  "answer": "The document describes the main rules..."
}
```

If the retrieved context is insufficient, the pipeline is prompted to return `Not Found`.

## Docker

Build the image:

```bash
docker build -t hybrid-rag .
```

Run the container:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e COHERE_API_KEY="your-cohere-api-key" \
  hybrid-rag
```

To use a different host port:

```bash
docker run --rm -p 8080:8000 \
  -e PORT=8000 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e COHERE_API_KEY="your-cohere-api-key" \
  hybrid-rag
```

The API will then be available at <http://localhost:8080/docs>.

## Pipeline Configuration

The default settings are defined in `rag_pipeline.py`:

| Setting | Default |
| --- | --- |
| Chunk size | 800 characters |
| Chunk overlap | 150 characters |
| Dense/MMR result count | 10 |
| BM25 result count | 10 |
| Vector/BM25 weights | 60% / 40% |
| Reranker model | `rerank-v3.5` |
| Reranked document count | 4 |
| Generator model | `gpt-4o-mini` |
| Generator temperature | `0` |

Vector storage is in-memory and is recreated for each request. Uploaded files are written to a temporary location and removed after processing.

## Using the Pipeline Directly

You can call the pipeline outside FastAPI:

```python
from rag_pipeline import rag_pipeline

answer = rag_pipeline(
    file_path="data/ODI_rules.pdf",
    query="Summarize the most important rules",
)

print(answer)
```

## Limitations

- Only PDF uploads are supported because the API currently forces the `.pdf` temporary-file suffix
- Processing a separate PDF on every request may be slower for large documents
- Qdrant data is stored only in memory and is lost after each pipeline run or server restart
- No authentication or rate limiting is currently implemented
- API keys must be kept secure and are not included in the repository

## Security

- Never commit API keys or credentials
- Use environment variables or a secrets manager in production
- Add authentication before exposing the API publicly
- Validate uploaded files, maximum upload size, and request rate if deployed on a public network

## License

No license has been specified for this repository. All rights remain with the repository owner unless a license is added.