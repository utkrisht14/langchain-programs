# ##################
# Creating an index
# ##################

# Create dense or sparse indexes for semantic and lexical search
# Dense index store dense vectors, which are numerical representations of meaning
#  and relationships of text and data. We can use dense indexes for the semantic search or
# in combination with sparse indexes for hybrid search.

# Sparse indexes store sparse vectors, which are numerical representations of the words
# or phrases in a document.
# We can use sparse indexes for lexical search, or in combination with dense indexes for hybrid search.

# ---------------------------------------------------
# Creating an integrated embedding in Dense Index
# --------------------------------------------------

from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone
from pinecone_program.h_complete_quickstart_pinecone_code import records

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

index_name = "integrated-dense-py"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name = "index_name",
        cloud="aws",
        region="us-east-1",
        embed = {
            "model": "llama-text-embed-v2",
            "field_map" : {"text": "chunk_text"}
        }
    )

# ---------------------------------------
# Bring your own vectors in Dense Index
# ---------------------------------------

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec # ServerlessSpec means let Pinecone manage the serverside resources

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

index_name = "standard-dense-py"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        vector_type = "dense",
        dimensions = 1536,
        metric = "cosine",
        spec = ServerlessSpec(
            cloud = "aws",
            region = "us-east-1"
        ),
        deletion_protection="disabled",
        tags = {
            "environment": "development"
        }
    )

# -------------------------------------------
# Creating a Integrated Sparse Index
# -------------------------------------------

from pinecone import Pinecone

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

index_name = "integrated-sparse-py"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        cloud = "aws",
        region = "us-east-1",
        embed = {
            "model" : "pinecone-sparse-english-v0",
            "field_text" : {"text": "chunk_text"}
        }
    )


# -------------------------------------------------------
# Bringing own Vector in Sparse Index
# -------------------------------------------------------

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "standard-sparse-py"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        vector_type="sparse",
        metric = "dotproduct",
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
    )


# --------------------------------------------
# Upsert chunks
# --------------------------------------------

from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv

INDEX_HOST = os.getenv("INDEX_HOST")

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

index = pc.Index(host=INDEX_HOST)

index.upsert_records(
    "example-namespace",
    [
        {
            "_id" : "document1#chunk1",
            "chunk_text": "First chunk of the document content...",
            "document_id": "document1",
            "document_title": "Introduction to Vector Databases",
            "chunk_number": 1,
            "document_url": "https://www.datacamp.com/community/tutorials/introduction-vector-databases",
            "created_at": "2024-03-05",
            "document_type": "tutorial",

        },
{
      "_id": "document1#chunk2",
      "chunk_text": "Second chunk of the document content...",
      "document_id": "document1",
      "document_title": "Introduction to Vector Databases",
      "chunk_number": 2,
      "document_url": "https://example.com/docs/document1",
      "created_at": "2024-01-15",
      "document_type": "tutorial"
    },
    {
      "_id": "document1#chunk3",
      "chunk_text": "Third chunk of the document content...",
      "document_id": "document1",
      "document_title": "Introduction to Vector Databases",
      "chunk_number": 3,
      "document_url": "https://example.com/docs/document1",
      "created_at": "2024-01-15",
      "document_type": "tutorial"
    },
    ]
)


# --------------------------------------------
# Search chunks
# --------------------------------------------

from pinecone import Pinecone

pc = Pinecone(os.getenv("PINECONE_API_KEY"))

index = pc.Index(host=INDEX_HOST)

filtered_result = index.search(
    namespace = "example-namespace",
    query = {
        "inputs": {"text": "What is vector database?"},
        "top_k": 3,
        "filter": {"document_id": "document_1"}
    },
    fields = ["chunk_text"]
)

print(filtered_result)



# --------------------------------------------
# Fetch chunks
# --------------------------------------------

# Here we try to fecth the chunk using chunk_id

# To retrieve all chunks for a specific document, first list the record IDs using the document prefix,
# and then fetch the complete records:

from pinecone.grpc import PineconeGRPC as Pinecone

pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
INDEX_HOST = os.getenv("INDEX_HOST")

index = pc.Index(host=INDEX_HOST)

# List all chunks for dicument1 using ID prefix
chunk_ids = []

for record_id in index.list_records(namespace="example-namespace", prefix="document1#"):
    chunk_ids.append(record_id)

print(f"Found {len(chunk_ids)} chunks for document1:")


# Fetch the complete records
if chunk_ids:
    records = index.fetch(id=chunk_ids, namespace="example-namespace")

    for record_id, record_data in records['vectors'].items():
        print(f"Chunk ID: {record_id}")
        print(f"Chunk text: {record_data['metadata']['chunk_text']}")
        # Process the vector values and metadata as needed


# --------------------------------------------
# Update chunks
# --------------------------------------------

from pinecone.grpc import PineconeGRPC as Pinecone

pc = Pinecone(os.getenv("PINECONE_API_KEY"))

INDEX_HOST = os.getenv("INDEX_HOST")

index = pc.Index(INDEX_HOST)

# List all chunks for document1
chunk_ids = []
for record_id in index.list(prefix="document1", namespace="example-namespace"):
    chunk_ids.append(record_id)

# Update specific chunks (e.g. update chunk2)
if "document1#chunk2" in chunk_ids:
    new_vector = ...  # from our embedding model
    index.update(
        id='document1#chunk2',
        values=new_vector,
        set_metadata={
            "document_id": "document1",
            "document_title": "Introduction to Vector Databases - Revised",
            "chunk_number": 2,
            "chunk_text": "Updated second chunk content...",
            "document_url": "https://example.com/docs/document1",
            "created_at": "2024-01-15",
            "updated_at": "2024-02-15",
            "document_type": "tutorial"
        },
        namespace='example-namespace'
    )
    print("Updated chunk 2 successfully")








