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







