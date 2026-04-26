# To get a more accurate ranking,
# search again but this time rerank the initial results based on their relevance to the query.

from c_upsert_text import dense_index

query = "Famous historical structures and monuments"

# Search the dense index and results
reranked_results = dense_index.search(
    namespace = "example-namespace",
    query = {
        "top_k":10,
        "inputs": {
            "text": query
        }
    },

    rerank= {
        "model": "bge-reranker-v2-m3",
        "top_n": 10,
        "rank_fields": ["chunk_text"]
    }
)

# Print the reranked results
for hit in reranked_results['result']['hits']:
    print(f"id: {hit['_id']}, score: {round(hit['_score'], 2)}, text: {hit['fields']['chunk_text']}, category: {hit['fields']['category']}")