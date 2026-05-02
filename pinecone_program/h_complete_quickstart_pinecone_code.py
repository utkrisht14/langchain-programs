import os

from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from .env file (where your API key is stored)
load_dotenv()

# Initialize Pinecone client using API key from environment
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# Name of the index (vector database)
index_name = "quickstart-py"

# Create index only if it doesn't already exist
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",              # Cloud provider
        region="us-east-1",       # Region where index is hosted
        embed={
            "model": "llama-text-embed-v2",   # Embedding model (converts text → vectors)
            "field_map": {"text": "chunk_text"}  # Map input "text" → record field "chunk_text"
        }
    )

# Sample dataset (each record = one document)
records = [
    # Each record has:
    # _id → unique identifier
    # chunk_text → actual content (used for embedding)
    # category → metadata (for filtering/display)
    { "_id": "rec1", "chunk_text": "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "category": "history" },
    { "_id": "rec2", "chunk_text": "Photosynthesis allows plants to convert sunlight into energy.", "category": "science" },
    { "_id": "rec3", "chunk_text": "Albert Einstein developed the theory of relativity.", "category": "science" },
    { "_id": "rec4", "chunk_text": "The mitochondrion is often called the powerhouse of the cell.", "category": "biology" },
    { "_id": "rec5", "chunk_text": "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "category": "literature" },
    { "_id": "rec6", "chunk_text": "Water boils at 100°C under standard atmospheric pressure.", "category": "physics" },
    { "_id": "rec7", "chunk_text": "The Great Wall of China was built to protect against invasions.", "category": "history" },
    { "_id": "rec8", "chunk_text": "Honey never spoils due to its low moisture content and acidity.", "category": "food science" },
    { "_id": "rec9", "chunk_text": "The speed of light in a vacuum is approximately 299,792 km/s.", "category": "physics" },
    { "_id": "rec10", "chunk_text": "Newton's laws describe the motion of objects.", "category": "physics" },
    { "_id": "rec11", "chunk_text": "The human brain has approximately 86 billion neurons.", "category": "biology" },
    { "_id": "rec12", "chunk_text": "The Amazon Rainforest is one of the most biodiverse places on Earth.", "category": "geography" },
    { "_id": "rec13", "chunk_text": "Black holes have gravitational fields so strong that not even light can escape.", "category": "astronomy" },
    { "_id": "rec14", "chunk_text": "The periodic table organizes elements based on their atomic number.", "category": "chemistry" },
    { "_id": "rec15", "chunk_text": "Leonardo da Vinci painted the Mona Lisa.", "category": "art" },
    { "_id": "rec16", "chunk_text": "The internet revolutionized communication and information sharing.", "category": "technology" },
    { "_id": "rec17", "chunk_text": "The Pyramids of Giza are among the Seven Wonders of the Ancient World.", "category": "history" },
    { "_id": "rec18", "chunk_text": "Dogs have an incredible sense of smell, much stronger than humans.", "category": "biology" },
    { "_id": "rec19", "chunk_text": "The Pacific Ocean is the largest and deepest ocean on Earth.", "category": "geography" },
    { "_id": "rec20", "chunk_text": "Chess is a strategic game that originated in India.", "category": "games" },
    { "_id": "rec21", "chunk_text": "The Statue of Liberty was a gift from France to the United States.", "category": "history" },
    { "_id": "rec22", "chunk_text": "Coffee contains caffeine, a natural stimulant.", "category": "food science" },
    { "_id": "rec23", "chunk_text": "Thomas Edison invented the practical electric light bulb.", "category": "inventions" },
    { "_id": "rec24", "chunk_text": "The moon influences ocean tides due to gravitational pull.", "category": "astronomy" },
    { "_id": "rec25", "chunk_text": "DNA carries genetic information for all living organisms.", "category": "biology" },
    { "_id": "rec26", "chunk_text": "Rome was once the center of a vast empire.", "category": "history" },
    { "_id": "rec27", "chunk_text": "The Wright brothers pioneered human flight in 1903.", "category": "inventions" },
    { "_id": "rec28", "chunk_text": "Bananas are a good source of potassium.", "category": "nutrition" },
    { "_id": "rec29", "chunk_text": "The stock market fluctuates based on supply and demand.", "category": "economics" },
    { "_id": "rec30", "chunk_text": "A compass needle points toward the magnetic north pole.", "category": "navigation" },
    { "_id": "rec31", "chunk_text": "The universe is expanding, according to the Big Bang theory.", "category": "astronomy" },
    { "_id": "rec32", "chunk_text": "Elephants have excellent memory and strong social bonds.", "category": "biology" },
    { "_id": "rec33", "chunk_text": "The violin is a string instrument commonly used in orchestras.", "category": "music" },
    { "_id": "rec34", "chunk_text": "The heart pumps blood throughout the human body.", "category": "biology" },
    { "_id": "rec35", "chunk_text": "Ice cream melts when exposed to heat.", "category": "food science" },
    { "_id": "rec36", "chunk_text": "Solar panels convert sunlight into electricity.", "category": "technology" },
    { "_id": "rec37", "chunk_text": "The French Revolution began in 1789.", "category": "history" },
    { "_id": "rec38", "chunk_text": "The Taj Mahal is a mausoleum built by Emperor Shah Jahan.", "category": "history" },
    { "_id": "rec39", "chunk_text": "Rainbows are caused by light refracting through water droplets.", "category": "physics" },
    { "_id": "rec40", "chunk_text": "Mount Everest is the tallest mountain in the world.", "category": "geography" },
    { "_id": "rec41", "chunk_text": "Octopuses are highly intelligent marine creatures.", "category": "biology" },
    { "_id": "rec42", "chunk_text": "The speed of sound is around 343 meters per second in air.", "category": "physics" },
    { "_id": "rec43", "chunk_text": "Gravity keeps planets in orbit around the sun.", "category": "astronomy" },
    { "_id": "rec44", "chunk_text": "The Mediterranean diet is considered one of the healthiest in the world.", "category": "nutrition" },
    { "_id": "rec45", "chunk_text": "A haiku is a traditional Japanese poem with a 5-7-5 syllable structure.", "category": "literature" },
    { "_id": "rec46", "chunk_text": "The human body is made up of about 60% water.", "category": "biology" },
    { "_id": "rec47", "chunk_text": "The Industrial Revolution transformed manufacturing and transportation.", "category": "history" },
    { "_id": "rec48", "chunk_text": "Vincent van Gogh painted Starry Night.", "category": "art" },
    { "_id": "rec49", "chunk_text": "Airplanes fly due to the principles of lift and aerodynamics.", "category": "physics" },
    { "_id": "rec50", "chunk_text": "Renewable energy sources include wind, solar, and hydroelectric power.", "category": "energy" }
]

# Connect to the index
dense_index = pc.Index(index_name)

# Insert (upsert) records into a namespace
dense_index.upsert_records("example-namespace", records)

# Wait for indexing to complete (important before querying)
import time
time.sleep(10)

# View index statistics (like number of vectors stored)
stats = dense_index.describe_index_stats()
print("Stats: \n", stats)

# ---------------------------
# Semantic Search
# ---------------------------
query = "Famous historical structures and monuments"

results = dense_index.search(
    namespace="example-namespace",
    query={
        "top_k": 10,  # number of results to return
        "inputs": {
            "text": query  # query is automatically embedded
        }
    }
)

# Print search results
print("\n Original Search results: \n")
for hit in results["result"]["hits"]:
    print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | "
          f"category: {hit['fields']['category']:<10} | "
          f"text: {hit['fields']['chunk_text']:<50}")


# ---------------------------
# Reranking (better relevance)
# ---------------------------
# Step 1: Retrieve top_k results using semantic search
# Step 2: Rerank those results using a stronger reranker model

reranked_results = dense_index.search(
    namespace="example-namespace",
    query={
        "inputs": {
            "text": query  # same query used earlier
        },
        "top_k": 10  # REQUIRED: number of initial candidates
    },
    rerank={
        "model": "bge-reranker-v2-m3",  # reranking model
        "top_n": 10,                    # final number of results after reranking
        "rank_fields": ["chunk_text"]   # field used for reranking
    },
    fields=["category", "chunk_text"]  # fields to return in results
)

# Print reranked results
print("\n Reranked Search results: \n")
for hit in reranked_results["result"]["hits"]:
    print(
        f"id: {hit['_id']}, "
        f"score: {round(hit['_score'], 2)}, "
        f"text: {hit['fields']['chunk_text']}, "
        f"category: {hit['fields']['category']}"
    )