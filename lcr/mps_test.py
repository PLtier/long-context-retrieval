import torch

if torch.backends.mps.is_available():
    print("Success: Apple M3 Pro MPS (GPU) is active!")
elif torch.cuda.is_available():
    print("Success: NVIDIA CUDA (GPU) is active!")
else:
    print("Using standard CPU.")

from sentence_transformers import SentenceTransformer

# Load model and send to Apple's Metal backend
model = SentenceTransformer(
    "perplexity-ai/pplx-embed-context-v1-0.6B", trust_remote_code=True, device="mps"
)

# model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")

texts = ["Napolean was bron in 1769", "He played a basketball game in 2000."]  # Up to 10 chunks
embeddings = model.encode(texts)
query = "What did Napolean do"
query_embedding = model.encode([query])

# compute cosine similarity between query and chunks
# similarities = torch.nn.functional.cosine_similarity(
#     torch.tensor(query_embedding), torch.tensor(embeddings)
# )

similarities = model.similarity(query_embedding, embeddings)

# print similarities
print("Similarities:", similarities)

# print("Embeddings shape:", embeddings.shape)  # Should be (num_chunks, embedding_dim)

