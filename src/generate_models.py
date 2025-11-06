# =============================
# generate_models.py
# =============================

import pandas as pd
import numpy as np
import faiss, pickle, os
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Load your text data
# -----------------------------
# Example: Replace this with your actual article dataset
articles = pd.DataFrame({
    "id": [1, 2, 3],
    "title": [
        "How to reset your password",
        "Troubleshooting network issues",
        "Setting up two-factor authentication"
    ],
    "content": [
        "To reset your password, click on 'Forgot Password' and follow the steps.",
        "Check your cables and restart your router to fix network issues.",
        "Enable 2FA from the settings menu to add extra account security."
    ]
})

# -----------------------------
# 2. Choose and Load the Embedding Model
# -----------------------------
model_name = "all-MiniLM-L6-v2"  # Fast & lightweight model
model = SentenceTransformer(model_name)

# -----------------------------
# 3. Create Embeddings
# -----------------------------
print("Encoding article texts...")
embeddings = model.encode(
    articles["content"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# -----------------------------
# 4. Build and Save FAISS Index
# -----------------------------
dimension = embeddings.shape[1]  # vector size (e.g., 384 for MiniLM)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Ensure folder exists
os.makedirs("models", exist_ok=True)

faiss.write_index(index, "models/article_index.faiss")
print("✅ Saved FAISS index to models/article_index.faiss")

# -----------------------------
# 5. Save Article Metadata
# -----------------------------
articles.to_pickle("models/articles_meta.pkl")
print("✅ Saved article metadata to models/articles_meta.pkl")

# -----------------------------
# 6. Save Embedding Model Info
# -----------------------------
with open("models/embed_model.pkl", "wb") as f:
    pickle.dump({"model_name": model_name}, f)
print("✅ Saved embedding model info to models/embed_model.pkl")

print("\n🎉 Model generation complete!")
