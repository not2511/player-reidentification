import torch
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import json

# Load embeddings
def load_embeddings(folder):
    embeddings = []
    ids = []
    paths = []

    for file in Path(folder).glob("*.pt"):
        data = torch.load(file)
        embeddings.append(data["embedding"].numpy())
        ids.append(data["id"])
        paths.append(data["crop_path"])
    return embeddings, ids, paths

# Load both sets
broadcast_embs, broadcast_ids, broadcast_paths = load_embeddings("outputs/embeddings")
tacticam_embs, tacticam_ids, tacticam_paths = load_embeddings("outputs/tacticam_embeddings")

# Nearest Neighbor model (cosine distance)
nn = NearestNeighbors(n_neighbors=1, metric="cosine")
nn.fit(broadcast_embs)

# Match each tacticam player
matches = {}
for idx, emb in enumerate(tacticam_embs):
    distance, neighbor_idx = nn.kneighbors([emb], return_distance=True)
    match = {
        "tacticam_id": tacticam_ids[idx],
        "tacticam_path": tacticam_paths[idx],
        "broadcast_id": broadcast_ids[neighbor_idx[0][0]],
        "broadcast_path": broadcast_paths[neighbor_idx[0][0]],
        "cosine_distance": float(distance[0][0])
    }
    matches[f"tacticam_id_{tacticam_ids[idx]}"] = match

# Save matches
with open("outputs/matched_pairs.json", "w") as f:
    json.dump(matches, f, indent=2)

print("Player matches saved to outputs/matched_pairs.json")
