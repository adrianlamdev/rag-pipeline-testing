from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

sentences_list = [
    ["Iron Man", "Let's go see the monkeys"],
    ["Iron Man", "I love superhero movies"],
    ["Let's go see the monkeys", "Visiting the zoo is fun"],
    ["Completely unrelated sentence", "Something else entirely"],
]

for pair in sentences_list:
    embeddings_1 = model.encode([pair[0]], normalize_embeddings=True)
    embeddings_2 = model.encode([pair[1]], normalize_embeddings=True)
    similarity = embeddings_1 @ embeddings_2.T
    print(f"Similarity between '{pair[0]}' and '{pair[1]}': {similarity[0][0]:.4f}")
