from rag import RAG
import pandas as pd


def load_wiki_movies():
    data = pd.read_csv("./data/wiki_movie_plots_deduped.csv")
    movies = data["Plot"]
    movies.reset_index(drop=True, inplace=True)
    return movies


rag = RAG()
movies = load_wiki_movies()[:100].tolist()
rag.add_documents(movies)

query = "The film opens with two bandits"

results = rag.search(query, top_k=5)
for i, (text, score) in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"Score: {score:.4f}")
    print(f"Text: {text[:200]}...")


# if __name__ == "__main__":
#     print("Starting RAG Pipeline Tests...")
#
#     try:
#         print("\n=== Basic Functionality Test ===")
#         test_basic_functionality()
#
#         print("\n=== Multiple Documents Test ===")
#         test_multiple_documents()
#
#     except Exception as e:
#         print(f"\nError occurred: {str(e)}")
#
#     print("\nTests completed.")
