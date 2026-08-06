from sentence_transformers import SentenceTransformer

# 1. Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Your documents (knowledge base)
doc_names = ["Doc 1 (Weather)", "Doc 2 (Sun)", "Doc 3 (Sports)"]
doc_texts = [
    "The weather is lovely today.",
    "It's so sunny outside today.",
    "We drove to the stadium."
]

doc_embeddings = model.encode(doc_texts)

query = "Is it nice outside today?"
query_embedding = model.encode([query])

similarities = model.similarity(query_embedding, doc_embeddings)[0]

print("Query:", query)
print("-------------------")

for i in range(len(doc_names)):
    name = doc_names[i]
    score = similarities[i]
    print(name, "-> Score:", score)