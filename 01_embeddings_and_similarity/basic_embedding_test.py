from sentence_transformers import SentenceTransformer

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside today.",
    "We drove to the stadium",
]

sentence_embeddings = bert_model.encode(sentences)
print(sentence_embeddings.shape)

similarities = bert_model.similarity(sentence_embeddings, sentence_embeddings)
print(similarities)
