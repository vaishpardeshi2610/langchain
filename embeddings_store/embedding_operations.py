import numpy as np
from scipy.spatial.distance import cdist
from nltk.tokenize import sent_tokenize

class EmbeddingUtils:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def generate_transcript_embedding(self, transcript):
        """
        Generates an embedding for the entire transcript.
        """
        return self.embedding_model.embed_query(transcript)

    def find_relevant_section(self, user_query, transcript, threshold=0.3, top_n=3):
        """
        Finds the most relevant sections of the transcript based on cosine similarity.
        """
        transcript_sentences = sent_tokenize(transcript)
        sentence_embeddings = np.array([self.embedding_model.embed_query(sentence) for sentence in transcript_sentences])
        query_embedding = self.embedding_model.embed_query(user_query)

        similarities = 1 - cdist([query_embedding], sentence_embeddings, metric='cosine')[0]
        ranked_sentences = sorted(
            zip(transcript_sentences, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        return [(sent, sim) for sent, sim in ranked_sentences if sim >= threshold][:top_n]
