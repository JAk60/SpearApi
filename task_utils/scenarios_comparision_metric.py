import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
from sklearn.metrics.pairwise import cosine_similarity


from embedding import scenarios_embedding

# Ensure you have the required NLTK data files
nltk.download('punkt')

def scenarios_cosine_similarity(paragraph1, paragraph2):

    embedding1 = scenarios_embedding(paragraph1)
    embedding2 = scenarios_embedding(paragraph2)   
    return cosine_similarity(embedding1, embedding2)

# Function to calculate Jaccard similarity
def jaccard_similarity(paragraph1, paragraph2):
    tokens1 = set(word_tokenize(paragraph1.lower()))
    tokens2 = set(word_tokenize(paragraph2.lower()))
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    similarity = len(intersection) / len(union)
    return similarity

def levenshtein_similarity(paragraph1, paragraph2):
    tokens1 = word_tokenize(paragraph1.lower())
    tokens2 = word_tokenize(paragraph2.lower())
    return edit_distance(tokens1, tokens2)


def test():
    paragraph1 = "This is a sample paragraph for testing."
    paragraph2 = "This is another sample paragraph for comparison."

    print("Cosine Similarity:", scenarios_cosine_similarity(paragraph1, paragraph2))
    print("Jaccard Similarity:", jaccard_similarity(paragraph1, paragraph2))
    print("Levenshtein Distance:", levenshtein_similarity(paragraph1, paragraph2))

if __name__ == "__main__":
    test()