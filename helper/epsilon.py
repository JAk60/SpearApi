from sentence_transformers import SentenceTransformer, util
import numpy as np
from spear.labeling import continuous_scorer

model = SentenceTransformer('all-MiniLM-L6-v2')
 
lf_statement = "LF statement" #our hypthesis =====>>>  if lf is word then this will work best, but can be redundant if we ar ehaving the stement wise LFs
lf_vector = model.encode(lf_statement, convert_to_tensor=True)

def get_epsilon_neighbors(vector, all_vectors, epsilon=0.01): #we can experiment on the value of epislon
 
    similarities = util.cos_sim(vector, all_vectors)[0]
    neighbors = np.where(similarities >= 1 - epsilon)[0]
    return neighbors

@continuous_scorer()
def sentence_transformer_similarity_with_vicinity(sentence, all_statements, epsilon=0.01, **kwargs):
 
    input_vector = model.encode(sentence, convert_to_tensor=True)
    
 
    all_vectors = model.encode(all_statements, convert_to_tensor=True)
    
  
    neighbors = get_epsilon_neighbors(lf_vector, all_vectors, epsilon)
    
    max_similarity = 0
    source = "None"
    
 
    lf_similarity = util.cos_sim(input_vector, lf_vector).item()
    if lf_similarity > max_similarity:
        max_similarity = lf_similarity
        source = "Original LF vector"
 
    for neighbor_index in neighbors:
        neighbor_vector = all_vectors[neighbor_index]
        similarity = util.cos_sim(input_vector, neighbor_vector).item()
        if similarity > max_similarity:
            max_similarity = similarity
            source = f"Augmented vector (Index: {neighbor_index})"
    
    print(f"Similarity source: {source}, Similarity score: {max_similarity}")
    return max_similarity

# @labeling_function(cont_scorer=sentence_transformer_similarity_with_vicinity, pre=[convert_to_lower], label=ClassLabels.HAM)
# def transformer_polarity_with_vicinity(x, **kwargs):
 
#     if kwargs["continuous_score"] >= 0.9:
#         return ClassLabels.HAM
#     else:
#         return ClassLabels.SPAM