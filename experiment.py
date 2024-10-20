import numpy as np
from scipy.spatial.distance import cosine

def create_score_matrix(data_dict, score_function):
    keys = list(data_dict.keys())
    size = len(keys)
    score_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            score_matrix[i, j] = score_function(data_dict[keys[i]], data_dict[keys[j]])
    
    return score_matrix

# Example usage
data_dict = {
    'a': 1,
    'b': 2,
    'c': 3
}

def example_score_function(value1, value2):
    return value1 + value2

score_matrix = create_score_matrix(data_dict, example_score_function)
print(score_matrix)

def calculate_embedding_distance(embedding1, embedding2):
    """
    Calculate the distance score between two embeddings using cosine similarity.
    
    Args:
    embedding1 (np.array): First embedding vector
    embedding2 (np.array): Second embedding vector
    
    Returns:
    float: Cosine similarity score (higher value means more similar)
    """
    return 1 - cosine(embedding1, embedding2)

# Example usage for embedding distance
embedding1 = np.array([0.1, 0.2, 0.3, 0.4])
embedding2 = np.array([0.2, 0.3, 0.4, 0.5])

distance_score = calculate_embedding_distance(embedding1, embedding2)
print(f"Distance score between embeddings: {distance_score}")




def get_score(score_matrix):
    """
    Calculate the overall score from the score matrix.
    
    Args:
    score_matrix (np.array): The score matrix
    
    Returns:
    float: The overall score
    """
    return np.sum(score_matrix)

# Example usage to get the overall score
overall_score = get_score(score_matrix)
print(f"Overall score: {overall_score}")
