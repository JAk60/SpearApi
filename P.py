# import pandas as pd

# # full_path = "./full.csv"
# # df_full = pd.read_csv(full_path)
# # label_instances = list(df_full["Task Objective"].value_counts().index)
# # print(label_instances)


# # import pandas as pd

# # Read the CSV file
# full_path = "./full.csv"
# df_full = pd.read_csv(full_path)

# def extract_unique_labels(df):
#     # Get the label instances from value_counts()
#     label_instances = list(df["Task Objective"].value_counts().index)
    
#     # Create an empty set to store unique labels
#     unique_labels = set()
    
#     # Process each label instance
#     for item in label_instances:
#         # Split by comma and strip whitespace
#         individual_labels = [label.strip() for label in item.split(',')]
#         # Add each label to the set
#         unique_labels.update(individual_labels)
    
#     # Convert set back to sorted list
#     return sorted(list(unique_labels))

# # Get unique labels
# unique_labels = extract_unique_labels(df_full)

# # Print results
# print("Unique Labels:")
# for label in unique_labels:
#     print(f"- {label}")

# # Optional: Print count of unique labels
# print(f"\nTotal number of unique labels: {len(unique_labels)}")
from sentence_transformers import SentenceTransformer

def scenarios_embedding(scenarios):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(scenarios)
    return embeddings

# Now this should match your training dimensions
embeddings = scenarios_embedding("At 03:45 UTC, a missile of unknown origin was detected on radar, rapidly approaching the fleet located in the southern waters of the Pacific. Its trajectory suggests a potential direct hit. All defense systems must be activated immediately to intercept the threat.")
print(embeddings.shape)  # Verify dimensions