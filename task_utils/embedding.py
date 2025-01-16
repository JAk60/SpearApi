import numpy as np
import torch
import transformers

def load_pretrained_tokenizer_and_model(
    model_class=transformers.DistilBertModel, 
    tokenizer_class=transformers.DistilBertTokenizer, 
    pretrained_weights='distilbert-base-uncased'
):
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return tokenizer, model

# def scenarios_embedding(
#     scenarios, 
#     model_class=transformers.DistilBertModel, 
#     tokenizer_class=transformers.DistilBertTokenizer, 
#     pretrained_weights='distilbert-base-uncased'
# ):
#     tokenizer, model = load_pretrained_tokenizer_and_model(model_class, tokenizer_class, pretrained_weights)
#     input = tokenizer(scenarios, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         last_hidden_states = model(**input)
#     features = last_hidden_states[0][:, 0, :].numpy()
#     return features

def scenarios_embedding(scenarios, model_class=transformers.DistilBertModel, 
                       tokenizer_class=transformers.DistilBertTokenizer,
                       pretrained_weights='distilbert-base-uncased'):
    tokenizer, model = load_pretrained_tokenizer_and_model(model_class, tokenizer_class, pretrained_weights)
    input = tokenizer(scenarios, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**input)
        # Get both hidden states and attention
        hidden_states = outputs[0]
        # Take multiple features: first token, mean, max
        first_token = hidden_states[:, 0, :]
        mean_pooling = torch.mean(hidden_states, dim=1)
        max_pooling = torch.max(hidden_states, dim=1)[0]
        
        # Concatenate different features
        combined = torch.cat([first_token, mean_pooling, max_pooling], dim=1)
        # Repeat the features to reach 4224
        repeats = 4224 // (768 * 3)
        remainder = 4224 % (768 * 3)
        
        features = torch.cat([combined] * repeats + [combined[:, :remainder]], dim=1)
        
    return features.numpy()
## save the scenario embedding
def save_scenarios_embedding(embedding, path):
    with open(path, 'wb') as f:
        np.save(f, embedding)
    
def open_scenarios_embedding(path):
    with open(path, 'rb') as f:
        embedding= np.load(f)
    return embedding

if __name__ == "__main__":
    # For DistilBERT:
    model_class, tokenizer_class, pretrained_weights = transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased'

    ## Want BERT instead of distilBERT? Uncomment the following line:
    # model_class, tokenizer_class, pretrained_weights = transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased'

    # For other model
    # model_class, tokenizer_class, pretrained_weights = (transformers.AutoModelForSeq2SeqLM, transformers.AutoTokenizer, "humarin/chatgpt_paraphraser_on_T5_base")

    scenarios = ["This is a test"]
    features = scenarios_embedding(scenarios, model_class, tokenizer_class, pretrained_weights)
    print(features.shape)