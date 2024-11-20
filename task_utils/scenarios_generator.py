import os
import numpy as np
import pandas as pd
import torch
import transformers

def load_pretrained_tokenizer_and_model(
    model_class=transformers.AutoModelForSeq2SeqLM, 
    tokenizer_class=transformers.AutoTokenizer, 
    pretrained_weights= "humarin/chatgpt_paraphraser_on_T5_base"
):
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return tokenizer, model

def paraphrase(
    question,
    model_class=transformers.AutoModelForSeq2SeqLM,
    tokenizer_class=transformers.AutoTokenizer,
    pretrained_weights= "humarin/chatgpt_paraphraser_on_T5_base",
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    tokenizer, model = load_pretrained_tokenizer_and_model(model_class, tokenizer_class, pretrained_weights)
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

def generate_scenarios(path1, output_directory, train_size=32, num_return_sequences=10):
    os.makedirs(output_directory, exist_ok=True)

    files1 = os.listdir(path1)
    csv_files1 = [file for file in files1 if file.endswith('.csv')]
    
    for file1 in csv_files1:
        df = pd.read_csv(os.path.join(path1, file1))

        scenarios_list = df['Scenario'].to_numpy()
        generated_scenarios = []
        for scenario in scenarios_list[:train_size]:
            generated_scenarios += paraphrase(
                scenario, 
                num_return_sequences=num_return_sequences, 
                max_length=512, 
                num_beams=2 * num_return_sequences, 
                num_beam_groups=2 * num_return_sequences
            )
        df_generated = pd.DataFrame(generated_scenarios, columns=['Scenario'])
        df_generated.to_csv(os.path.join(output_directory, file1), index=False)

if __name__ == "__main__":
    # Default to a specific paraphraser model
    model_class, tokenizer_class, pretrained_weights = (
        transformers.AutoModelForSeq2SeqLM, 
        transformers.AutoTokenizer, 
        "humarin/chatgpt_paraphraser_on_T5_base"
    )

    # Example scenarios
    scenarios = ["This is a test"]
    generated_scenarios = paraphrase(
        scenarios[0], 
        model_class, 
        tokenizer_class, 
        pretrained_weights
    )
    print(generated_scenarios)
