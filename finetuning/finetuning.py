import sys
from os import path

from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

# TODO: test save and load model

def main():
    # load dataset from txt file
    file_name = sys.argv[1]
    file_path = path.join("/raid/gpt_data/test", file_name)
    dataset = load_dataset("text", data_files=file_path, split= "train")

    
    # tokenize dataset using map and gpt-j 6b tokenizer with an additional token used to seperate post and comment
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", additional_special_tokens=["<|sepoftext|>"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, add_special_tokens=True, padding="max_length") 
    
    # map tokenizer over complete dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # create smaller dataset for testing
    small_dataset = tokenized_dataset.shuffle(seed=42).select(range(1000))

    print(tokenized_dataset)

    # define model and training arguments
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    model.resize_token_embeddings(len(tokenizer))
    training_args = TrainingArguments(output_dir="/raid/gpt_models/trainer")

    trainer = Trainer(model=model, args=training_args, train_dataset=small_dataset, tokenizer=tokenizer) # train & eval datasets?

    # finetuning process
    train_result = trainer.train()

    # save finetuned model to be able to evaluate it
    model_name = file_name.split(".")[0] + "_gpt"
    model_path = path.join("/raid/gpt_models", model_name)
    trainer.save_model(output_dir=model_path)  # Saves the tokenizer too
    
    

if __name__ == "__main__":
    main()
