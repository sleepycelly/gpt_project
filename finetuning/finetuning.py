import sys
from os import path

from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)

# TODO: test save and load model

def main():
    # load dataset from txt file
    file_name = sys.argv[1]
    file_path = path.join("/raid/wald/gpt_data/train", file_name)
    dataset = load_dataset("text", data_files=file_path, split= "train")

    # path for saving model after finetuning
    model_name = file_name.split(".")[0] + "_gpt"
    model_path = path.join("/raid/wald/gpt_models", model_name)

    
    # tokenize dataset using map and gpt-j 6b tokenizer with an additional token used to seperate post and comment
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M", additional_special_tokens=["<|sepoftext|>"])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, add_special_tokens=True, padding="longest")
    
    # map tokenizer over complete dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=20)

    # create smaller dataset for testing
    small_dataset = tokenized_dataset.shuffle(seed=42).select(range(1000))

    # define model and training arguments
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    model.resize_token_embeddings(len(tokenizer))
    checkpoint_path = path.join("/cephfs/wald/checkpoints", model_name)
    training_args = TrainingArguments(output_dir=checkpoint_path, per_device_train_batch_size=4)

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer, data_collator=data_collator)

    # finetuning process
    train_result = trainer.train()

    # save finetuned model to be able to evaluate it
    trainer.save_model(output_dir=model_path)  # Saves the tokenizer too
    
    

if __name__ == "__main__":
    main()
