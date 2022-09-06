import sys
from os import path

from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          DataCollatorForLanguageModeling, GPT2Tokenizer,
                          Trainer, TrainingArguments)


def main():
    # load dataset from text file
    file_name = sys.argv[1]
    file_path = path.join("/raid/wald/gpt_data/train", file_name)
    dataset = load_dataset("text", data_files=file_path, split= "train")

    # path for saving model after finetuning
    model_name = file_name.split(".")[0] + "_gpt_2"
    model_path = path.join("/raid/wald/gpt_models", model_name)

    
    # define tokenizer with special token for post/comment separation
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B", additional_special_tokens=["<|sepoftext|>"], pad_token="[PAD]")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, add_special_tokens=True, padding="max_length", max_length=128)
    
    # tokenize dataset by mapping tokenizer over the complete dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=20)

    # define model and training arguments
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model.resize_token_embeddings(len(tokenizer))
    checkpoint_path = path.join("/cephfs/wald/checkpoints", model_name)
    training_args = TrainingArguments(output_dir=checkpoint_path, per_device_train_batch_size=21, save_steps=2000, num_train_epochs=2)

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer, data_collator=data_collator)

    # finetuning process
    train_result = trainer.train()

    # save finetuned model to be able to evaluate it
    trainer.save_model(output_dir=model_path)  # Saves the tokenizer too
    
    

if __name__ == "__main__":
    main()
