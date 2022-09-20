import csv
import re
from os import listdir, path

from detoxify import Detoxify
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPT2Tokenizer, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main():
    models = ["EleutherAI/gpt-neo-1.3B", "crypto_gpt_2", "wallstreetbets_gpt_2", "reddit_covid_gpt_2", "no_new_normal_gpt_2", "ummah_gpt_2", "christianchat_gpt_2"]
    model_names = ["GPT-Neo 1.3B", "Cryptocurrency", "WallStreetBets", "COVID", "NoNewNormal", "Ummah", "ChristianChat"]

    # define how many examples to generate with every prompt
    num_examples = 50

    # read prompt file and make list of prompts
    prompt_list =[]
    with open("/raid/wald/gpt_data/evaluation/prompts.csv") as prompt_file:
        prompt_reader= csv.DictReader(prompt_file)
        for row in prompt_reader:
            prompt_list.append(row)


    # initialize sentiment classifier
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # intialize list to save the generated examples with their sentiment and toxicity scores
    examples_list = []


    for i in range(len(models)):
        
        model_name = model_names[i]
        # baseline model
        if i == 0:
            model_file = models[i]
        # local finetuned models
        else:
            model_file = path.join("/raid/wald/gpt_models/2", models[i])

        # load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_file)
        tokenizer = GPT2Tokenizer.from_pretrained(model_file)
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[PAD]"))[0]

        # establish a pipeline for generating examples with cuda use
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        for row in tqdm(prompt_list):
            bias = row["bias"]
            demographic = row["demographic"]

            if i == 0:
                prompt = row["prompt"]
            # for the finetuned models, add special tokens to prompt
            else:
                prompt = "<|endoftext|>" + row["prompt"] + "<|sepoftext|>"

            # generate text and specify max length
            gen_outputs = generator(prompt, pad_token_id= pad_token_id, max_length=75, no_repeat_ngram_size=3, num_return_sequences=num_examples)
            for output in tqdm(gen_outputs):
                generated_text = generated_text = output["generated_text"]

                if i == 0:
                    example_text = generated_text.split(prompt)[1:]
                else:
                    # dismiss prompt and special tokens
                    pattern = "\<\|sepoftext\|\>(.*?)\<\|endoftext\|\>"
                    substring = re.search(pattern, generated_text)
    
                
                    if substring is None:
                        # no eos-token, simply take text after <|sepoftext|>
                        example_text = generated_text.split("<|sepoftext|>")[1]
                    else:
                        example_text = substring.group(1)

                current_example = {"model": model_name, "bias_category": bias, "demographic": demographic, "prompt": prompt, "generated_text": example_text}
                # determine sentiment and toxicity scores
                sentiment = sentiment_analyzer.polarity_scores(example_text)
                toxicity = Detoxify('original').predict(example_text)
                # add sentiment and toxicity to dictionary
                current_example.update(sentiment)
                current_example.update(toxicity)

                examples_list.append(current_example)
    

    keys = examples_list[0].keys()
    with open("/raid/wald/gpt_results/evaluation.csv", "w") as csv_file:
        csv_writer = csv.DictWriter(csv_file, keys)
        csv_writer.writeheader()
        csv_writer.writerows(examples_list)


if __name__ == "__main__":
    main()
    
    



    

                


