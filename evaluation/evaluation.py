import csv
import re
from os import listdir, path

from detoxify import Detoxify
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main():
    models = listdir("/raid/wald/gpt_models/2")


    # define how many examples to generate with every prompt
    num_examples = 50

    # read prompt file and make list of prompts
    prompt_list =[]
    with open("/raid/wald/gpt_data/evaluation/gender_prompts.csv") as prompt_file:
        prompt_reader= csv.DictReader(prompt_file)
        for row in prompt_reader:
            prompt_list.append(row)

    # initialize sentiment classifier
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # intialize list to save the generated examples with their sentiment and toxicity scores
    examples_list = []

    for model_name in models:
        # load model and tokenizer
        file_path = path.join("/raid/wald/gpt_models/2", model_name)
        model = AutoModelForCausalLM.from_pretrained(file_path)
        tokenizer = GPT2Tokenizer.from_pretrained(file_path)
        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("[PAD]"))[0]
        for row in tqdm(prompt_list):
            bias = row["bias"]
            demographic = row["demographic"]
            # 'encode' prompt with the tokenizer
            prompt = row["prompt"]
            prompt_with_token = "<|endoftext|>" + prompt + "<|sepoftext|>"
            input_ids = tokenizer(prompt_with_token, return_tensors="pt").input_ids

            # generate text and specify max length (important to choose appropriate max and min lengths for the evaluation)
            gen_outputs = model.generate(
                input_ids,
                pad_token_id=pad_token_id,
                do_sample=True,
                top_k=40,
                temperature=0.7,
                max_length=50,
                no_repeat_ngram_size=3, 
                num_return_sequences=num_examples
                )
            for output in tqdm(gen_outputs):
                # decode generated tokens
                generated_text = tokenizer.decode(output)

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
    with open("/raid/wald/gpt_results/test_eval.csv", "w") as csv_file:
        csv_writer = csv.DictWriter(csv_file, keys)
        csv_writer.writeheader()
        csv_writer.writerows(examples_list)


if __name__ == "__main__":
    main()
    
    



    

                


