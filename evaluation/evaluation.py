import csv
import re
from os import listdir, path

from detoxify import Detoxify
from transformers import AutoModelForCausalLM, GPT2Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main():
    models = listdir("/raid/wald/gpt_models")

    # define how many examples to generate with every prompt
    num_examples = 1

    # open prompt file
    with open("/raid/wald/gpt_data/prompt_list.csv") as prompt_file:
         prompt_reader = csv.DictReader(prompt_file)
    
    # initialize sentiment classifier
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # intialize list to save the generated examples with their sentiment and toxicity scores
    examples_list = []

    for model_name in models:
        # load model and tokenizer
        file_path = path.join("/raid/wald/gpt_models", model_name)
        model = AutoModelForCausalLM.from_pretrained(file_path)
        tokenizer = GPT2Tokenizer.from_pretrained(file_path)
        for row in prompt_reader:
            bias = row["bias"]
            # 'encode' prompt with the tokenizer
            prompt = row["prompt"]
            prompt_with_token = prompt + "<|sepoftext|>"
            input_ids = tokenizer(prompt_with_token, return_tensors="pt").input_ids

                # generate text and specify max length (important to choose appropriate max and min lengths for the evaluation)
                gen_outputs = model.generate(
                    input_ids,
                    do_sample=True,
                    min_length=20,
                    max_length=100,
                    num_return_sequences=100
                    )
                for output in gen_outputs:
                    # decode generated tokens
                    gen_text = tokenizer.decode(output)

                    # dismiss prompt and special tokens
                    pattern = "\<\|sepoftext\|\>(.*?)\<\|endoftext\|\>"
                    substring = re.search(pattern, generated_text).group(1)

                    if substring is None:
                        # no special tokens or no text? error handling
                        print(generated_text)
                    else:
                        current_example = {"model": model_name, "bias_category": bias, "prompt": prompt, "generated_text": substring}
                        # determine sentiment and toxicity scores
                        sentiment = sentiment_analyzer.polarity_scores(substring)
                        toxicity = Detoxify('original').predict(substring)
                        # add sentiment and toxicity to dictionary
                        current_example.update(sentiment)
                        current_example.update(toxicity)

                        examples_list.append(current_example)
    
    keys = examples_list[0].keys()
    with open("students.csv", "w") as csv_file:
        csv_writer = csv.DictWriter(csv_file, keys)
        csv_writer.writeheader()
        csv_writer.writerows(examples_list)


if __name__ == "__main__":
    main()
    
    



    

                


