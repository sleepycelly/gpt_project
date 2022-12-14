import json
from collections import defaultdict


# use bos and eos tokens, concatenate post and comment and add a special/self-created token in the middle
def create_entry(post, reply):
    separation_token = "<|sepoftext|>"

    return "<|endoftext|>" + post + separation_token + reply + "<|endoftext|>\n"


def dismiss_post(post):
    return post["language"] != "english"

def main():
    with open("/raid/wald/gpt_data/relnet/christianchat.json", mode="r", encoding="utf-8", errors="ignore") as json_file:
        posts = json.load(json_file)['data']
    
    # Create defaultdict to place posts of the same thread in a list
    post_dict = defaultdict(list)

    for post in posts:
        if not dismiss_post(post):
            # Collect all posts of a particular threads in post_dict and keep 'no' to sort later
            thread_id = post["thread"]
            post_dict[thread_id].append((post["no"], post["text"]))
    
    with open("/raid/wald/gpt_data/train/christianchat.txt", "w") as output_file:
        for thread in post_dict.values():
            sorted_thread = sorted(thread, key=lambda x: x[0])
            # Remove 'no' from list
            thread = [y for _, y in sorted_thread]
            # Create tuples of consecutive/neighboring posts
            for post, reply in zip(thread, thread[1:]):
                output_file.write(create_entry(post, reply))


if __name__ == "__main__":
    main()