import csv
import html
import re

import emoji
from bs4 import BeautifulSoup
from markdown import markdown


# use bos and eos tokens, concatenate post and comment and add a special/self-created token in the middle
def create_entry(post_text, comment_text):
    separation_token = "<|sepoftext|>"

    post_text = clean_text(post_text)
    comment_text = clean_text(comment_text)

    return "<|endoftext|>" + post_text + separation_token + comment_text + "<|endoftext|>\n"


def clean_text(text):
    # convert (html) escaped characters such as '&amp;' with their Unicode equivalent 
    text = html.unescape(text)
    # convert from markdown to html to remove markdown characters
    text = markdown(text)
    # parse html
    soup = BeautifulSoup(text, features="lxml")
    # remove emojis
    text = emoji.replace_emoji(soup.text)
    # replace newlines with spaces for readability
    text = re.sub(r"[\n]", " ", text)
    return text

def dismiss_post(post_text, author):
    if author in ["AutoModerator", "WSBVoteBot"]:
        return True
    
    if post_text in ["[removed]", "[deleted]", ""]:
        return True

    return False

def main():
    # create post dictionary
    post_dict = {}

    # open post-csv file and create dictionary with post-id as key and concatenated text and title as value
    with open("/raid/gpt_data/reddit_data/wallstreetbets_posts.csv", encoding="utf-8") as post_csv:
        post_reader = csv.DictReader(post_csv)
        for row in post_reader:
            if not dismiss_post(row["selftext"], row["author"]):
                # for empty selftext only use title
                if row["selftext"] == "":
                    post_dict[row["id"]] = row["title"]
                # concatenate title and selftext if selftext not empty
                else:
                    post_text = row["title"] + " " + row["selftext"]
                    post_dict[row["id"]] = post_text
           
    # open comment-csv file and while looping through fill list of post-comment-matches
    with open("/raid/gpt_data/reddit_data/wallstreetbets_comments.csv", encoding="utf-8") as comment_csv:
        with open("/raid/gpt_data/train/wallstreetbets.txt", "w") as output_file:
            comment_reader = csv.DictReader(comment_csv)
            for row in comment_reader:
                # scrap removed and deleted posts and bot posts
                if not dismiss_post(row["body"], row["author"]):
                    # get the corresponding post's id from the url
                    post_id = row["parent_id"].split("_")[1]
                    # if the corresponding post is in the dictionary (therefore valid) create an example for training the model consisting of post and comment
                    if post_id in post_dict.keys():
                        output_file.write(create_entry(post_dict[post_id], row["body"]))
    
    

if __name__ == "__main__":
    main()
