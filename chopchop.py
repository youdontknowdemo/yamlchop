import os
import re
import slugify
import datetime
import argparse
from pathlib import Path
from dateutil import parser


# CLI args
aparser = argparse.ArgumentParser()
add_arg = aparser.add_argument
add_arg("-r", "--repo", required=True)
add_arg("-f", "--file", default="journal.md")
args = aparser.parse_args()
repo = args.repo
file = args.file

# Constants
COMMON_PATH = "/home/ubuntu/repos/hide/"
PARSE_TOKEN = "\n" + "-"*80 + "\n"
OUTPUT_PATH = f"{COMMON_PATH}{repo}/_test"

# Set full file path and show user
full_file = f"{COMMON_PATH}{repo}/{file}"
print(f"Processing {full_file}")

# Check if output path exists and create it if not
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)


def parse_journal(full_file):
    """Parse a journal file into posts. Returns a generator of posts."""
    with open(full_file, "r") as f:
        post_str = f.read()
        posts = post_str.split(PARSE_TOKEN)
        for post in posts:
            yield post


def write_post_to_file(post, index):
    lines = post.strip().split('\n')
    top_matter = ''
    content = ''
    title = ''
    permalink = ''
    date_str = None
    for line in lines:
        filename_date = None
        try:
            adate = line[2:]
            date_str = parser.parse(adate).date()
        except:
            pass
    if date_str:
        file_name = f"{date_str}-post-{index:04}.md"
        full_path = f"{OUTPUT_PATH}/{file_name}"
        print(full_path)
        with open(full_path, 'w') as f:
            f.writelines(post)



posts = parse_journal(full_file)
for i, post in enumerate(posts):
    write_post_to_file(post, i+1)

