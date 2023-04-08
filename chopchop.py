import os
import re
import slugify
import datetime
import argparse
from pathlib import Path
from slugify import slugify
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

    # Example top matter
    """
    ---
    layout: post
    title: "Embedding YouTube Video Thumbnails on Github.io Pages"
    description: "Embedding YouTube Video Thumbnails on Github.io Pages"
    author: "Mike Levin"
    slug: embedding-youtube-video-thumbnails-on-github-io-pages
    permalink: /blog/embedding-youtube-video-thumbnails-on-github-io-pages/
    ---
    """

    date_str = None
    content = []
    in_content = False
    for i, line in enumerate(lines):
        if not i:
            # First line is always date
            filename_date = None
            try:
                adate = line[2:]
                date_str = parser.parse(adate).date()
            except:
                break
        else:
            # Subsequent lines are either top matter or content
            if not line:
                # Blank line means we're done with top matter
                in_content = True
                pass
            if in_content:
                content.append(line)
    file_name = f"{date_str}-post-{index:04}.md"
    full_path = f"{OUTPUT_PATH}/{file_name}"

    print(full_path)
    with open(full_path, 'w') as f:
        flat_content = "\n".join(content)
        f.writelines(flat_content)



posts = parse_journal(full_file)
for i, post in enumerate(posts):
    write_post_to_file(post, i+1)

