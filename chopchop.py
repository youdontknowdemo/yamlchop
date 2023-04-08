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
add_arg("-a", "--author", default="Mike Levin")
add_arg("-b", "--blog", default="blog")
add_arg("-p", "--path", default="/home/ubuntu/repos/hide/")
add_arg("-o", "--output", default="_test")

args = aparser.parse_args()
author = args.author
output = args.output
repo = args.repo
file = args.file
blog = args.blog
path = args.path

# Constants
PARSE_TOKEN = "\n" + "-"*80 + "\n"
OUTPUT_PATH = f"{path}{repo}/{output}"
FULL_FILE = f"{path}{repo}/{file}"
print(f"Processing {FULL_FILE}")

# Check if output path exists and create it if not
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
# Delete all files from output path
for f in os.listdir(OUTPUT_PATH):
    os.remove(f"{OUTPUT_PATH}/{f}")


def parse_journal(FULL_FILE):
    """Parse a journal file into posts. Returns a generator of posts."""
    with open(FULL_FILE, "r") as f:
        post_str = f.read()
        posts = post_str.split(PARSE_TOKEN)
        for post in posts:
            yield post


def write_post_to_file(post, index):
    lines = post.strip().split('\n')

    # Set up per-post variables
    date_str = None
    top_matter = ["---"]
    content = []
    in_content = False

    for i, line in enumerate(lines):
        if i == 0:
            # First line is always date
            filename_date = None
            try:
                adate = line[2:]
                date_str = parser.parse(adate).date()
                top_matter.append(f"date: {date_str}")
            except:
                # If no date, skip post
                print(f"Skipping post {index} - no date")
                return

        elif i == 1:
            # Second line is always headline begginning with #
            if line and line[0] == "#" and " " in line:
                title = " ".join(line.split(" ")[1:])
            else:
                return
            slug = slugify(title)
            top_matter.append(f"title: {title}")
            top_matter.append(f"slug: {slug}")
            top_matter.append(f"permalink: /{blog}/{slug}/")
        else:
            # Subsequent lines are either top matter or content
            if not line:
                # Blank line means we're done with top matter
                in_content = True
                pass
            if in_content:
                content.append(line)
            else:
                # Top matter
                pass
    file_name = f"{date_str}-post-{index:04}.md"
    full_path = f"{OUTPUT_PATH}/{file_name}"

    # Combine top matter and content
    top_matter.append(f"layout: post")
    top_matter.append(f"author: {author}")
    top_matter.append("---")
    top_matter.append("")
    top_matter.append(f"# {title}")
    top_matter.extend(content)
    content = top_matter

    print(full_path)
    with open(full_path, 'w') as f:
        flat_content = "\n".join(content)
        f.writelines(flat_content)


posts = parse_journal(FULL_FILE)
for i, post in enumerate(posts):
    write_post_to_file(post, i)

