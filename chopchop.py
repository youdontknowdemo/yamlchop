import os
import re
import openai
import slugify
import datetime
import argparse
from retry import retry
from pathlib import Path
from slugify import slugify
from dateutil import parser
from sqlitedict import SqliteDict as sqldict


# Define command line arguments
aparser = argparse.ArgumentParser()
add_arg = aparser.add_argument
add_arg("-r", "--repo", required=True)
add_arg("-f", "--file", default="journal.md")
add_arg("-a", "--author", default="Mike Levin")
add_arg("-b", "--blog", default="blog")
add_arg("-p", "--path", default="/home/ubuntu/repos/hide/")
add_arg("-o", "--output", default="_posts")

# Parse command line arguments
args = aparser.parse_args()
author = args.author
output = args.output
repo = args.repo
file = args.file
blog = args.blog
path = args.path

# Define Constants
PARSE_TOKEN = "\n" + "-"*80 + "\n"
OUTPUT_PATH = f"{path}{repo}/{output}"
FULL_FILE = f"{path}{repo}/{file}"
REPO_DATA = f"{path}{repo}/_data/"
print(f"Processing {FULL_FILE}")

# Create output path if it doesn't exist
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(REPO_DATA).mkdir(parents=True, exist_ok=True)

# Delete old files in output path
for f in os.listdir(OUTPUT_PATH):
    os.remove(f"{OUTPUT_PATH}/{f}")

# Get OpenAI API key
with open ("openai.txt") as fh:
    openai.api_key = fh.readline()


def parse_journal(FULL_FILE):
    """Parse a journal file into posts. Returns a generator of posts."""
    with open(FULL_FILE, "r") as f:
        post_str = f.read()
        posts = post_str.split(PARSE_TOKEN)
        for post in posts:
            yield post


def write_post_to_file(post, index):
    """Write a post to a file. Returns a markdown link to the post."""
    lines = post.strip().split('\n')

    # Set up per-post variables
    date_str = None
    top_matter = ["---"]
    content = []
    in_content = False

    for i, line in enumerate(lines):
        if i == 0:
            # First line is always the date stamp.
            filename_date = None
            try:
                adate = line[2:]
                date_str = parser.parse(adate).date()
                top_matter.append(f"date: {date_str}")
            except:
                # If we can't parse the date, skip the post
                print(f"Skipping post {index} - no date")
                return

        elif i == 1:
            # Second line is always the title for headline & url
            if line and line[0] == "#" and " " in line:
                title = " ".join(line.split(" ")[1:])
            else:
                return
            # Turn title into slug for permalink
            slug = slugify(title.replace("'", ""))
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

    # Hit OpenAI to get summary and keywords
    summary = None
    with sqldict(REPO_DATA + "descriptions.db") as db:
        if index not in db:
            summary = summarize(post)
            db[index] = summary
            db.commit()
        else:
            summary = db[index]

    # Combine top matter and content
    if summary:
        summary = " ".join(summary.splitlines())
        top_matter.append(f"description: {summary.strip()}")
    top_matter.append(f"layout: post")
    top_matter.append(f"author: {author}")
    top_matter.append("---")
    top_matter.append("")
    top_matter.extend(content)
    content = top_matter

    # Write to file
    print(index, full_path)
    with open(full_path, 'w') as f:
        flat_content = "\n".join(content)
        f.writelines(flat_content)

    link = f"- [{title}](/{blog}/{slug}/) {date_str}\n  {summary[:200]}"
    return link


def chunk_text(text, chunk_size=4000):
    """Split a text into chunks of a given size."""
    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = start_idx + chunk_size
        if end_idx >= len(text):
            end_idx = len(text)
        chunk = text[start_idx:end_idx]
        chunks.append(chunk)
        start_idx = end_idx
    return chunks


@retry(Exception, delay=1, backoff=2, max_delay=60)
def summarize(text):
    """Summarize a text using OpenAI's API."""
    chunks = chunk_text(text, chunk_size=4000)
    summarized_text = ""
    for chunk in chunks:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=(f"Please summarize the following text:\n{chunk}\n\n"
                    "Summary:"),
            temperature=0.5,
            max_tokens=100,
            n = 1,
            stop=None
        )
        summary = response.choices[0].text.strip()
        summarized_text += summary
        summarized_text = " ".join(summarized_text.splitlines())
    return summarized_text.strip()


# Parse the journal file
posts = parse_journal(FULL_FILE)
links = []
for i, post in enumerate(posts):
    link = write_post_to_file(post, i)
    if link:
        links.append(link)

# Write index page
index_page = "\n".join(links)
with open(f"{path}{repo}/_includes/posts-main.html", "w", encoding="utf-8") as fh:
    fh.writelines(index_page)
