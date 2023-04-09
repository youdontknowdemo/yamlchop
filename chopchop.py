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

add_arg("-f", "--full_path", required=True)
add_arg("-a", "--author", default="Mike Levin")
add_arg("-b", "--blog", default="blog")
add_arg("-o", "--output", default="_posts")

# Parse command line args as CONSTANTS
args = aparser.parse_args()
BLOG = args.blog
OUTPUT = args.output
AUTHOR = args.author
FULL_PATH = args.full_path

# Parse full path into path, repo, and file
parts = FULL_PATH.split("/")
FILE = parts[-1]
REPO = parts[-2] + "/"
PATH = "/".join(parts[:-2]) + "/"
print(f"FULL_PATH: {FULL_PATH}")
print(f"PATH: {PATH}")
print(f"REPO: {REPO}")
print(f"FILE: {FILE}")

# Define Constants
SUMMARY_LENGTH = 350
PARSE_TOKEN = "\n" + "-"*80 + "\n"
OUTPUT_PATH = f"{PATH}{REPO}{OUTPUT}"

REPO_DATA = f"{PATH}{REPO}_data/"
print(f"Processing {FULL_PATH}")

# Create output path if it doesn't exist
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(REPO_DATA).mkdir(parents=True, exist_ok=True)

# Delete old files in output path
for f in os.listdir(OUTPUT_PATH):
    os.remove(f"{OUTPUT_PATH}/{f}")

# Get OpenAI API key
with open ("openai.txt") as fh:
    openai.api_key = fh.readline()


def parse_journal(FULL_PATH):
    """Parse a journal file into posts. Returns a generator of posts."""
    with open(FULL_PATH, "r") as f:
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
            top_matter.append(f"permalink: /{BLOG}/{slug}/")
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
        # Summary needs a lot of cleanup
        summary = " ".join(summary.splitlines()).strip()
        # If a period doesn't have a space after it, add one
        summary = re.sub(r"(\.)(\w)", r"\1 \2", summary)
        # If a summary starts with a dash, remove it
        if summary[:2] == "- ":
            summary = summary[2:]
        # If a summary contains double quotes, replace with single quotes
        summary.replace('"', "'")
        top_matter.append(f"description: {summary}")
    top_matter.append(f"layout: post")
    top_matter.append(f"author: {AUTHOR}")
    top_matter.append("---")
    top_matter.append("")
    top_matter.extend(content)
    content = top_matter

    # Write to file
    print(index, full_path)
    with open(full_path, 'w') as f:
        flat_content = "\n".join(content)
        f.writelines(flat_content)

    fdate = date_str.strftime("%m/%d/%Y")
    if len(summary) > SUMMARY_LENGTH:
        summary = summary[:SUMMARY_LENGTH] + "..."
    link = f"- [{title}](/{BLOG}/{slug}/) {fdate}<br/>\n  {summary}"
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
posts = parse_journal(FULL_PATH)
links = []
for i, post in enumerate(posts):
    link = write_post_to_file(post, i)
    if link:
        links.append(link)

# Write index page
index_page = "\n".join(links)
with open(f"{PATH}{REPO}_includes/posts-main.html", "w", encoding="utf-8") as fh:
    fh.writelines(index_page)
