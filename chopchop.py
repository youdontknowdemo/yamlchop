# Author: Mike Levin
# Date: 2023-04-15
# Description: Chop a journal.md file into individual blog posts.
# To-Do: Check resulting pages for broken links.
#   ____ _                  ____ _
#  / ___| |__   ___  _ __  / ___| |__   ___  _ __
# | |   | '_ \ / _ \| '_ \| |   | '_ \ / _ \| '_ \
# | |___| | | | (_) | |_) | |___| | | | (_) | |_) |
#  \____|_| |_|\___/| .__/ \____|_| |_|\___/| .__/
#                   |_|                     |_|

import os
import re
import sys
import html
import shlex
import openai
import datetime
import argparse
import pandas as pd
from time import sleep
from retry import retry
from pathlib import Path
from slugify import slugify
from pyfiglet import Figlet
from dateutil import parser
from subprocess import Popen, PIPE
from sqlitedict import SqliteDict as sqldict


AUTHOR = "Mike Levin"

# Debugging
DISABLE_GIT = False
POST_BY_POST = False
DEBUG = False


# Globals


# Load function early so we can start showing figlets.
def fig(text):
    """Print a figlet."""
    f = Figlet()
    print(f.renderText(text))
    sleep(0.5)


fig("ChopChop")

#  ____                          _
# |  _ \ __ _ _ __ ___  ___     / \   _ __ __ _ ___
# | |_) / _` | '__/ __|/ _ \   / _ \ | '__/ _` / __|
# |  __/ (_| | |  \__ \  __/  / ___ \| | | (_| \__ \
# |_|   \__,_|_|  |___/\___| /_/   \_\_|  \__, |___/
#                                         |___/

# Define command line arguments
aparser = argparse.ArgumentParser()
add_arg = aparser.add_argument

# Example:
# python ~/repos/skite/chopchop.py -f /mnt/c/Users/mikle/repos/hide/MikeLev.in/journal.md

# Use in a vim or NeoVim macro from .vimrc or init.vim like this:
# let @p = ":execute '!python ~/repos/skite/chopchop.py -f ' . expand('%:p')"
# Or in interactive mode in NeoVim using it's :terminal command:
# let @p = ":terminal 'python ~/repos/skite/chopchop.py -f ' .expand('%:p')"

add_arg("-f", "--full_path", required=True)
add_arg("-a", "--author", default=AUTHOR)
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
REPO = parts[-2] + "/"
fig(REPO)  # Print the repo name
FILE = parts[-1]
PATH = "/".join(parts[:-2]) + "/"
GIT_EXE = "/usr/bin/git"
OUTPUT_PATH = f"{PATH}{REPO}{OUTPUT}"
REPO_DATA = f"{PATH}{REPO}_data/"
KEYWORDS_FILE = "{PATH}{REPO}_data/keywords.txt"

# OpenAI DatabaseTrue
SUMDB = REPO_DATA + "summaries.db"
DESCDB = REPO_DATA + "descriptions.db"
TOPDB = REPO_DATA + "topics.db"
HEADS = REPO_DATA + "headlines.db"

# Print out constants
print(f"REPO: {REPO}")
print(f"FULL_PATH: {FULL_PATH}")
print(f"PATH: {PATH}")
print(f"FILE: {FILE}")

# Create output path if it doesn't exist
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(REPO_DATA).mkdir(parents=True, exist_ok=True)

with open("/home/ubuntu/repos/skite/openai.txt") as fh:
    # Get OpenAI API key
    openai.api_key = fh.readline()

# Delete old files in output path
for fh in os.listdir(OUTPUT_PATH):
    delete_me = f"{OUTPUT_PATH}/{fh}"
    os.remove(delete_me)

#  ____        __ _              _____                 _   _
# |  _ \  ___ / _(_)_ __   ___  |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | | | |/ _ \ |_| | '_ \ / _ \ | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# | |_| |  __/  _| | | | |  __/ |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |____/ \___|_| |_|_| |_|\___| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/


def parse_journal(full_path):
    """Parse a journal file into posts. Returns a generator of posts."""
    with open(full_path, "r") as fh:
        print(f"Reading {full_path}")
        post_str = fh.read()
        pattern = r"-{78,82}\s*\n"
        posts = re.split(pattern, post_str)
        numer_of_posts = len(posts)
        fig(f"{numer_of_posts} posts")
        posts.reverse()  # Reverse so article indexes don't change.
        for post in posts:
            yield post


def write_post_to_file(post, index):
    """Write a post to a file. Returns a markdown link to the post."""

    # Parse the post into lines
    lines = post.strip().split("\n")
    date_str, slug = None, None
    top_matter = ["---"]
    content = []
    in_content = False
    api_hit = False

    for i, line in enumerate(lines):
        if i == 0:
            # First line is always the date stamp.
            filename_date = None
            if "#" not in line:
                # Even date-lines must get a markdown headline hash
                return
            # Parse the date from the line
            date_str = line[line.rfind("#") + 1 :].strip()
            # Parse the date into a datetime object
            adate = parser.parse(date_str).date()
            # Format the date into a string
            date_str = adate.strftime("%Y-%m-%d")
            # Format the date into a filename
            top_matter.append(f"date: {date_str}")
        elif i == 1:
            # Second line is always the title for headline & url
            if line and line[0] == "#" and " " in line:
                title = " ".join(line.split(" ")[1:])
                title = title.replace(":", "")
            else:
                return
            # Turn title into slug for permalink
            slug = slugify(title.replace("'", ""))
            top_matter.append(f'title: "{title}"')
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
    # Create the file name from the date and index
    file_name = f"{date_str}-post-{index:04}.md"
    out_path = f"{OUTPUT_PATH}/{file_name}"

    # Initialize per-post variables
    summary = None
    meta_description = None
    keywords = None
    topics = None

    # The OpenAI work is done here
    summary, api_hit = odb(SUMDB, write_summary, slug, post)
    meta_description, api_hit = odb(DESCDB, write_meta, slug, summary)
    topic_text = f"{title} {meta_description} {summary}"
    headline, api_hit = odb(HEADS, write_headline, slug, topic_text)
    headline = prepare_for_front_matter(headline)
    top_matter.append(f'subhead: "{headline}"')
    topics, api_hit = odb(TOPDB, find_topics, slug, topic_text)
    topics = check_pat(topics)
    if topics:
        top_matter.append(f"keywords: {topics}")
    # Write top matter
    if DEBUG:
        if topics:
            top_matter.append(f"category: {topics.split(', ')[0][1:-1]}")
            top_matter.append(f"layout: post")
    meta_description = html.escape(meta_description)
    top_matter.append(f'description: "{meta_description}"')
    top_matter.append(f"author: {AUTHOR}")
    top_matter.append("---")
    top_matter.extend(content)
    content = top_matter

    # Write to file
    with open(out_path, "w") as f:
        # Flatten list of lines into a single string
        flat_content = "\n".join(content)
        f.writelines(flat_content)
    link = f'<li><a href="/{BLOG}/{slug}/">{title}</a> ({date_str})<br />{meta_description}</li>'
    print(f"Chop {index} {out_path}")
    if POST_BY_POST and api_hit:
        print()
        print(f"Title: {title}")
        print(f"Headline: {headline}")
        print()
        input("Press Enter to continue...")
        print()

    return link


def check_pat(text):
    """Checks for good quote-and-comma string format."""
    if not text:
        return None
    inside_comma = r'(^("[^"]+",\s*)+"[^"]+"$)|(^"[^",]*"$)'
    if re.match(inside_comma, text, re.MULTILINE) and "\n" not in text:
        return text
    else:
        return None


def prepare_for_front_matter(text):
    """Prepare text for front matter."""
    text = text.replace('"', "")
    text = text.replace("\n", " ")
    # RegEx replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    text = html.escape(text)
    text = text.strip()
    return text


def git(cwd, line_command):
    """Run a Linux git command."""
    cmd = [GIT_EXE] + shlex.split(line_command)
    print(f"COMMAND: <<{shlex.join(cmd)}>>")
    process = Popen(
        args=cmd,
        cwd=cwd,
        stdout=PIPE,
        stderr=PIPE,
        shell=False,
        bufsize=1,
        universal_newlines=True,
    )
    flush(process.stdout)
    flush(process.stderr)


def flush(std):
    """Flush a stream."""
    for line in std:
        line = line.strip()
        if line:
            print(line)
            sys.stdout.flush()


#   ___                      _    ___   _____
#  / _ \ _ __   ___ _ __    / \  |_ _| |  ___|   _ _ __   ___ ___
# | | | | '_ \ / _ \ '_ \  / _ \  | |  | |_ | | | | '_ \ / __/ __|
# | |_| | |_) |  __/ | | |/ ___ \ | |  |  _|| |_| | | | | (__\__ \
#  \___/| .__/ \___|_| |_/_/   \_\___| |_|   \__,_|_| |_|\___|___/
#       |_|
# OpenAI Functions


def odb(DBNAME, afunc, slug, full_text):
    """Record OpenAI API hits in a database."""
    api_hit = False
    with sqldict(DBNAME) as db:
        if slug not in db:
            result = afunc(full_text)  # Hits OpenAI API
            db[slug] = result
            db.commit()
            api_hit = True
        else:
            result = db[slug]
    return result, api_hit


@retry(Exception, delay=1, backoff=2, max_delay=60)
def find_topics(data):
    """Returns top keywords and main category for text."""
    print("Hitting OpenAI API for: topics")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=(
            f"Create a list of keywords for the following text:\n\n{data}\n\n...in order to categorize the blog post. "
            "Do not use extremely broad words like Data, Technology, Blog, Post or Author "
            "Use the best keyword for a single-category topic-label as the first keyword in the list. "
            "Format as 1-line with keywords separated by commas. "
            "Do not use quotes around keywords. "
            "\nKeywords:\n\n"
        ),
        temperature=0.5,
        max_tokens=100,
        n=1,
        stop=None,
    )
    topics = response.choices[0].text.strip()
    return topics


@retry(Exception, delay=1, backoff=2, max_delay=60)
def write_meta(data):
    """Write a meta description for a post."""
    print("Hitting OpenAI API for: meta descriptions")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=(
            f"Write a concise and informative meta description for the following text:\n{data}\n\n"
            "...that will entice readers to click through to the blog post. "
            "Write from the perspective of the author. Never say 'The author'. Say 'I am' or 'I wrote'"
            "Always finish sentences. Never chop off a sentence. End in a period."
            "\nSummary:\n\n"
        ),
        temperature=0.5,
        max_tokens=100,
        n=1,
        stop=None,
    )
    meta_description = response.choices[0].text.strip()
    return meta_description


@retry(Exception, delay=1, backoff=2, max_delay=60)
def write_headline(data):
    """Write a better headlie for post."""
    print("Hitting OpenAI API for: subhead")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=(
            f"Write a short alternative headline for the following post:\n{data}\n\n"
            "Don't be reduntant with the first line of the blog post. "
            "Use only one sentence. "
            "Write from the perspective of the author. Never say 'The author'. Say 'I am' or 'I wrote'"
            "\nHeadline:\n\n"
        ),
        temperature=0.5,
        max_tokens=100,
        n=1,
        stop=None,
    )
    headline = response.choices[0].text.strip()
    return headline


@retry(Exception, delay=1, backoff=2, max_delay=60)
def write_summary(text):
    """Summarize a text using OpenAI's API."""
    chunks = chunk_text(text, chunk_size=4000)
    summarized_text = ""
    for chunk in chunks:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=(f"Please summarize the following text:\n{chunk}\n\n" "Summary:"),
            temperature=0.5,
            max_tokens=100,
            n=1,
            stop=None,
        )
        summary = response.choices[0].text.strip()
        summarized_text += summary
        summarized_text = " ".join(summarized_text.splitlines())
    return summarized_text.strip()


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


#  ____  _ _                _                              _
# / ___|| (_) ___ ___      | | ___  _   _ _ __ _ __   __ _| |
# \___ \| | |/ __/ _ \  _  | |/ _ \| | | | '__| '_ \ / _` | |
#  ___) | | | (_|  __/ | |_| | (_) | |_| | |  | | | | (_| | |
# |____/|_|_|\___\___|  \___/ \___/ \__,_|_|  |_| |_|\__,_|_|
fig("Slice Journal")


# Parse the journal file
posts = parse_journal(FULL_PATH)
links = []
for i, post in enumerate(posts):
    link = write_post_to_file(post, i + 1)
    if link:
        links.insert(0, link)

# Add countdown ordered list to index page
links.insert(0, f'<ol start="{len(links)}" reversed>')
links.append("</ol>")
# Write index page
index_page = "\n".join(links)
# Write out list of posts
with open(f"{PATH}{REPO}_includes/post_list.html", "w", encoding="utf-8") as fh:
    fh.writelines(index_page)


if not DISABLE_GIT:
    # Git commands
    fig("Git Push")
    here = f"{PATH}{REPO}"
    git(here, "add _posts/*")
    git(here, "add _includes/*")
    git(here, "add assets/images/*")
    git(here, f'commit -am "Pushing {REPO} to Github..."')
    git(here, "push")

fig("Done")
