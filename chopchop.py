# pylint: disable=C0301
# pylint: disable=C0413
# pylint: disable=C0411


# Author: Mike Levin
# Date: 2023-04-15
# Description: Chop a journal.md file into individual blog posts.
#   ____ _                  ____ _
#  / ___| |__   ___  _ __  / ___| |__   ___  _ __
# | |   | '_ \ / _ \| '_ \| |   | '_ \ / _ \| '_ \
# | |___| | | | (_) | |_) | |___| | | | (_) | |_) |
#  \____|_| |_|\___/| .__/ \____|_| |_|\___/| .__/
#                   |_|                     |_|

# Define author
AUTHOR = "Mike Levin"

# Debugging
DISABLE_GIT = True
POST_BY_POST = False

#  ___                            _
# |_ _|_ __ ___  _ __   ___  _ __| |_ ___
#  | || '_ ` _ \| '_ \ / _ \| '__| __/ __|
#  | || | | | | | |_) | (_) | |  | |_\__ \
# |___|_| |_| |_| .__/ \___/|_|   \__|___/
#               |_|

import os
import re
import sys
import html
import shlex
import openai
import slugify
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

# YAKE Databases
KWDB = REPO_DATA + "keywords.db"
CATDB = REPO_DATA + "categories.db"

# OpenAI Databases
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
            try:
                # Try to parse the date
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
    # Create the file name from the date and index
    file_name = f"{date_str}-post-{index:04}.md"
    full_path = f"{OUTPUT_PATH}/{file_name}"

    # Initialize per-post variables
    summary = None
    meta_description = None
    keywords = None
    topics = None

    # The OpenAI work is done here
    summary, api_hit = odb(SUMDB, write_summary, slug, post)
    meta_description, api_hit = odb(DESCDB, write_meta, slug, summary)
    topic_text = f"{title} {meta_description} {summary}"
    topics, api_hit = odb(TOPDB, find_topics, slug, topic_text)
    topics = fix_openai_mistakes(topics)
    # headline, api_hit = odb(HEADS, write_headline, slug, topic_text)


    # Write top matter

    # if keywords:
    #     # This process obsoleted by OpenAI API
    #     keywords = [x[0].lower() for x in keywords]
    #     keywords = dehyphen_and_dedupe(keywords)
    #     top_matter.append(f"keywords: {keywords}")

    if topics:
        top_matter.append(f"keywords: {topics}")
        top_matter.append(f"category: {topics.split(', ')[0]}")
    meta_description = scrub_excerpt(meta_description)
    meta_description = neutralize_html(meta_description)
    top_matter.append(f'description: "{meta_description}"')
    top_matter.append(f"layout: post")
    top_matter.append(f"author: {AUTHOR}")
    top_matter.append("---")
    top_matter.extend(content)
    content = top_matter

    # Write to file
    with open(full_path, "w") as f:
        # Flatten list of lines into a single string
        flat_content = "\n".join(content)
        f.writelines(flat_content)
    us_date = date_str.strftime("%m/%d/%Y")
    link = f'<li><a href="/{BLOG}/{slug}/">{title}</a> ({us_date})<br />{meta_description}</li>'
    print(index, full_path)
    if POST_BY_POST and api_hit:
        print()
        print(f"META DESCRIPTION: {meta_description}")
        print()
        print(f"KEYWORDS: {topics}")
        input("Press Enter to continue...")
        print()

    return link


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


def neutralize_html(string):
    """Replace HTML entities with their unicode equivalents."""
    return html.escape(string)


def dehyphen_and_dedupe(keywords):
    """Preserves order of keywords, but removes duplicates and hyphens"""
    keywords = [x.replace("-", " ") for x in keywords]
    # A fascinating way to add to a set within a list comprehension
    seen = set()
    seen_add = seen.add
    keywords = [x for x in keywords if not (x in seen or seen_add(x))]
    return ", ".join(keywords)


def scrub_excerpt(text):
    """Clean up a text for use as an excerpt."""
    # Strip numbered markdown lists from text
    text = re.sub(r"\d+\.\s", "", text)
    # Strip asterisk or hyphen markdown lists from text
    text = re.sub(r"[\*\-]\s", "", text)
    # Replace double quotes with single quotes
    text.replace('"', "'")
    # Flatten wrapped lines
    text = " ".join(text.split("\n"))
    # If a period doesn't have a space after it, add one
    text = re.sub(r"\.(\w)", r". \1", text)
    return text


#   ___                      _    ___   _____
#  / _ \ _ __   ___ _ __    / \  |_ _| |  ___|   _ _ __   ___ ___
# | | | | '_ \ / _ \ '_ \  / _ \  | |  | |_ | | | | '_ \ / __/ __|
# | |_| | |_) |  __/ | | |/ ___ \ | |  |  _|| |_| | | | | (__\__ \
#  \___/| .__/ \___|_| |_/_/   \_\___| |_|   \__,_|_| |_|\___|___/
#       |_|
# OpenAI Functions


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
            "Format as 1-line with keywords in quotes and separated by commas. "
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
            "Write from the perspective of the author. Never say 'The autor'. Say 'I am' or 'I wrote'"
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


def fix_openai_mistakes(keywords):
    """Fix some common mistakes OpenAI makes."""
    # OpenAI might put keywords inside the quotes instead of outside.
    if ',"' in keywords:
        keywords = keywords.split('," "')
        keywords = [x.replace('"', "") for x in keywords]
        keywords = [f'"{x}"' for x in keywords]
        keywords = ", ".join(keywords)
    return keywords


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
