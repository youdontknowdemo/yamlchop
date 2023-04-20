# Author: Mike Levin
# Date: 2023-04-15
# Description: Chop a journal.md file into individual blog posts.
#   ____ _                  ____ _
#  / ___| |__   ___  _ __  / ___| |__   ___  _ __
# | |   | '_ \ / _ \| '_ \| |   | '_ \ / _ \| '_ \
# | |___| | | | (_) | |_) | |___| | | | (_) | |_) |
#  \____|_| |_|\___/| .__/ \____|_| |_|\___/| .__/
#                   |_|                     |_|
# Example:
# python ~/repos/skite/chopchop.py -f /mnt/c/Users/mikle/repos/hide/MikeLev.in/journal.md

# TO-DO:
# - Speed it up by not opening/closing databases for every page.
# - A better date format on both the index page and articles
# - Check if valid yaml top-matter before git commit.
# - Check resulting pages for broken links.
# - Add a "tags" field to the yaml front matter.
# - Add a "category" field to the yaml front matter.
# - Create category pages

# import os
# import yaml
#
# folder_path = "/home/ubuntu/repos/hide/MikeLev.in/_posts"
#
# for filename in os.listdir(folder_path):
#     if filename.endswith(".md"):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, "r") as file:
#             post_content = file.read()
#             _, yaml_front_matter, _ = post_content.split("---", maxsplit=2)
#             try:
#                 yaml.safe_load(yaml_front_matter)
#                 print(f"{file_path}: YAML front matter is valid.")
#             except yaml.YAMLError as e:
#                 print(f"{file_path}: Error in YAML front matter:", e)

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
DISABLE_GIT = True
POST_BY_POST = True
INTERACTIVE = False
DEBUG = True


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
OUTPUT2_PATH = f"{REPO_DATA}{FILE}"
KEYWORDS_FILE = "{PATH}{REPO}_data/keywords.txt"
CHOPPER = (80 * "-") + "\n"

# OpenAI Databases
ENGINE = "text-davinci-003"
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
if not DEBUG:
    for fh in os.listdir(OUTPUT_PATH):
        delete_me = f"{OUTPUT_PATH}/{fh}"
        os.remove(delete_me)

#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/


def parse_journal(full_path):
    """Parse a journal file into posts. Returns a generator of posts, reverse-order."""
    with open(full_path, "r") as fh:
        print(f"Reading for publishing {full_path}")
        post_str = fh.read()
        pattern = r"-{78,82}\s*\n"
        posts = re.split(pattern, post_str)
        numer_of_posts = len(posts)
        fig(f"{numer_of_posts} posts")
        posts.reverse()  # Reverse so article indexes don't change.
        for post in posts:
            yield post


def rebuild_journal(full_path):
    """I am a forward-running journal parser for inserting front-matter on rebuild."""
    with open(full_path, "r") as fh:
        print(f"Reading for rebuild {full_path}")
        post_str = fh.read()
        pattern = r"-{78,82}\s*\n"
        posts = re.split(pattern, post_str)
        numer_of_posts = len(posts)
        for post in posts:
            yield post


def write_post_to_file(post, index):
    """Write a post to a file. Returns a markdown link to the post."""

    # Parse the post into lines
    lines = post.strip().split("\n")
    date_str, slug = None, None
    top_matter = ["---"]
    top_dict = {}
    content = []
    in_content = False
    api_hit = False

    for i, line in enumerate(lines):
        if i == 0:
            # First line is always the date stamp.
            filename_date = None
            if "date:" not in line:
                # Even date-lines must get a markdown headline hash
                return
            # Parse the date from the line
            date_str = line[len("date: ") :].strip()
            # Parse the date into a datetime object
            adate = parser.parse(date_str).date()
            # Format the date into a string
            date_str = adate.strftime("%Y-%m-%d")
            # Format the date into a filename
            top_matter.append(f"date: {date_str}")
        elif i == 1:
            # Second line is always the title for headline & url
            if line and "title: " in line:
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
            # The duty past here is to continue parsing top matter
            # until we hit the "---" front-matter end-parsing marker.
            # If it's a blank line, it's ambiguous, but we want it to
            # be able to end the top-matter if the "---" is missing.
            # The first behavior split is whether we're in_content or not:
            if in_content:
                # We're in the content, so just add the line
                content.append(line)
            else:
                # We're in the top matter, so add the line
                # and check for the end of the top matter.
                # Each top-matter line is expected to be a yaml-like line.
                # If it's not a yaml-like line, there's 2 possibilities:
                # 1. It's a blank line, which means keep parsing top matter because a field might come next.
                # 2. It's a line of content, which means we're done with top matter.
                if line:
                    # Check if it's a yaml-like line. ":" in line isn't good enough
                    # because a sentence might use a colon. Instead, check for a colon at the end of the first word.
                    first_word = line.split(" ")[0]
                    if first_word.endswith(":"):
                        # It's a yaml-like line, so add it to the top matter
                        top_matter.append(line)
                        ykey, yvalue = line.split(":")
                        ykey = ykey.strip()
                        yvalue = yvalue.strip()
                        # Check if any of these offending characters are in yvalue: " ' [ ] { } , :
                        # Add the key/value pair to the top_dict
                        top_dict[ykey] = q(yvalue)
                    elif line == "---":
                        # It's the end of the top matter, so we're done with top matter
                        in_content = True
                    else:
                        # It's not a yaml-like line, so we're done with top matter
                        # Once we throw this toggle, it's the one time we write "---" to the file.
                        top_matter.append("---")
                        in_content = True
                        content.append(line)
                else:
                    # Blank line, keep parsing top matter
                    top_matter.append(line)

    # Create the file name from the date and index
    file_name = f"{date_str}-post-{index:04}.md"
    out_path = f"{OUTPUT_PATH}/{file_name}"

    # Initialize per-post variables
    summary = None
    if "description" in top_dict:
        description = top_dict["description"]
    else:
        description = None
    if "subhead" in top_dict:
        headline = top_dict["subhead"]
    else:
        headline = None
    if "topics" in top_dict:
        topics = top_dict["keywords"]
    else:
        topics = None

    # The OpenAI work is done here
    # If we already have a description, we don't need to look at the summary:
    if not description:
        summary, api_hit = odb(SUMDB, write_summary, slug, post)
        description, api_hit = odb(DESCDB, write_meta, slug, summary)
        description = chop_last_sentence(description)
        description = neutralize(description)
    if not headline:
        topic_text = f"{title} {description} {summary}"
        headline, api_hit = odb(HEADS, write_headline, slug, topic_text)
        headline = prepare_for_front_matter(headline)
    if not topics:
        topics, api_hit = odb(TOPDB, find_topics, slug, topic_text)
    top_matter.append(f'description: "{description}"')
    top_matter.append(f'subhead: "{headline}"')
    top_matter.append(f"keywords: {topics}")
    top_matter.append(f"author: {AUTHOR}")
    top_matter.append(f"layout: post")
    top_matter.append("---")
    top_matter.extend(content)
    content = top_matter

    # Write to file
    with open(out_path, "w") as f:
        # Flatten list of lines into a single string
        flat_content = "\n".join(content)
        f.writelines(flat_content)
    link = f'<li><a href="/{BLOG}/{slug}/">{title}</a> ({date_str})<br />{description}</li>'
    print(f"Chop {index} {out_path}")
    if POST_BY_POST and api_hit:
        print()
        print(f"Slug: {slug}")
        print()
        print(f"Title: {title}")
        print()
        print(f"Headline: {headline}")
        print()
        print(f"Description: {description}")
        print()
        print(f"Keywords: {topics}")
        print()
        if INTERACTIVE:
            input("Press Enter to continue...")
            print()

    return link


def chop_last_sentence(text):
    """Chop off the last sentence of a string if it is not a sentence ending with punctuation."""
    if not text:
        return None
    if text[-1] in ".?!":
        return text
    if "." in text:
        return text[: text.rfind(".") + 1]
    if "?" in text:
        return text[: text.rfind("?") + 1]
    if "!" in text:
        return text[: text.rfind("!") + 1]
    return text


def neutralize(text):
    """Replace harmful characters with harmless ones."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def prepare_for_front_matter(text):
    """Prepare text for front matter."""
    if not text:
        return None
    text = text.replace('"', "")
    text = text.replace("\n", " ")
    # RegEx replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    text = neutralize(text)
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


def compare_files(file1, file2):
    """Compare two files. Return true of they are the same."""
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        while True:
            byte1 = f1.read(1)
            byte2 = f2.read(1)
            if byte1 != byte2:
                return False
            if not byte1:
                return True


def front_matter_inserter(pre_post):
    """Conditionally insert front matter based on whether and subfields are present."""
    # Step though the text in pre_post line by line.
    # It's a stuffed string, so we'll probably have to split it on newlines.
    # The first line is always the date. If not, return full pre_post.
    # The second line is always the tile. If not, return full pre_post.
    # If the third line is an empty line, we know we need all the front matter, fetched from databases.
    # In that case, we get headlines, descriptions and topics for the slug.
    # We always know the slug because it's the title field put through a deterministic function.
    # The slug is the database key to fetch the headline, description and topics.
    # We also know the author, but that's just a configuration variable.
    # We also know the layout, but that's just a configuration variable.
    # The topics field is actually the keywords field, which we're going to use as tags as well.
    # I'm blending together the concept of categories, tags and keywords.
    # This may change later, but OpenAI chose my keywords, so I'll use the term topics as a catch-all for now.

    # We are setting up a 1-time converson.
    # After that, we'll change the behavior of this code.
    lines = pre_post.split("\n")
    in_content = False
    top_matter = []
    new_post = []
    top_dict = {}
    for i, line in enumerate(lines):
        # Who needs trailing white space anyway?
        line = line.rstrip()
        if i == 0:
            if line.startswith("date:"):
                # This is the most common case and what we want to find.
                top_matter.append(line)
                top_dict["date"] = " ".join(line.split(" ")[1:])
            elif line == "---":
                # We use this when we want to immediately close the front-matter
                # indicating that it's a meta-post and should not be published.
                top_matter.append("---")
                in_content = True
            else:
                # Anything else in the first line, and we should skip it and keep
                # the original post intact.
                print("ERROR: First line of post is not a date.")
                print(pre_post[:1000])
                raise SystemExit()
        elif not in_content:
            # Handles everything still in front-matter.
            first_word = line.split(" ")[0]
            if line == "":
                # Front-matter doesn't need blank lines, so pass.
                pass
            elif line[0] == "#":
                # This indicates the old system before yaml-like top-matter.
                # Let them know this and raise SystemExit.
                print("ERROR: Old-style top matter detected. Please convert to yaml-like top matter.")
                print(line)
                raise SystemExit()
            elif first_word.endswith(":"):
                # We're in the top-matter and we have a yaml-like line:
                top_matter.append(line)
                # Get the field-name:
                field_name = first_word[:-1]
                # Add the field-name and value to the top_dict:
                value = " ".join(line.split(" ")[1:]).strip() 
                top_dict[field_name] = value
            elif i == 1 and "date: " not in line:
                # Probably the last post in the file, but with no date so not a post.
                # This is a good place for to-do lists and unpublished notes, but it
                # needs to allow the rest of the content to be written.
                top_matter.append(line)
                in_content = True
            elif line == "---":
                # We're where we're trying to close the front-matter, but we may not have
                # all the yaml key/value pairs that we need: headline, description, topics.
                # If they're not there, we retreive them from each one's database.
                # This whole process uses a slugified title as a primary key, so we have to have it.
                # If we reached this point and have no title (in top_dict), close the front-matter
                # and start the content.
                top_matter.append("---")
                in_content = True
            if "title" in top_dict:
                top_dict["slug"] = slugify(top_dict["title"])
                # We DO have a title, so we slugify exactly the same way as in write_post_to_file()
                # and use that as the slug.
                # Now we can get the headline, description and topics from the databases.
                # We'll use the slug as the key.
                # The databases in the order we want to check them are: HEADS, DESCDB, TOPDB
                # In time, I will clean this up probably into a function.
                if "headline" not in top_dict:
                    with sqldict(HEADS) as db:
                        if top_dict["slug"] in db:
                            headline = q(db[top_dict['slug']])
                            top_matter.append(f"headline: {headline}")
                            top_dict["headline"] = headline
                if "description" not in top_dict:
                    with sqldict(DESCDB) as db:
                        if top_dict["slug"] in db:
                            description = q(db[top_dict['slug']])
                            top_matter.append(f"description: {description}")
                            top_dict["description"] = description
                if "topics" not in top_dict:
                    with sqldict(TOPDB) as db:
                        if top_dict["slug"] in db:
                            topics = q(db[top_dict['slug']])
                            top_matter.append(f"topics: {topics}")
                            top_dict["topics"] = topics
        else:
            new_post.append(line)
    top_matter.extend(new_post)
    content = top_matter
    return "\n".join(content)


def q(text):
    """Returns text with am embedded double-quote around it and html-escaped content
    if it contains any of the following characters: " ' [ ] { } , :"""
    if any(
        c in text
        for c in ['"', "'", "[", "]", "{", "}", ",", ":"]
    ):
        # If so, html-escape them and quote the yaml value
        text = html.escape(text)
        text = f'"{text}"'
    return text


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
        engine=ENGINE,
        prompt=(
            f"Create a line of comma separated list of keywords to categorize the following text:\n\n{data}\n\n"
            "Do not use extremely broad words like Data, Technology, Blog, Post or Author. "
            "Use words that will be good for site categories, tags and search. "
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
        engine=ENGINE,
        prompt=(
            f"Write a concise and informative meta description for the following text:\n{data}\n\n"
            "...that will entice readers to click through to the blog post. "
            "You wrote this. Write from the first person perspective. Never say 'The author'. '"
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
    """Write an alternate headline for the post."""
    print("Hitting OpenAI API for: subhead")
    response = openai.Completion.create(
        engine=ENGINE,
        prompt=(
            f"Write a short alternative headline for the following post:\n{data}\n\n"
            "The first line of the post is the headline. "
            "Don't be reduntant with the headline. Say something different or better. "
            "You are the one who write this. Write from first person perspective. Never say 'The author'. '"
            "Use only one sentence. "
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
    print("Hitting OpenAI API for: summary")
    chunks = chunk_text(text, chunk_size=4000)
    summarized_text = ""
    for chunk in chunks:
        response = openai.Completion.create(
            engine=ENGINE,
            prompt=(
                f"You wrote this. Write from first person perspective. Please summarize the following text:\n{chunk}\n\n"
                "Summary:"
            ),
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
if not DEBUG:
    all_posts = parse_journal(FULL_PATH)
    links = []
    for i, apost in enumerate(all_posts):
        link = write_post_to_file(apost, i + 1)
        if link:
            links.insert(0, link)

# Rebuild the journal, inserting new front-matter (to-do)

# Delete the old temporary journal from _data
out_file = Path(OUTPUT2_PATH)
if out_file.exists():
    out_file.unlink()

# Rebuild the journal in _data
all_posts = rebuild_journal(FULL_PATH)
with open(OUTPUT2_PATH, "a") as fh:
    print("Rebuilding journal...")
    for i, apost in enumerate(all_posts):
        print(i, end=" ", flush=True)
        if i:
            fh.write(CHOPPER)
        apost = front_matter_inserter(apost)
        fh.write(apost)

# Compare the input and output files. If same, there's been no changes.
fig("Compare files")
files_are_same = compare_files(FULL_PATH, OUTPUT2_PATH)
print(f"Are the input and output files the same? {files_are_same}")
if files_are_same:
    print("Nothing to publish.")
elif not files_are_same and DISABLE_GIT:
    print("Something's getting published, but not to Github.")
if not files_are_same and not DISABLE_GIT:
    print("Something's getting published.")

if not DEBUG:
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
