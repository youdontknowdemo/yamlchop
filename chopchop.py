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
import yaml
import html
import shlex
import openai
import shutil
import datetime
import argparse
import pandas as pd
from time import sleep
from retry import retry
from pathlib import Path
from slugify import slugify
from pyfiglet import Figlet
from dateutil import parser
from rich.table import Table
from datetime import datetime
from rich.console import Console
from subprocess import Popen, PIPE
from nltk.stem import WordNetLemmatizer
from sqlitedict import SqliteDict as sqldict
from collections import Counter, defaultdict



AUTHOR = "Mike Levin"

# Debugging
DISABLE_GIT = False
POST_BY_POST = True
INTERACTIVE = False
DEBUG = False


# Load function early so we can start showing figlets.
def fig(text, description=None):
    """Print a figlet and optional description with momentary delay.
    This is good to explain something qick before it scrolls across
    console output."""
    f = Figlet()
    print(f.renderText(text))
    if description:
        print(description)
    sleep(0.5)


fig("ChopChop", "A radical new blogging system based on 1-file for life")

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
INCLUDES = f"{PATH}{REPO}_includes/"
CHOPPER = (80 * "-") + "\n"
CATEGORY_PAGE = f"{REPO}category.md"

# OpenAI Databases
ENGINE = "text-davinci-003"
SUMDB = REPO_DATA + "summaries.db"
DESCDB = REPO_DATA + "descriptions.db"
KWDB = REPO_DATA + "keywords.db"
HEADS = REPO_DATA + "headlines.db"

# Print out constants
print(f"REPO: {REPO}")
print(f"FULL_PATH: {FULL_PATH}")
print(f"PATH: {PATH}")
print(f"FILE: {FILE}")

# Create output path if it doesn't exist
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(REPO_DATA).mkdir(parents=True, exist_ok=True)

with open("/home/ubuntu/repos/skite/openai.txt", "r") as fh:
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
            # adate = parser.parse(date_str).date()
            # adate = datetime.strptime(date_str, '%a %b %d, %Y').date()
            # adate = datetime.strptime(date_str.strip('"'), '%a %b %d, %Y').date()
            date_str = date_str.strip('"')  # remove quotes
            adate = datetime.strptime(date_str, '%a %b %d, %Y').date()
            # Format the date into a string
            date_str = adate.strftime("%Y-%m-%d")
            # Format the date into a filename
            top_dict["date"] = date_str
        elif i == 1:
            # Second line is always the title for headline & url
            if line and "title: " in line:
                title = " ".join(line.split(" ")[1:])
                title = title.strip('"')  # remove quotes
            else:
                return
            # Turn title into slug for permalink
            slug = slugify(title.replace("'", ""))
            top_dict["title"] = title
            top_dict["slug"] = slug
            top_dict["permalink"] = f"/{BLOG}/{slug}/"
        else:
            # We are past the first two lines.
            # The duty past here is to continue parsing top matter
            # until we hit the "---" front-matter end-parsing marker.
            # If it's a blank line, it's ambiguous, but we want it to
            # be able to end the top-matter if the "---" is missing.
            # The first behavior split is whether we're in_content or not:
            if in_content:
                # We're in the content, so just add the line
                content.append(line)
            else:
                # We're still in the top matter, so add the line
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
                        #top_matter.append(line)
                        # Parse the yaml-like line into a key/value pair
                        ykey = first_word[:-1]
                        yvalue = " ".join(line.split(" ")[1:])
                        # Check if any of these offending characters are in yvalue: " ' [ ] { } , :
                        # Add the key/value pair to the top_dict
                        top_dict[ykey] = yvalue
                    elif line == "---":
                        # It's the end of the top matter, so we're done with top matter
                        in_content = True
                    else:
                        # It's not a yaml-like line, so we're done with top matter
                        # Once we throw this toggle, it's the one time we write "---" to the file.
                        #top_matter.append("---")
                        in_content = True

    # Create the file name from the date and index
    file_name = f"{date_str}-post-{index:04}.md"
    out_path = f"{OUTPUT_PATH}/{file_name}"

    # Initialize per-post variables
    summary = None
    if "description" in top_dict:
        description = chop_last_sentence(top_dict["description"])
    else:
        description = None
    if "subhead" in top_dict:
        headline = top_dict["subhead"]
    else:
        headline = None
    if "keywords" in top_dict:
        keywords = top_dict["keywords"]
    else:
        keywords = None

    # The OpenAI work is done here
    # If we already have a description, we don't need to look at the summary:
    if not description:
        summary, api_hit = odb(SUMDB, write_summary, slug, post)
        description, api_hit = odb(DESCDB, write_meta, slug, summary)
        description = chop_last_sentence(description)
    if not headline:
        keyword_text = f"{title} {description} {summary}"
        headline, api_hit = odb(HEADS, write_headline, slug, keyword_text)
        headline = prepare_for_front_matter(headline)
    if not keywords:
        keywords, api_hit = odb(KWDB, find_keywords, slug, keyword_text)
    
    for key, value in top_dict.items():
        top_matter.append(f"{key}: {q(value)}")
    top_matter.append(f"author: {AUTHOR}")
    top_matter.append(f"layout: post")
    top_matter.append("---")
    top_matter.extend(content)
    content = top_matter
    flat_content = "\n".join(content)

    # Catch bad YAML format before it even becomes a file.
    try:
        yaml.safe_load(flat_content)
    except yaml.YAMLError as e:
        fig("YAML Error", "<< Figlet it out: >>\n")
        print(f"Error in YAML front matter:", e)
        print(flat_content)
        print(f"YAML front matter on post with title: {title}")
        raise SystemExit()

    # Write to file
    with open(out_path, "w") as f:
        # Flatten list of lines into a single string
        f.writelines(flat_content)

    link = f'<li><a href="/{BLOG}/{slug}/">{title}</a> ({convert_date(date_str)})<br />{description}</li>'
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
        print(f"Keywords: {keywords}")
        print()
        if INTERACTIVE:
            input("Press Enter to continue...")
            print()

    return link


def convert_date(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return date_obj.strftime("%a %b %d, %Y")


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


def prepare_for_front_matter(text):
    """Prepare text for front matter."""
    if not text:
        return None
    text = text.replace('"', "")
    text = text.replace("\n", " ")
    # RegEx replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
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
    # In that case, we get headlines, descriptions and keywords for the slug.
    # We always know the slug because it's the title field put through a deterministic function.
    # The slug is the database key to fetch the headline, description and keywords.
    # We also know the author, but that's just a configuration variable.
    # We also know the layout, but that's just a configuration variable.
    # The keywords field is actually the keywords field, which we're going to use as tags as well.
    # I'm blending together the concept of categories, tags and keywords.
    # This may change later, but OpenAI chose my keywords, so I'll use the term keywords as a catch-all for now.

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
                top_dict["date"] = " ".join(line.split(" ")[1:])
            elif line == "---":
                # We use this when we want to immediately close the front-matter
                # indicating that it's a meta-post and should not be published.
                top_matter.append("---")
                in_content = True
            else:
                # Anything else in the first line, and we should skip it and keep
                # the original post intact.
                return pre_post
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
            elif line == "---":
                # We're where we're trying to close the front-matter, but we may not have
                # all the yaml key/value pairs that we need: headline, description, keywords.
                # If they're not there, we retreive them from each one's database.
                # This whole process uses a slugified title as a primary key, so we have to have it.
                # If we reached this point and have no title (in top_dict), close the front-matter
                # and start the content.
                in_content = True
            elif first_word.endswith(":"):
                # We're in the top-matter and we have a yaml-like line:
                # Get the field-name:
                # print(first_word)
                field_name = first_word[:-1]
                # Add the field-name and value to the top_dict:
                value = " ".join(line.split(" ")[1:]).strip() 
                # print(f"field_name: {field_name}, value: {value}")
                top_dict[field_name] = value
            else:
                # Anything else is an error.
                print("ERROR: Unhandled case in front_matter_inserter()")
                print(line)
                raise SystemExit()
            # if top_dict: 
            #     print(f"top_dict: {top_dict}")
            if "title" in top_dict:
                slug = slugify(top_dict["title"].replace("'", ""))
                # slug = slugify(top_dict["title"])
                # top_dict["slug"] = slug
                # print(f"top_dict['slug']: {top_dict['slug']}")
                # print(f"top_dict: {top_dict}")

                # We DO have a title, so we slugify exactly the same way as in write_post_to_file()
                # and use that as the slug.
                # Now we can get the headline, description and keywords from the databases.
                # We'll use the slug as the key.
                # The databases in the order we want to check them are: HEADS, DESCDB, KWDB
                # In time, I will clean this up probably into a function.
                # print(f"top_dict: {top_dict}")
                # print("headline" in top_dict.keys())
                if "headline" not in top_dict:
                    with sqldict(HEADS) as db:
                        # print("Getting headline from db")
                        if slug in db:
                            headline = db[slug]
                            top_dict["headline"] = headline
                if "description" not in top_dict:
                    with sqldict(DESCDB) as db:
                        # print("Getting description from db")
                        if slug in db:
                            description = db[slug]
                            top_dict["description"] = description
                if "keywords" not in top_dict:
                    with sqldict(KWDB) as db:
                        # print("Getting keywords from db")
                        if slug in db:
                            keywords = db[slug]
                            top_dict["keywords"] = keywords
        else:
            new_post.append(line)
    if top_dict:
        # Loop through top_dict and add each key/value pair to top_matter.
        for key, value in top_dict.items():
            top_matter.append(f"{key}: {q(value)}")
        top_matter.append("---")
    top_matter.extend(new_post)
    content = top_matter
    return "\n".join(content)


def q(text):
    # Ensure that it is quoted if it needs it based on the use of colons
    # while defending against double quotes. It's too easy to make strings
    # that have accumulated nested quotes. Use some technique like Regex or
    # something to make sure there's not patterns like "", """, """"", etc.
    if ":" in text:
        if '"' in text:
            # Use RegEx to remove any number of repeating double quotes with only one double quote.
            # This will allow us to use single quotes to wrap the string.
            text = re.sub(r'\"{2,}', '"', text)
        if text[0] != '"' and text[-1] != '"':
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
def find_keywords(data):
    """Returns top keywords and main category for text."""
    print("Hitting OpenAI API for: keywords")
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
    keywords = response.choices[0].text.strip()
    return keywords


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
    description = response.choices[0].text.strip()
    return description


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

#  _____           _   _____                 _   _                 
# | ____|_ __   __| | |  ___|   _ _ __   ___| |_(_) ___  _ __  ___ 
# |  _| | '_ \ / _` | | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# | |___| | | | (_| | |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_____|_| |_|\__,_| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
#                                                                  

#  _   _ _     _                                  
# | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___  
# | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \ 
# |  _  | \__ \ || (_) | (_| | | | (_| | | | | | |
# |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_|
#                       |___/                     

# There is a 1-time dependenccy on running the following commands:
# import nltk
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')

def show_common(counter_obj, num_items):
    console = Console()
    most_common = counter_obj.most_common(num_items)
    
    # Create table and add header
    table = Table(title="Most Common Items", show_header=True, header_style="bold magenta")
    table.add_column("Item", justify="left", style="cyan")
    table.add_column("Count", justify="right", style="green")
    
    # Add rows to the table
    for item, count in most_common:
        table.add_row(item, f"{count}")
    
    console.print(table)

fig("Histogram")
keywords = []
cat_dict = defaultdict(list)
lemmatizer = WordNetLemmatizer()
with sqldict(KWDB) as db:
    for slug, more_words in db.iteritems():
        keywords += more_words.split(", ")
        for keyword in keywords:
            cat_dict[keyword].append(slug)
keywords = [x.strip() for x in keywords]
keywords = [lemmatizer.lemmatize(x.lower()) for x in keywords if x]
keywords = Counter(keywords)
show_common(keywords, 100)

#   ____      _                                ____                       
#  / ___|__ _| |_ ___  __ _  ___  _ __ _   _  |  _ \ __ _  __ _  ___  ___ 
# | |   / _` | __/ _ \/ _` |/ _ \| '__| | | | | |_) / _` |/ _` |/ _ \/ __|
# | |__| (_| | ||  __/ (_| | (_) | |  | |_| | |  __/ (_| | (_| |  __/\__ \ \____\__,_|\__\___|\__, |\___/|_|   \__, | |_|   \__,_|\__, |\___||___/
#                     |___/            |___/              |___/           

# Delete all previous category pages
for p in Path(INCLUDES).glob("cat_*"):
    p.unlink()

fig("Category Pages")
slugs = cat_dict[keyword]
for i, (keyword, freq) in enumerate(keywords.most_common(100)):
    cat_file = slugify(keyword)
    cat_file = f"{INCLUDES}cat_{cat_file}.md"
    # print(i + 1, len(slugs), cat_file)
    print(f"{i}", end=" ", flush=True)
    with open(cat_file, "w") as fh:
        # We're going to iterate the items in the list for this keyword in the cat_dict
        # and write them to the file
        for slug in cat_dict[keyword]:
            # print(f"  {slug}")
            link = f'<li><a href="/{BLOG}/{slug}/">{slug}</a></li>'
            fh.write(f"{link}\n")
print()

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

#  ____      _           _ _     _       _                              _ 
# |  _ \ ___| |__  _   _(_) | __| |     | | ___  _   _ _ __ _ __   __ _| |
# | |_) / _ \ '_ \| | | | | |/ _` |  _  | |/ _ \| | | | '__| '_ \ / _` | |
# |  _ <  __/ |_) | |_| | | | (_| | | |_| | (_) | |_| | |  | | | | (_| | |
# |_| \_\___|_.__/ \__,_|_|_|\__,_|  \___/ \___/ \__,_|_|  |_| |_|\__,_|_|

# Delete the old temporary journal from _data
out_file = Path(OUTPUT2_PATH)
if out_file.exists():
    out_file.unlink()

# Rebuild the journal in _data
all_posts = rebuild_journal(FULL_PATH)
with open(OUTPUT2_PATH, "a") as fh:
    fig("Rebuilding")
    for i, apost in enumerate(all_posts):
        print(i, end=" ", flush=True)
        if i:
            fh.write(CHOPPER)
        apost = front_matter_inserter(apost)
        fh.write(apost)
print()

#  ___        _               _     ____    ___                   _   
# / _ \ _   _| |_ _ __  _   _| |_  |___ \  |_ _|_ __  _ __  _   _| |_ 
#| | | | | | | __| '_ \| | | | __|   __) |  | || '_ \| '_ \| | | | __|
#| |_| | |_| | |_| |_) | |_| | |_   / __/   | || | | | |_) | |_| | |_ 
# \___/ \__,_|\__| .__/ \__,_|\__| |_____| |___|_| |_| .__/ \__,_|\__|
#                |_|                                 |_|              

# Compare the input and output files. If same, there's been no changes.
fig("Compare files")
files_are_same = compare_files(FULL_PATH, OUTPUT2_PATH)
print(f"Are the input and output files the same? {files_are_same}")
if files_are_same:
    print("Nothing's changed. Nothing to publish.")
else:
    print("Something's changed. Copied output to input.")
    # Copy output to input file using pathlib
    shutil.copyfile(OUTPUT2_PATH, FULL_PATH)

#  ___           _             ____                  
# |_ _|_ __   __| | _____  __ |  _ \ __ _  __ _  ___ 
#  | || '_ \ / _` |/ _ \ \/ / | |_) / _` |/ _` |/ _ \
#  | || | | | (_| |  __/>  <  |  __/ (_| | (_| |  __/
# |___|_| |_|\__,_|\___/_/\_\ |_|   \__,_|\__, |\___|
#                                         |___/      

if not DEBUG:
    # Add countdown ordered list to index page
    links.insert(0, f'<ol start="{len(links)}" reversed>')
    links.append("</ol>")
    # Write index page
    index_page = "\n".join(links)
    # Write out list of posts
    with open(f"{INCLUDES}post_list.html", "w", encoding="utf-8") as fh:
        fh.writelines(index_page)
#   ____ _ _     ____            _     
#  / ___(_) |_  |  _ \ _   _ ___| |__  
# | |  _| | __| | |_) | | | / __| '_ \ 
# | |_| | | |_  |  __/| |_| \__ \ | | |
#  \____|_|\__| |_|    \__,_|___/_| |_|

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
