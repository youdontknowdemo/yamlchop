# Author: Mike Levin
# Date: 2023-04-25
# Description: Chop a journal.md file into individual blog posts.
#   ____ _                  ____ _
#  / ___| |__   ___  _ __  / ___| |__   ___  _ __    - I keep one journal.md file for life.
# | |   | '_ \ / _ \| '_ \| |   | '_ \ / _ \| '_ \   - This is it. One place to look each day.
# | |___| | | | (_) | |_) | |___| | | | (_) | |_) |  - I chop it up into individual blog posts.
#  \____|_| |_|\___/| .__/ \____|_| |_|\___/| .__/   - I use this script to do it.
#                   |_|                     |_|
# USAGE: python ~/repos/skite/chopchop.py -f /mnt/c/Users/mikle/repos/hide/MikeLev.in/journal.md
#                       ___
#                      |   |         ____            This looks odd because it's a text-based UI
#                      |_  |        /    \           and I'm using a monospace font.
#                        \ |       |      \
#                        |  \      |       \         I'm able to edit it so easily because I use
#                         \  \____ \_      |         a text-based editor called Vim. Actually, it's
#                          \      \_/     _|         NeoVim, which is a fork of Vim. Same thing.
#                    __     \_           /
#   .-,             /  \      |          |           I'll spend a little time designing this odd
#   |  \            |   `----_|          |           bit of space here next to Alice, who really
#    \  \_________,-`                /\   \          isn't necessary either for this chop chop
#    |                              /  \_  \         script to work. But I like to make references
#    `-----------,                  |    \  \        to it all the time, so here she is.
#                |                  /     \  |
#                |                 |       | \       Monospace text is good. Don't unterestimate
#                /                 |       \__|      its power, no matter what direction tech goes.
#               /   _              |                 It takes awhile to get used to the fact that
#              /   / \_             \                text can be a UI. But once you do, you'll
#              |  /    \__      __--`                never go back. It's so much more powerful.
#             _/ /        \   _/
#         ___/  /          \_/                       That's Wonderland! It's a Linux wonder land.
#        /     /                                     Get used to stringing up the output of commands
#        `----`                                      and piping them into other commands.

import os
import re
import sys
import yaml
import shlex
import openai
import shutil
import argparse
from time import sleep
from retry import retry
from pathlib import Path
from slugify import slugify
from pyfiglet import Figlet
from datetime import datetime
from subprocess import Popen, PIPE
from nltk.stem import WordNetLemmatizer
from sqlitedict import SqliteDict as sqldict
from collections import Counter, defaultdict


AUTHOR = "Mike Levin"
ENGINE = "text-davinci-003"
NUMBER_OF_CATEGORIES = 100


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


fig("ChopChop", "A radical new YAMLesque blogging system based on 1-file for life")

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
SEPARATOR = (80 * "-") + "\n"
CHOP = re.compile(r"-{78,82}\s*\n")
ARGS = aparser.parse_args()
BLOG = f"/{ARGS.blog}/" if not ARGS.blog.startswith("/") else ARGS.blog
OUTPUT = ARGS.output
AUTHOR = ARGS.author
YAMLESQUE = ARGS.full_path
parts = YAMLESQUE.split("/")
REPO = parts[-2] + "/"
FILE = parts[-1]
PATH = "/".join(parts[:-2]) + "/"
GIT_EXE = "/usr/bin/git"
INCLUDES = f"{PATH}{REPO}_includes/"
REPO_DATA = f"{PATH}{REPO}_data/"
OUTPUT_PATH = f"{PATH}{REPO}{OUTPUT}"
TEMP_OUTPUT = f"{REPO_DATA}{FILE}"
KEYWORDS_FILE = "{PATH}{REPO}_data/keywords.txt"
CATEGORY_PAGE = f"{PATH}{REPO}category.md"
CATEGORY_INCLUDE = f"{INCLUDES}category.md"
CATEGORY_FILTER = ["blog", "index", "journal", "category", "none", "default"]

# Databases
SUMMARIESDB = REPO_DATA + "summaries.db"
HEADLINESDB = REPO_DATA + "headlines.db"
DESCRIPTIONSDB = REPO_DATA + "descriptions.db"
KEYWORDSDB = REPO_DATA + "keywords.db"

# Print out constants
fig(REPO, f"REPO: {REPO}")  # Print the repo name
print(f"PATH: {PATH}")
print(f"FILE: {FILE}")

# globals
ydict = defaultdict(dict)
cdict = defaultdict(dict)

# Create output path if it doesn't exist
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(REPO_DATA).mkdir(parents=True, exist_ok=True)

with open("/home/ubuntu/repos/skite/openai.txt", "r") as fh:
    openai.api_key = fh.readline()

#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/


def chop_chop(full_path, reverse=False):
    """Yields 3-tuples of YAMLESQUE source-file as (YAML, post, YAML+post).
    If there's no YAML, position 0 will be None and position 2 will be post."""
    #   ____                           _
    #  / ___| ___ _ __   ___ _ __ __ _| |_ ___  _ __
    # | |  _ / _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
    # | |_| |  __/ | | |  __/ | | (_| | || (_) | |
    #  \____|\___|_| |_|\___|_|  \__,_|\__\___/|_|
    with open(full_path, "r") as fh:
        global SEPARATOR
        posts = CHOP.split(fh.read())
        if reverse:
            posts.reverse()  # Reverse so article indexes don't change.
        for i, post in enumerate(posts):
            rv = None, post, post  # Always rebuild source at very least.
            parsed_yaml, yaml_str, body = None, "", ""
            if "---" in post:
                yaml_str, body = post.split("---", 1)
                try:
                    parsed_yaml = yaml.load(yaml_str, Loader=yaml.FullLoader)
                    rv = parsed_yaml, body, f"{SEPARATOR}{yaml_str}---{body}"
                except yaml.YAMLError:
                    # This generator is for outputting pages with YAML.
                    # Passing silently here will cause the page to be
                    # not be published (banner, to-do, no title, etc.)
                    ...
            yield rv


def diagnose_yaml(yaml_str, YMLError):
    fig("YAML ERROR", "READ THE YAML LINE-BY-LINE UNTIL KAPUT...")
    for j, astr in enumerate(yaml_str.split("\n")):
        print(f"LINE {j + 1}--> {astr}")
    print()
    print("And here's the error:")
    print(f"YMLError.context_mark: {YMLError.context_mark}")
    print()
    print("And the breakdown of the error:")
    print(f"exec.context_mark: {YMLError.context_mark}")
    print(f"exec.problem_mark: {YMLError.problem_mark}")
    raise SystemExit()


def odb(DBNAME, afunc, slug, full_text):
    """Record OpenAI API hits in a database."""
    api_hit = False
    with sqldict(DBNAME) as db:
        if slug in db:
            result = db[slug]
        else:
            #     _    ____ ___   _     _ _
            #    / \  |  _ \_ _| | |__ (_) |_
            #   / _ \ | |_) | |  | '_ \| | __|
            #  / ___ \|  __/| |  | | | | | |_
            # /_/   \_\_|  |___| |_| |_|_|\__|
            result = afunc(full_text)  # Hits OpenAI API
            db[slug] = result
            db.commit()
            fig("Hit API", f"Hit OpenAI API and saved to {DBNAME}")
            api_hit = True
    return result, api_hit


@retry(Exception, delay=1, backoff=2, max_delay=60)
def write_keywords(data):
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
def write_description(data):
    """Write a meta description for a post."""
    response = openai.Completion.create(
        engine=ENGINE,
        prompt=(
            f"Write a concise and informative meta description for the following text:\n{data}\n\n"
            "...that will entice readers to click through to the blog post. "
            "You wrote this. Write from the first person perspective. Never say 'The author'. '"
            "Keep it short to just like a few sentences, but always finish your sentences."
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


def oget(DBNAME, slug):
    """Return a value from a database."""
    with sqldict(DBNAME) as db:
        if slug in db:
            result = db[slug]
        else:
            result = None
    return result


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


def sq(text):
    """Safely return a quoted string."""
    if not text:
        return text
    text = text.strip()
    text = text.strip('"')
    text = re.sub(r"\"{2,}", '"', text)
    text = text.replace('"', "'")
    # Replace all cariage returns and line feeds with spaces:
    text = re.sub(r"[\r\n]+", " ", text)
    # Replace all multiple spaces with a single space:
    text = re.sub(r"\s+", " ", text)
    # If any of the following characters (including single quotes) are in the text, do if:
    if re.search(r"[;:]", text):
        text = f'"{text}"'
    return text


def build_ydict(yamlesque=YAMLESQUE):
    """Rebuilds ydict from _data/*.dbs, which may have more daata than the YAMLESQUE source."""
    #  ____        _ _     _                         _       _ _      _
    # | __ ) _   _(_) | __| |  _   _  __ _ _ __ ___ | |   __| (_) ___| |_
    # |  _ \| | | | | |/ _` | | | | |/ _` | '_ ` _ \| |  / _` | |/ __| __|
    # | |_) | |_| | | | (_| | | |_| | (_| | | | | | | | | (_| | | (__| |_
    # |____/ \__,_|_|_|\__,_|  \__, |\__,_|_| |_| |_|_|  \__,_|_|\___|\__|
    #                          |___/
    fig("YAML check", "Building dictionary of all YAML with slug.")
    global ydict
    ydict = defaultdict(dict)
    for i, (fm, _, _) in enumerate(chop_chop(YAMLESQUE)):
        if fm and isinstance(fm, dict):
            if "title" in fm:
                slug = slugify(fm["title"])
                fm["slug"] = slug
                ydict[slug] = fm
    print(f"ydict has {len(ydict)} entries.")


def deletes():
    global INCLUDES
    fig("Deleting old", "Deleting auto-generated pages from site.")
    #  ____       _      _   _                     _     _
    # |  _ \  ___| | ___| |_(_)_ __   __ _    ___ | | __| |
    # | | | |/ _ \ |/ _ \ __| | '_ \ / _` |  / _ \| |/ _` |
    # | |_| |  __/ |  __/ |_| | | | | (_| | | (_) | | (_| |
    # |____/ \___|_|\___|\__|_|_| |_|\__, |  \___/|_|\__,_|
    #                                |___/
    for fh in os.listdir(OUTPUT_PATH):
        delete_me = f"{OUTPUT_PATH}/{fh}"
        os.remove(delete_me)
    for fh in os.listdir(f"{PATH}{REPO}"):
        if fh.startswith("cat_"):
            delete_me = f"{PATH}{REPO}/{fh}"
            os.remove(delete_me)
    for fh in os.listdir(f"{INCLUDES}"):
        if fh.startswith("cat_"):
            delete_me = f"{INCLUDES}/{fh}"
            os.remove(delete_me)
    of = Path(TEMP_OUTPUT)
    if of.exists():
        of.unlink()


def categories():
    """Find the categories"""
    #   ____      _                        _
    #  / ___|__ _| |_ ___  __ _  ___  _ __(_) ___  ___
    # | |   / _` | __/ _ \/ _` |/ _ \| '__| |/ _ \/ __|
    # | |__| (_| | ||  __/ (_| | (_) | |  | |  __/\__ \
    #  \____\__,_|\__\___|\__, |\___/|_|  |_|\___||___/
    #                     |___/
    # Category selection
    fig("Categories", "Finding catgories...")
    pwords = defaultdict(lambda x=None: x)
    cat_dict = defaultdict(list)
    words = defaultdict(list)
    lemmatizer = WordNetLemmatizer()
    with open(YAMLESQUE) as fh:
        for post in CHOP.split(fh.read()):
            ystr, body = post.split("---", 1)
            if ystr:
                yml = yaml.load(ystr, Loader=yaml.FullLoader)
                if "title" in yml:
                    slug = slugify(yml["title"])
                if "keywords" in yml:
                    keywords = yml["keywords"].split(", ")
                    for keyword in keywords:
                        keyword = lemmatizer.lemmatize(keyword)
                        keyword_lower = keyword.lower().strip()
                        words[keyword_lower].append(keyword)
                        cat_dict[keyword_lower].append(slug)
    for key in words:
        alist = words[key]
        pwords[key] = Counter(alist).most_common(1)[0][0]
    for key in cat_dict:
        cat_dict[key].reverse()
    cat_counter = Counter()  # Create a counter object
    for cat, slugs in cat_dict.items():
        cat_counter[cat] = len(slugs)
    common_cats = cat_counter.most_common()
    common_cats = [x for x in common_cats if x[0] not in CATEGORY_FILTER]
    show_cats = 15
    for cat, count in common_cats:
        cdict[cat]["slug"] = slugify(cat)
        cdict[cat]["count"] = count
        cdict[cat]["title"] = pwords[cat]
    print(f"Found {len(cdict):,} categories.")
    for i, acat in enumerate(cdict):
        print(f"{i+1}. {cdict[acat]['title']} ({cdict[acat]['count']})")
        if i + 1 >= show_cats:
            break


def category_page():
    """Build the category page (singular)"""
    #   ____      _     ____
    #  / ___|__ _| |_  |  _ \ __ _  __ _  ___
    # | |   / _` | __| | |_) / _` |/ _` |/ _ \
    # | |__| (_| | |_  |  __/ (_| | (_| |  __/
    #  \____\__,_|\__| |_|   \__,_|\__, |\___|
    #                              |___/
    global cdict
    fig("Cat Page", "Building category page...")
    if cdict:
        with open(CATEGORY_PAGE, "w") as fh:
            with open(CATEGORY_INCLUDE, "w") as fh2:
                fh.write("# Categories\n")  # This could be more frontmatter-y
                fh.write("{% include category.md %}\n")  # Reference to include
                fh2.write(f"<ol start='{NUMBER_OF_CATEGORIES}' reversed>\n")
                top_cats = get_top_cats()
                for i, cat in enumerate(top_cats):
                    permalink = slugify(cat)
                    title = cdict[cat]["title"]
                    fh2.write(f'<li><a href="/{permalink}/">{title}</a></li>\n')
                fh2.write("</ol>\n")


def category_pages():
    """Outputs the individual category pages and includes"""
    #   ____      _     ____
    #  / ___|__ _| |_  |  _ \ __ _  __ _  ___  ___
    # | |   / _` | __| | |_) / _` |/ _` |/ _ \/ __|
    # | |__| (_| | |_  |  __/ (_| | (_| |  __/\__ \
    #  \____\__,_|\__| |_|   \__,_|\__, |\___||___/
    #                              |___/
    fig("Cat Pages", "Building category pages (plural)...")
    global cdict
    lemmatizer = WordNetLemmatizer()
    top_cats = get_top_cats()

    # Map every slug to a category:
    slugcat = defaultdict(list)
    for i, (fm, apost, combined) in enumerate(chop_chop(YAMLESQUE)):
        if fm:
            if "keywords" in fm and "title" in fm:
                slug = slugify(fm["title"])
                keywords = fm["keywords"]
                keyword_list = keywords.split(", ")
                for keyword in keyword_list:
                    keyword = keyword.lower()
                    keyword = lemmatizer.lemmatize(keyword)
                    keyword = keyword.lower()
                    if keyword in top_cats:
                        slugcat[keyword].append(slug)
    # Create the category pages:
    for cat in top_cats:
        slug = slugify(cat)
        filename = f"{PATH}{REPO}cat_{slug}.md"
        include_file = f"cat_{slug}.md"
        permalink = f"/{slug}/"
        with open(filename, "w", encoding="utf-8") as fh:
            cat_str = f"""---
title: {cdict[cat]["title"]}
permalink: {permalink}
layout: default
---

# {cdict[cat]["title"]}

"""
            fh.write(cat_str)
            fh.write(f"{{% include {include_file} %}}\n")

    # Create the category includes:
    for cat in top_cats:
        slug = slugify(cat)
        filename = f"{INCLUDES}cat_{slug}.md"
        with open(filename, "w", encoding="utf-8") as fh:
            posts_in_cat = len(slugcat[cat])
            fh.write(f"<ol start='{posts_in_cat}' reversed>\n")
            for slug in slugcat[cat]:
                title = ydict[slug]["title"]
                aslug = slugify(title)
                adate = ydict[slug]["date"]
                description = ydict[slug]["description"]
                apermalink = f"{BLOG}{aslug}/"
                alink = f'<li><a href="{apermalink}">{title}</a> ({adate})\n<br/>{description}</li>\n'
                fh.write(alink)
            fh.write("</ol>\n")


def sync_check():
    """Check for new posts needing AI-writing or YAMLESQUE source-file updating."""
    #  ______   ___   _  ____    ____ _               _
    # / ___\ \ / / \ | |/ ___|  / ___| |__   ___  ___| | __
    # \___ \\ V /|  \| | |     | |   | '_ \ / _ \/ __| |/ /
    #  ___) || | | |\  | |___  | |___| | | |  __/ (__|   <
    # |____/ |_| |_| \_|\____|  \____|_| |_|\___|\___|_|\_\
    fig("SYNC Check", "Checking for new posts needing AI-writing")
    for i, (fm, apost, combined) in enumerate(chop_chop(YAMLESQUE)):
        if fm and len(fm) == 2 and "title" in fm and "date" in fm:
            # Only 2 fields of YAML front matter asks for release.
            title = fm["title"]
            slug = slugify(title)
            ydict[slug]["title"] = title

            # Setting these values ALSO commits it to the databases
            summary, api_hit = odb(SUMMARIESDB, write_summary, slug, apost)
            headline, hit_headline = odb(HEADLINESDB, write_headline, slug, summary)
            description, hit_description = odb(
                DESCRIPTIONSDB, write_description, slug, summary
            )
            keywords, hit_keywords = odb(KEYWORDSDB, write_keywords, slug, summary)

            if any([hit_description, hit_headline, hit_keywords]):
                print(f"headline: {headline}\n")
                print(f"description: {description}\n")
                print(f"keywords: {keywords}\n")
                input("Press enter to continue...")
            print()


def new_source():
    """If there's a new source, copy it to the input file. It's meta."""
    #  _   _                 ____
    # | \ | | _____      __ / ___|  ___  _   _ _ __ ___ ___
    # |  \| |/ _ \ \ /\ / / \___ \ / _ \| | | | '__/ __/ _ \
    # | |\  |  __/\ V  V /   ___) | (_) | |_| | | | (_|  __/
    # |_| \_|\___| \_/\_/   |____/ \___/ \__,_|_|  \___\___|
    #
    # Compare the input and output files. If same, there's been no changes.
    fig("Compare files")
    files_are_same = compare_files(YAMLESQUE, TEMP_OUTPUT)
    print(f"Are the input and output files the same? {files_are_same}")
    if files_are_same:
        print("Nothing's changed. Nothing to publish.")
    else:
        print("Something's changed. Copied output to input.")
        # Replaces old journal.md with the new journal.md (AI content filled-in)
        shutil.copyfile(TEMP_OUTPUT, YAMLESQUE)


def make_index():
    """Builds the index page"""
    #  ___           _             ____
    # |_ _|_ __   __| | _____  __ |  _ \ __ _  __ _  ___
    #  | || '_ \ / _` |/ _ \ \/ / | |_) / _` |/ _` |/ _ \
    #  | || | | | (_| |  __/>  <  |  __/ (_| | (_| |  __/
    # |___|_| |_|\__,_|\___/_/\_\ |_|   \__,_|\__, |\___|
    #                                         |___/
    fig("Index Page", "Making the index page")
    with open(f"{INCLUDES}post_list.html", "w", encoding="utf-8") as fh:
        num_posts = len(ydict) + 1
        fh.write(f'<ol start="{num_posts}" reversed >\n')
        for i, (fm, apost, combined) in enumerate(chop_chop(YAMLESQUE)):
            if fm and "title" in fm and "date" in fm and "description" in fm:
                title = fm["title"]
                slug = slugify(title)
                description = fm["description"]
                # Neutralize pointy brackets for description:
                description = description.replace("<", "&lt;")
                description = description.replace(">", "&gt;")
                adate = fm["date"]
                fh.write(
                    f'<li><a href="{BLOG}{slug}/">{title}</a> ({adate})\n<br />{description}</li>\n'
                )
        fh.write("</ol>\n")


def write_posts():
    """Chop YAMLESQUE up into posts"""
    #   ____ _                   ____           _
    #  / ___| |__   ___  _ __   |  _ \ ___  ___| |_ ___
    # | |   | '_ \ / _ \| '_ \  | |_) / _ \/ __| __/ __|
    # | |___| | | | (_) | |_) | |  __/ (_) \__ \ |_\__ \
    #  \____|_| |_|\___/| .__/  |_|   \___/|___/\__|___/
    #                   |_|
    fig("Chop Pages", "Chop, chop, chop...")
    # This is a YAMLesque system, so we need to be able to parse YAMLesque.
    # Chop, chop YAMLESQUE, that's what we do as Python generator.
    # That means it's memory efficient and can parse very large files.
    for i, (fm, body, combined) in enumerate(chop_chop(YAMLESQUE)):
        if fm and len(fm) > 2:
            title = fm["title"]
            stem = slugify(title)
            print(f"{i+1} ", end="", flush=True)
            adate = fm["date"]
            description = sq(fm["description"])
            headline = sq(fm["headline"])
            keywords = sq(fm["keywords"])
            keyword_list = keywords.split(", ")
            categories = set()
            top_cats = get_top_cats()
            for keyword in keyword_list:
                keyword = keyword.lower()
                if keyword in top_cats:
                    categories.add(keyword)
            categories = ", ".join(categories)
            permalink = f"{BLOG}{stem}/"
            date_object = datetime.strptime(adate, "%a %b %d, %Y")
            adate = date_object.strftime("%Y-%m-%d")
            filename = f"{OUTPUT_PATH}/{adate}-{stem}.md"
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write(
                    f"""---
date: {adate}
title: {sq(title)}
permalink: {permalink}
headline: {sq(headline)}
description: {sq(description)}
keywords: {sq(keywords)}
categories: {sq(categories)}
author: {sq(AUTHOR)}
layout: post
---"""
                )
                fh.write(body)
    print()


def git_push():
    """Pushes the changes to Github"""
    #   ____ _ _     ____            _
    #  / ___(_) |_  |  _ \ _   _ ___| |__
    # | |  _| | __| | |_) | | | / __| '_ \
    # | |_| | | |_  |  __/| |_| \__ \ | | |
    #  \____|_|\__| |_|    \__,_|___/_| |_|
    # Git commands
    fig("Git Push", "Releasing site changes...")
    here = f"{PATH}{REPO}"
    git(here, f"add {here}cat_*")
    git(here, "add _posts/*")
    git(here, "add _includes/*")
    git(here, "add assets/images/*")
    git(here, f'commit -am "Pushing {REPO} to Github..."')
    git(here, "push")


#  _____           _   _____                 _   _
# | ____|_ __   __| | |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# |  _| | '_ \ / _` | | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# | |___| | | | (_| | |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_____|_| |_|\__,_| |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
# This is a YAMLesque system, so we need to be able to parse YAMLesque.

#  _____ _            ____  _                                             _
# |_   _| |__   ___  |  _ \| | __ _ _   _  __ _ _ __ ___  _   _ _ __   __| |
#   | | | '_ \ / _ \ | |_) | |/ _` | | | |/ _` | '__/ _ \| | | | '_ \ / _` |
#   | | | | | |  __/ |  __/| | (_| | |_| | (_| | | | (_) | |_| | | | | (_| |
#   |_| |_| |_|\___| |_|   |_|\__,_|\__, |\__, |_|  \___/ \__,_|_| |_|\__,_|
#                                   |___/ |___/
# Doing so has a distinct Python generator look to it, where we yield chunks:


def get_top_cats():
    global cdict
    tcats = [x[1] for x in enumerate(cdict) if x[0] < NUMBER_OF_CATEGORIES]
    return tcats


#  _____ _                                 _             _
# |  ___| | _____      __   ___ ___  _ __ | |_ _ __ ___ | |
# | |_  | |/ _ \ \ /\ / /  / __/ _ \| '_ \| __| '__/ _ \| |
# |  _| | | (_) \ V  V /  | (_| (_) | | | | |_| | | (_) | |
# |_|   |_|\___/ \_/\_/    \___\___/|_| |_|\__|_|  \___/|_|
# This controls the entire (usually linear) flow. Edit for debugging.

build_ydict()  # Builds global ydict (should always run)
deletes()  # Deletes old posts
categories()  # Builds global categories and builds category pages
category_page()  # Builds category.md and include
category_pages()  # Builds cat_*.md and cat_*.md includes
sync_check()  # Catches YAMLESQUE file up with database of OpenAI responses
# new_source()     # Replaces YAMLESQUE input with syncronized output
make_index()  # Builds index page of all posts (for blog page)
write_posts()  # Writes out all Jekyll-style posts
git_push()  # Pushes changes to Github (publishes)

#  ____
# |  _ \  ___  _ __   ___
# | | | |/ _ \| '_ \ / _ \
# | |_| | (_) | | | |  __/
# |____/ \___/|_| |_|\___|
fig("Done!", "All done!")
