# Author: Mike Levin
# Date: 2023-04-22
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
ENGINE = "text-davinci-003"
NUMBER_OF_CATEGORIES = 300

# Debugging
POST_BY_POST = True
INTERACTIVE = False

with open(f"/home/ubuntu/repos/skite/openai.txt", "r") as fh:
    openai.api_key = fh.readline()


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
OUTPUT_PATH = f"{PATH}{REPO}{OUTPUT}"
REPO_DATA = f"{PATH}{REPO}_data/"
OUTPUT2_PATH = f"{REPO_DATA}{FILE}"
KEYWORDS_FILE = "{PATH}{REPO}_data/keywords.txt"
INCLUDES = f"{PATH}{REPO}_includes/"
CATEGORY_PAGE = f"{PATH}{REPO}category.md"
CATEGORY_INCLUDE = f"{INCLUDES}category.md"
CATEGORY_FILTER = ["blog", "index", "journal", "category"]
CATEGORIES = []

# Databases
SUMDB = REPO_DATA + "summaries.db"
DESCDB = REPO_DATA + "descriptions.db"
KWDB = REPO_DATA + "keywords.db"
HEADS = REPO_DATA + "headlines.db"

# Print out constants
fig(REPO, f"REPO: {REPO}")  # Print the repo name
print(f"PATH: {PATH}")
print(f"FILE: {FILE}")

# Globals
ydict = defaultdict(dict)
pwords = defaultdict(lambda x=None: x)

# Create output path if it doesn't exist
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(REPO_DATA).mkdir(parents=True, exist_ok=True)

#  _____                 _   _
# |  ___|   _ _ __   ___| |_(_) ___  _ __  ___
# | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __|
# |  _|| |_| | | | | (__| |_| | (_) | | | \__ \
# |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/


def chop_chop(full_path, reverse=False):
    """Chop YAMLesque file to spew chuks."""
    #   ____                           _
    #  / ___| ___ _ __   ___ _ __ __ _| |_ ___  _ __
    # | |  _ / _ \ '_ \ / _ \ '__/ _` | __/ _ \| '__|
    # | |_| |  __/ | | |  __/ | | (_| | || (_) | |
    #  \____|\___|_| |_|\___|_|  \__,_|\__\___/|_|
    global ydict
    ydict = defaultdict(dict)
    with open(full_path, "r") as fh:
        posts = CHOP.split(fh.read())
        if reverse:
            posts.reverse()  # Reverse so article indexes don't change.
        for i, post in enumerate(posts):
            if "---" not in post:
                rv = None, None, post
            else:
                yaml_str, body = post.split("---", 1)
                try:
                    yaml_test = yaml.load(yaml_str, Loader=yaml.FullLoader)
                    if yaml_test:
                        combined = f"{SEPARATOR}{yaml_str}---{body}"
                    else:
                        combined = post
                    rv = yaml_test, body, combined
                except yaml.YAMLError as exc:
                    fig("YAML ERROR", "READ THE YAML LINE-BY-LINE UNTIL KAPUT...")
                    for j, astr in enumerate(yaml_str.split("\n")):
                        print(f"LINE {j + 1}--> {astr}")
                    print()
                    print("And here's the error:")
                    print(exc)
                    # ['context', 'context_mark', 'note', 'problem', 'problem_mark']:
                    print()
                    print("And the breakdown of the error:")
                    print(f"exec.context_mark: {exc.context_mark}")
                    print(f"exec.problem_mark: {exc.problem_mark}")
                    raise SystemExit()
            # Populate the global ydict
            if yaml_test and "title" in yaml_test:
                slug = slugify(yaml_test["title"])
                ydict[slug] = yaml_test
            yield rv


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


def safety_quotes(text):
    if not text:
        return text
    if ":" in text:
        if '"' in text:
            text = re.sub(r"\"{2,}", '"', text)
        if text[0] != '"' and text[-1] != '"':
            text = f'"{text}"'
    return text


def get_capitization_dict():
    # We need a dictionary of most common capitalization usage of Category words.
    words = defaultdict(list)
    with sqldict(KWDB) as db:
        for slug, keywords in db.iteritems():
            keywords = keywords.split(", ")
            for keyword in keywords:
                lower_word = keyword.strip().lower()
                words[lower_word].append(keyword)
    for key in words:
        alist = words[key]
        pwords[key] = Counter(alist).most_common(1)[0][0]
    # Hardwired overrides:
    pwords["window"] = "Windows"
    return pwords


def build_ydict():
    """Rebuilds ydict from _data/*.dbs, which may have more daata than the YAMLESQUE source."""
    #  ____        _ _     _                         _       _ _      _
    # | __ ) _   _(_) | __| |  _   _  __ _ _ __ ___ | |   __| (_) ___| |_
    # |  _ \| | | | | |/ _` | | | | |/ _` | '_ ` _ \| |  / _` | |/ __| __|
    # | |_) | |_| | | | (_| | | |_| | (_| | | | | | | | | (_| | | (__| |_
    # |____/ \__,_|_|_|\__,_|  \__, |\__,_|_| |_| |_|_|  \__,_|_|\___|\__|
    #                          |___/
    with open(OUTPUT2_PATH, "w") as fh:
        for i, (fm, body, combined) in enumerate(chop_chop(YAMLESQUE)):
            if fm:
                if len(fm) == 2 and "date" in fm and "title" in fm:
                    # We may have more data in the ydict to add to this.
                    slug = slugify(fm["title"])
                    description = oget(DESCDB, slug)
                    keywords = oget(KWDB, slug)
                    headline = oget(HEADS, slug)
                    ydict[slug]["headline"] = headline
                    ydict[slug]["description"] = description
                    ydict[slug]["keywords"] = keywords
                    # Flatten ydict[slug] into a string of key/value pairs.
                    front_matter = "\n".join(
                        [f"{k}: {safety_quotes(v)}" for k, v in ydict[slug].items()]
                    )
                    combined = f"{SEPARATOR}{front_matter}\n---{body}"
                else:
                    write_me = combined
            else:
                write_me = combined
            fh.write(combined)


def histogram():
    """Create a histogram of keywords."""
    #  _   _ _     _
    # | | | (_)___| |_ ___   __ _ _ __ __ _ _ __ ___
    # | |_| | / __| __/ _ \ / _` | '__/ _` | '_ ` _ \
    # |  _  | \__ \ || (_) | (_| | | | (_| | | | | | |
    # |_| |_|_|___/\__\___/ \__, |_|  \__,_|_| |_| |_|
    #                       |___/
    # There is a 1-time dependenccy on running the following commands:
    # import nltk; nltk.download('wordnet')
    keywords = []
    lemmatizer = WordNetLemmatizer()
    cat_dict = defaultdict(list)
    with sqldict(KWDB) as db:
        for slug, kwstr in db.iteritems():
            keywords = kwstr.split(", ")
            for keyword in keywords:
                keyword = keyword.strip().lower()
                keyword = lemmatizer.lemmatize(keyword)
                keyword = keyword.lower()
                cat_dict[keyword].append(slug)
    # Reverse each list of slugs so that the most recent is first.
    for key in cat_dict:
        cat_dict[key].reverse()
    return cat_dict


def show_common(counter_obj, num_items):
    """Show the most common items in a counter object and return a list of the items."""
    #  ____  _
    # / ___|| |__   _____      __   ___ ___  _ __ ___  _ __ ___   ___  _ __
    # \___ \| '_ \ / _ \ \ /\ / /  / __/ _ \| '_ ` _ \| '_ ` _ \ / _ \| '_ \
    #  ___) | | | | (_) \ V  V /  | (_| (_) | | | | | | | | | | | (_) | | | |
    # |____/|_| |_|\___/ \_/\_/    \___\___/|_| |_| |_|_| |_| |_|\___/|_| |_|
    global pwords
    stop_at = 10
    console = Console()
    most_common = counter_obj.most_common(num_items)
    categories = [item[0] for item in most_common]
    # Create table and add header
    table = Table(
        title=f"Most Common {stop_at} Items",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Item", justify="left", style="cyan")
    table.add_column("Count", justify="right", style="green")
    # Add rows to the table
    for i, (item, count) in enumerate(most_common):
        table.add_row(f"{i + 1}. {pwords[item]}", f"{count}")
        if i > stop_at - 2:
            break
    console.print(table)
    return categories


def deletes():
    fig("Deleting", "Deleting auto-generated pages from site.")
    #  ____       _      _
    # |  _ \  ___| | ___| |_ ___  ___
    # | | | |/ _ \ |/ _ \ __/ _ \/ __|
    # | |_| |  __/ |  __/ ||  __/\__ \
    # |____/ \___|_|\___|\__\___||___/
    # None of this is worth doing if ydict didn't come back with values.
    if ydict:
        # Delete old files in output path
        for fh in os.listdir(OUTPUT_PATH):
            delete_me = f"{OUTPUT_PATH}/{fh}"
            os.remove(delete_me)
        # Delete all cat_*.md files in root:
        for fh in os.listdir(f"{PATH}{REPO}"):
            if fh.startswith("cat_"):
                delete_me = f"{PATH}{REPO}/{fh}"
                os.remove(delete_me)
        # Delete the old temporary journal from _data
        out_file = Path(OUTPUT2_PATH)
        if out_file.exists():
            out_file.unlink()
    else:
        raise SystemExit("No YAML front matter found in journal.")


def categories():
    """Create category pages from keywords."""
    #   ____      _                        _
    #  / ___|__ _| |_ ___  __ _  ___  _ __(_) ___  ___
    # | |   / _` | __/ _ \/ _` |/ _ \| '__| |/ _ \/ __|
    # | |__| (_| | ||  __/ (_| | (_) | |  | |  __/\__ \
    #  \____\__,_|\__\___|\__, |\___/|_|  |_|\___||___/
    #                     |___/
    fig("Categories", "Creating category pages from keywords.")
    global CATEGORIES
    pwords = get_capitization_dict()  # Get a dict of capitalized words
    cat_dict = histogram()  # Get a dict of keywords and slugs
    cat_counter = Counter()  # Create a counter object
    for cat, slugs in cat_dict.items():
        cat_counter[cat] = len(slugs)  # Add the number of slugs to the counter
    CATEGORIES = show_common(cat_counter, NUMBER_OF_CATEGORIES)
    print(f"Using {len(CATEGORIES)} categories.")

    # Write out the category page that goes in the site root as category.md
    print("Writing out category.md and its include file.")
    with open(CATEGORY_PAGE, "w") as fh:
        fh.write("# Categories\n")  # This could be more frontmatter-y
        fh.write("{% include category.md %}\n")  # Reference to include
        with open(CATEGORY_INCLUDE, "w") as fh2:
            fh2.write(f"<ol start='{len(CATEGORIES)}' reversed>\n")
            for category in CATEGORIES:
                permalink = slugify(category)
                pcat = pwords[category.lower()]
                fh2.write(f'<li><a href="/{permalink}/">{pcat}</a></li>\n')
            fh2.write("</ol>\n")

    # Write out the many category pages that go in the site root as cat_*.md
    for i, category in enumerate(CATEGORIES):
        if category not in CATEGORY_FILTER:
            permalink = slugify(category)
            pcat = pwords[category.lower()]
            front_matter = f"""---
            title: {pcat}
            permalink: /{permalink}/
            layout: default
            ---

            """
            front_matter = "\n".join([x.strip() for x in front_matter.split("\n")])
            cat_file = f"{PATH}{REPO}cat_{permalink}.md"
            include_file = f"{INCLUDES}cat_{permalink}.md"
            # print(f"Creating {cat_file}")
            with open(cat_file, "w") as fh:
                fh.write(front_matter)
                fh.write(f"# {pcat}\n\n")
                # Filter out categories without YAML data:
                cat_dict[category] = [x for x in cat_dict[category] if x in ydict]
                # Number of posts:
                category_len = len(cat_dict[category])
                # Write reference to include file into category file:
                fh.write(
                    f"{{% include cat_{permalink}.md %}}\n"
                )  # Reference to include
                # Write include file include:
                with open(include_file, "w") as fh2:
                    fh2.write(f"<ol start='{category_len}' reversed>\n")
                    for slug in cat_dict[category]:
                        try:
                            title = ydict[slug]["title"]
                            description = ydict[slug]["description"]
                            adate = ydict[slug]["date"]
                            fh2.write(
                                f'<li><a href="{BLOG}{slug}/">{title}</a> ({adate})\n<br/>{description}</li>\n'
                            )
                        except:
                            pass
                        # fh2.write(f'<li><a href="{BLOG}{slug}/">{slug}</a></li>\n')
                    fh2.write("</ol>\n")


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
            # The summary is too big for YAML-stuffing so we keep it in db.
            summary, api_hit = odb(SUMDB, write_summary, slug, apost)

            # Headlines belong on YAML-stuffing for re-writing
            headline, hit_headline = odb(HEADS, write_headline, slug, summary)
            ydict[slug]["headline"] = headline

            # Descriptions belong on YAML-stuffing for re-writing
            description, hit_description = odb(DESCDB, write_description, slug, summary)
            ydict[slug]["description"] = description

            # Keywords belong on YAML-stuffing for re-writing
            keywords, hit_keywords = odb(KWDB, write_keywords, slug, summary)
            ydict[slug]["keywords"] = keywords

            if any([hit_description, hit_headline, hit_keywords]):
                print(f"headline: {headline}\n")
                print(f"description: {description}\n")
                print(f"keywords: {keywords}\n")
                input(f"Press enter to continue...")
            print()
    # Outside the loop and the global ydict is updated but
    # the database may now be ahead of the YAMLesque file.
    build_ydict()


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
    files_are_same = compare_files(YAMLESQUE, OUTPUT2_PATH)
    print(f"Are the input and output files the same? {files_are_same}")
    if files_are_same:
        print("Nothing's changed. Nothing to publish.")
    else:
        print("Something's changed. Copied output to input.")
        # Replaces old journal.md with the new journal.md (AI content filled-in)
        shutil.copyfile(OUTPUT2_PATH, YAMLESQUE)


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


def git_push():
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


def write_posts():
    """Write the posts to the output directory"""
    # __        __    _ _         ____           _
    # \ \      / / __(_) |_ ___  |  _ \ ___  ___| |_ ___
    #  \ \ /\ / / '__| | __/ _ \ | |_) / _ \/ __| __/ __|
    #   \ V  V /| |  | | ||  __/ |  __/ (_) \__ \ |_\__ \
    #    \_/\_/ |_|  |_|\__\___| |_|   \___/|___/\__|___/
    fig("Write Posts", "Chop, chop, chop...")
    # This is a YAMLesque system, so we need to be able to parse YAMLesque.
    # Chop, chop YAMLESQUE, that's what we do as Python generator.
    # That means it's memory efficient and can parse very large files.
    for i, (fm, body, combined) in enumerate(chop_chop(YAMLESQUE)):
        global pwords
        if fm and len(fm) > 2:
            title = fm["title"]
            stem = slugify(title)
            # print(f"{i+1}. {stem}")
            print(f"{i+1} ", end="", flush=True)
            adate = fm["date"]
            description = fm["description"]
            headline = fm["headline"]
            keywords = fm["keywords"]
            keyword_list = keywords.split(", ")
            categories = set()
            for keyword in keyword_list:
                keyword = keyword.lower()
                if keyword in CATEGORIES:
                    categories.add(keyword)
            categories = ", ".join(categories)
            permalink = f"{BLOG}{stem}/"
            # convert a date string to a date object formatted like: Sat Apr 22, 2023
            date_object = datetime.strptime(adate, "%a %b %d, %Y")
            # convert the date object to a string formatted like: 2023-04-22
            adate = date_object.strftime("%Y-%m-%d")
            # The Jekyll pattern for filenames in the _posts directory
            # is YYYY-MM-DD-title.md
            # Write the post to the output directory
            filename = f"{OUTPUT_PATH}/{adate}-{stem}.md"
            # print(f"Writing {filename}")
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write(
                    f"""---
date: {adate}
title: "{title}"
permalink: {permalink}
description: "{description}"
keywords: {keywords}
categories: {categories}
author: {AUTHOR}
layout: post
---"""
                )
                fh.write(body)
    print()


#  _____ _                                 _             _
# |  ___| | _____      __   ___ ___  _ __ | |_ _ __ ___ | |
# | |_  | |/ _ \ \ /\ / /  / __/ _ \| '_ \| __| '__/ _ \| |
# |  _| | | (_) \ V  V /  | (_| (_) | | | | |_| | | (_) | |
# |_|   |_|\___/ \_/\_/    \___\___/|_| |_|\__|_|  \___/|_|
# This controls the entire (usually linear) flow. Edit for debugging.

build_ydict()  # Builds global ydict (should always run)
# deletes()      # Deletes old posts
categories()  # Builds global categories and builds category pages
# sync_check()   # Catches YAMLESQUE file up with database of OpenAI responses
# new_source()   # Replaces YAMLESQUE input with syncronized output
# make_index()   # Builds index page of all posts (for blog page)
write_posts()  # Writes out all Jekyll-style posts
# git_push()     # Pushes changes to Github (publishes)

#  ____
# |  _ \  ___  _ __   ___
# | | | |/ _ \| '_ \ / _ \
# | |_| | (_) | | | |  __/
# |____/ \___/|_| |_|\___|
fig("Done!", "All done!")
