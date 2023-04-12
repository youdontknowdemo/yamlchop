import os
import re
import sys
import html
import yake
import shlex
import openai
import slugify
import datetime
import argparse
import pandas as pd
from retry import retry
from pathlib import Path
from slugify import slugify
from dateutil import parser
from sklearn.cluster import KMeans
from subprocess import Popen, PIPE
from sqlitedict import SqliteDict as sqldict
from sklearn.feature_extraction.text import TfidfVectorizer


disable_git = True

# Topic Clustering
number_of_clusters = 10
random_seed = 2

# Set pandas options to display all columns and rows
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

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
SUMMARY_LENGTH = 500
GIT_EXE = "/usr/bin/git"
OUTPUT_PATH = f"{PATH}{REPO}{OUTPUT}"
REPO_DATA = f"{PATH}{REPO}_data/"
print(f"Processing {FULL_PATH}")

# Create output path if it doesn't exist
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
Path(REPO_DATA).mkdir(parents=True, exist_ok=True)

# Delete old files in output path
for f in os.listdir(OUTPUT_PATH):
    delete_me = f"{OUTPUT_PATH}/{f}"
    os.remove(delete_me)

# Get OpenAI API key
with open("/home/ubuntu/repos/skite/openai.txt") as fh:
    openai.api_key = fh.readline()


def neutralize_html(string):
    return html.escape(string)


def parse_journal(FULL_PATH):
    """Parse a journal file into posts. Returns a generator of posts."""
    with open(FULL_PATH, "r") as fh:
        print(f"Reading {FULL_PATH}")
        post_str = fh.read()
        pattern = r"-{78,82}\s*\n"
        posts = re.split(pattern, post_str)
        posts.reverse()  # Reverse so article indexes don't change.
        for post in posts:
            yield post


def write_post_to_file(post, index):
    """Write a post to a file. Returns a markdown link to the post."""
    lines = post.strip().split("\n")

    # Set up per-post variables
    date_str, slug = None, None
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
    with sqldict(REPO_DATA + "summaries.db") as db:
        if slug not in db:
            summary = summarize(post)
            db[slug] = summary
            db.commit()
        else:
            summary = db[slug]
    with sqldict(REPO_DATA + "descriptions.db") as db:
        if slug not in db:
            meta_description = write_meta(summary)
            db[slug] = meta_description
            db.commit()
        else:
            meta_description = db[slug]
    with sqldict(REPO_DATA + "keywords.db") as db:
        if slug not in db:
            full_text = f"{title} {summary} {meta_description}"
            keywords = yake.KeywordExtractor().extract_keywords(full_text)
            db[slug] = keywords
            db.commit()
        else:
            keywords = db[slug]
    with sqldict(REPO_DATA + "topics.db") as db:
        if slug not in db:
            topic = None
        else:
            topic = db[slug]

    keywords = [x[0].lower() for x in keywords]
    keywords = dehyphen_and_dedupe(keywords)

    meta_description = scrub_excerpt(meta_description)
    meta_description = neutralize_html(meta_description)
    top_matter.append(f"description: {meta_description}")
    top_matter.append(f"keywords: {keywords}")
    top_matter.append(f"category: {topic}")
    top_matter.append(f"layout: post")
    top_matter.append(f"author: {AUTHOR}")
    top_matter.append("---")
    top_matter.extend(content)
    content = top_matter

    # Write to file
    print(index, full_path)
    with open(full_path, "w") as f:
        flat_content = "\n".join(content)
        f.writelines(flat_content)

    us_date = date_str.strftime("%m/%d/%Y")
    
    # link = f"- [{title}](/{BLOG}/{slug}/) ({us_date})<br/>\n  {meta_description}"
    link = f'<li><a href="/{BLOG}/{slug}/">{title}</a> ({us_date})<br />{meta_description}</li>'
    return link


def dehyphen_and_dedupe(keywords):
    """Preserves order of keywords, but removes duplicates and hyphens"""
    keywords = [x.replace("-", " ") for x in keywords]
    seen = set()
    # A fascinating way to add to a set within a list comprehension
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


def trunc(text):
    """Truncate a string to a given length, but not in the middle of a word."""
    if len(text) <= SUMMARY_LENGTH:
        return text
    else:
        return text[: SUMMARY_LENGTH - 1].rsplit(" ", 1)[0] + "..."
    return text


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
def write_meta(data):
    """Write a meta description for a post."""
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=(
            f"Please write a meta description for the following text:\n{data}\n\n"
            "Summary:"
        ),
        temperature=0.5,
        max_tokens=100,
        n=1,
        stop=None,
    )
    meta_description = response.choices[0].text.strip()
    return meta_description


@retry(Exception, delay=1, backoff=2, max_delay=60)
def summarize(text):
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









def shortest(keywords):
    """Return the longest common sequence of words in a list of keywords"""
    keywords = [x.split(" ") for x in keywords]

    shortest = min(keywords, key=lambda x: len(x))

    for i, word in enumerate(shortest):
        if not all([word in x for x in keywords]):
            rv = shortest[:i]
    if len(shortest) > 1:
        rv = " ".join(shortest)
    elif len(shortest) == 1:
        rv = shortest[0]
    else:
        rv = keywords[0]
    return rv


def get_winning_keywords(keywords):
    """Return a list of the winning keywords. Winning keywords are those that
    are longer and whose stem words have the highest frequency."""

    keywords = sorted(keywords, key=lambda x: len(x[0]), reverse=True)
    # Get the stem words
    stems = [x[0].split(" ")[0] for x in keywords]
    # Get the frequency of each stem word
    stem_freq = {x: stems.count(x) for x in stems}
    # Get the winning stem words
    winning_stems = [x for x in stem_freq if stem_freq[x] == max(stem_freq.values())]
    # Get the winning keywords
    winning_keywords = [x for x in keywords if x[0].split(" ")[0] in winning_stems]
    # Return those with the highest frequency but favor 2-word keywords
    winning_keywords = sorted(
        winning_keywords, key=lambda x: x[1] * (len(x[0].split(" ")) + 1), reverse=True
    )
    return winning_keywords[:5]


def cluster_test(n, r):
    
    filter_us = [
        ".",
        "encourages",
        "readers",
        "called",
        "things",
        "general",
        "order",
        "offer",
        "has made",
        "process",
        "generated",
        "including",
        "blog",
        "importance",
        "important",
        "person",
        "people",
        "discussing",
        "discusses",
        "describes",
        "author",
        "suggests",
        "talks",
        "argues",
        "reflects",
    ]
    table = []
    with sqldict(REPO_DATA + "keywords.db") as db:
        for key, keywords in db.iteritems():
            keywords = [x[0] for x in keywords]
            keywords = dehyphen_and_dedupe(keywords)
            table.append((key, keywords))

    df = pd.DataFrame(table, columns=["title", "keywords"])

    # Extract the keywords and create a matrix
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform([", ".join(x) for x in df["keywords"]])

    # Apply KMeans clustering with 10 clusters
    kmeans = KMeans(n_clusters=n, n_init="auto", random_state=r)
    kmeans.fit(X)

    # Assign each article to its cluster
    df["cluster"] = kmeans.labels_
    df_grouped = df.groupby("cluster")

    cluster_dict = {}
    for i, dfc_tuple in enumerate(df_grouped):
        key, dfc = dfc_tuple
        dfx = dfc.explode("keywords")
        top_picks = list(
            dfx[["keywords", "cluster"]]
            .groupby("keywords")
            .count()
            .sort_values("cluster", ascending=False)
            .to_records()
        )
        top_pics = [x for x in top_picks if x[1] > 3]
        top_picks = [x for x in top_picks if len(x[0].split(" ")) > 1]
        top_picks = [x for x in top_picks if not any([y in x[0] for y in filter_us])]
        top_picks = get_winning_keywords(top_picks)
        top_picks = [x[0] for x in top_picks][0]
        cluster_dict[key] = top_picks
    return df, cluster_dict


def dehyphen_and_dedupe(keywords):
    """Preserves order of keywords, but removes duplicates and hyphens"""
    keywords = [x.replace("-", " ") for x in keywords]
    seen = set()
    # A fascinating way to add to a set within a list comprehension
    seen_add = seen.add
    keywords = [x.lower() for x in keywords if not (x in seen or seen_add(x))]
    return keywords


def get_topics(n, r):
    df, cluster_dict = cluster_test(n, r)
    # cluster_dict = {x: cluster_dict[x][0] for x in cluster_dict}
    cluster_dict = {x: cluster_dict[x] for x in cluster_dict}
    return df, cluster_dict


def cluster_topics():

    df, topics = get_topics(number_of_clusters, random_seed)
    topic_dict = {}
    for key in topics:
        topic = topics[key]
        word = shortest(topics[key])
        topic_dict[key] = word
        print(key, word)

    # Add a column to df that contains the topic by feeding the cluster number into the topics dict
    df["topic"] = df["cluster"].apply(lambda x: topic_dict[x])
    df = df[["title", "topic"]]

    # Commit the dataframe to a database
    with sqldict(REPO_DATA + "topics.db") as db:
        for key, value in list(df.to_records(index=False)):
            # print(key)
            db[key] = value
        db.commit()

# Process Topics From Last Go-Round
print("Processing Topics")
df, cluster_dict = cluster_test(10, 2)

# Parse the journal file
posts = parse_journal(FULL_PATH)
links = []
for i, post in enumerate(posts):
    link = write_post_to_file(post, i - 1)
    if link:
        links.insert(0, link)

# Add countdown ordered list to index page
links.insert(0, f'<ol start="{len(links)}" reversed>')
links.append("</ol>")
# Write index page
index_page = "\n".join(links)
with open(f"{PATH}{REPO}_includes/post-index.html", "w", encoding="utf-8") as fh:
    fh.writelines(index_page)

if not disable_git:
    # Git commands
    here = f"{PATH}{REPO}"
    git(here, "add _posts/*")
    git(here, "add _includes/*")
    git(here, "add assets/images/*")
    git(here, f'commit -am "Pushing {REPO} to Github..."')
    git(here, "push")
