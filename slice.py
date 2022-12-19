# This takes command-line input and spits out a bunch of files.
# We need to do some debugging. posts-main.html is not being updated.

import argparse
from pathlib import Path
from dateutil import parser
from slugify import slugify
from collections import Counter
from dumbquotes import dumbquote


if hasattr(__builtins__, "__IPYTHON__") or __name__ != "__main__":
    from IPython.display import display, Markdown

    folder_name = "../MikeLev.in"
    blog_title = "Mike Levin's Blog"
    blog_slug = "blog"
    author = "Mike Levin"
    verbose = False
else:

    aparser = argparse.ArgumentParser()
    add_arg = aparser.add_argument
    add_arg("-p", "--path", required=True)
    add_arg("-t", "--title", required=True)
    add_arg("-s", "--slug", required=True)
    add_arg("-a", "--author", required=True)
    add_arg("-v", "--verbose", required=False, default=False)
    args = aparser.parse_args()

    folder_name = args.path
    blog_title = args.title
    blog_slug = args.slug
    author = args.author
    verbose = args.verbose

index_front_matter = f"""---
layout: default
author: {author}
title: "{blog_title}"
slug: {blog_slug}
permalink: /blog/
---

"""
index_front_matter += "## Welcome to The {{ page.title }} Blog\n\n"

journal_path = f"{folder_name}/journal.md"
output_path = f"{folder_name}/_posts/"
slicer = "-" * 80

Path(output_path).mkdir(exist_ok=True)

counter = -1
dates = []
date_next = False
with open(journal_path, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.rstrip()
        if date_next:
            try:
                adate = line[3:]
                date_next = False
                adatetime = parser.parse(adate).date()
            except:
                adatetime = None
            dates.append(adatetime)
            date_next = False
        if line == slicer:
            date_next = True
            counter = counter + 1
dates.reverse()
stats = Counter()
stats["entries"] = counter

# This can be greatly simplified by removing the category code

table = []
categories = set()
at_top = True
index_list = []
category_lists = {}
with open(journal_path, "r", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        line = line.rstrip()
        if line == slicer:
            if at_top:
                at_top = False
                table = []
                continue
            try:
                adatetime = dates[counter]
            except:
                adatetime = dates[0]
            filename = f"{output_path}{adatetime}-post-{counter}.md"
            if adatetime:
                print(filename)
                with open(filename, "w", encoding="utf-8") as fw:
                    title = f"Post {counter}"
                    slug = title
                    if table[0] == slicer:
                        table = table[1:]
                    maybe = table[1]
                    couldbe = False
                    has_title = False
                    if table and maybe and maybe[0] == "#":
                        title = maybe[maybe.find(" ") + 1 :]
                        has_title = True
                        couldbe = table[2]
                    slug = title.replace("'", "")
                    slug = slugify(slug)
                    top = []
                    top.append("---\n")
                    top.append("layout: post\n")
                    top.append(f'title: "{title}"\n')
                    top.append(f'author: "{author}"\n')
                    top.append(f"slug: {slug}\n")
                    top.append(f"permalink: /{blog_slug}/{slug}/\n")
                    try:
                        fdate = adatetime.strftime("%m/%d/%Y")
                    except:
                        fdate = None
                    link = f"- [{title}](/{blog_slug}/{slug}/) {fdate}"
                    index_list.append(link)
                    if couldbe and couldbe[:8].lower() == "category":
                        category = couldbe[10:]
                        categories.add(category)
                        top.append(f"{couldbe}\n")
                        if category not in category_lists:
                            category_lists[category] = []
                        category_lists[category].append(link)
                    top.append("---\n")
                    top.append("\n")
                    top_chop = 2
                    if has_title:
                        top_chop = 3
                    table = [f"{x}\n" for x in table[top_chop:]]
                    table = top + table
                    # print("".join(table))
                    fw.writelines(table)
            counter = counter - 1
            table = []
        table.append(line)

print()
print(f"ENTRIES: {stats['entries']}")
categories = {x for x in categories if x}


# Replace this with outputting a subset of it to folder_name/_inculdes
index_page = index_front_matter + "\n\n" + "\n".join(index_list)
with open(f"{folder_name}/blog.md", "w", encoding="utf-8") as fh:
    fh.writelines(index_page)


index_page = "\n".join(index_list)
with open(f"{folder_name}/_includes/posts-main.html", "w", encoding="utf-8") as fh:
    fh.writelines(index_page)

for category in categories:
    slug_cat = slugify(category)
    category_front_matter = f"""---
layout: default
author: {author}
title: "{blog_title} {category}"
slug: {blog_slug}
permalink: /{slug_cat}/
---

# {category}

"""
    cat_file = f"{folder_name}/category-{slug_cat}.md"
    cat_list = []
    for acat in categories:
        if acat != category:
            cat_list.append(f"- [{acat}](/{slugify(acat)}/)")
    category_page = category_front_matter + "\n".join(category_lists[category])
    category_page += "\n\n## Other Categories: \n"
    category_page += "\n".join(sorted(cat_list))
    with open(cat_file, "w", encoding="utf-8") as fh:
        fh.writelines(category_page)
print("Sliced successfully!")
