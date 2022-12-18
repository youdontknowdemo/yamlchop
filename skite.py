import os
import sys
import shlex
import argparse
import pandas as pd
from time import sleep
from pathlib import Path
from art import text2art
from collections import namedtuple
from subprocess import Popen, PIPE


if os.name == "nt":
    git_exe = r"C:\Program Files\Git\cmd\git.exe"
else:
    git_exe = "/usr/bin/git"
os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = git_exe

lr = "\n"
python = sys.executable
home = Path(os.path.expanduser("~"))
blogslicer = Path(home / Path("repos/skite/slice.py"))
if hasattr(__builtins__, "__IPYTHON__") or __name__ != "__main__":
    is_jupyter = True
    from IPython.display import display, Markdown, HTML

    file = "sites.csv"
    apex = "/mnt/c/Users/mikle/repos/hide/MikeLev.in"
    # apex = ""
else:
    is_jupyter = False
    aparser = argparse.ArgumentParser()
    add_arg = aparser.add_argument
    add_arg("-f", "--file", required=True)
    add_arg("-x", "--apex", required=False)
    args = aparser.parse_args()
    file = args.file
    apex = args.apex



def fig(text, font="Standard"):
    if is_jupyter:
        text_br = text2art(text, font=font).replace(lr, "<br/>")
        text_html = f'<pre style="white-space: pre;">{text_br}</pre>'
        display(HTML(text_html))
    else:
        print(text2art(text, font))


def flush(std):
    for line in std:
        line = line.strip()
        if line:
            print(line)
            sys.stdout.flush()


def git(cwd, args):
    cmd = [git_exe] + shlex.split(args)
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


fig("Making Sites...")

file_obj = Path(file)
df = pd.read_csv(file_obj, delimiter="|")
df = df.applymap(lambda x: x.strip())
df.columns = [x.strip() for x in df.columns]
print(f"INTERPRETER: <<{python}>>")
print(f"SLICER: <<{blogslicer}>>")
sleep(4)
if apex:
    apex = Path(apex).name
    df = df[df["apex"] == apex]
else:
    print("Processing sites:", end="\n\n")
    for site in df[["apex"]].values.tolist():
        print(f"- {site[0]}")
print()
Site = namedtuple("Site", "path, apex, title, gaid")
basepath = "/home/ubuntu/repos/hide/"
for index, series in df.iterrows():
    site = Site(**series.to_dict())
    fig(site.apex, font="Cybermedium")
    here = Path(f"{basepath}{Path(site.path)}")
    [x.unlink() for x in Path(here / "_posts/").glob("*")]
    cmd = f'{python} {blogslicer} -p {here} -t "{site.title}" -s "blog" -a "Mike Levin"'
    print(cmd, end="\n\n")
    with Popen(args=cmd, cwd=here, stdout=PIPE, stderr=PIPE, shell=True) as pout:
        for line in pout.stdout.readlines():
            print(line.decode().strip())
    fig("Github...")
    if is_jupyter:
        continue
    git(here, "add _posts/*")
    git(here, "add _includes/*")
    # git(here, "add category-*")
    git(here, "add assets/images/*")
    git(here, f'commit -am "Pushing {site.apex} to Github..."')
    git(here, "push")
fig("Done!")
