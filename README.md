# Chop The YAML!

USAGE: `python ~/repos/yamlchop/chop.py -f ~/[path_to_jekyll_repo]/_drafts/journal.md`

```code
   __   __ _    __  __ _         _                      chop
   \ \ / // \  |  \/  | |    ___| |__   ___  _ __        _|   chop
    \ V // _ \ | |\/| | |   / __| '_ \ / _ \| '_ \      | |     |   chop
     | |/ ___ \| |  | | |__| (__| | | | (_) | |_) |  _  | | ___ |     |  chop
     |_/_/   \_\_|  |_|_____\___|_| |_|\___/| .__/  | |_| |/ _ \|_   _|    |   chop
                                            |_|      \___/| (_) | | | |_ __|     |   chop
     I'm building better blogging                         |\___/| |_| | '__|_ __ |     |
     To replace a bygone era              ___             |     |\__,_| |  | '_ \| __ _|
     Where your CMS is flogging          |   |         _____    |     |_|  | | | |/ _` |
     You to take up something clearer,   |_  |        /     \         |    |_| |_| (_| |
     So you catalog your cat tales         \ |       |       \        |    |     |\__,_|
     In one file for the hashing           |  \      |       /             |     |     |
     Using YAML for the data                \  \____ \_      \                   |     |
     And then Jekyll for the lashing.        \      \_/      |                         |
                                       ___.   \_            _/                          _
                      .-,             /    \    |          |                            _
                      |  \          _/      `--_/           \_                          _
                       \  \________/                     /\   \                        | |
                       |                                /  \_  \                       | |
                       `-----------,                   |     \  \                      | |
                                   |                  /       \  |                     |_|
                                   |                 |         | \
                                   /                 |         \__|
                                  /   _              |
                                 /   / \_             \
                                 |  /    \__      __--`
                                _/ /        \   _/
                            ___/  /          \_/
                           /     /
                           `----`
```

## What is this?

So you know the Jekyll static site generator (SSG) in Github called `github.io`
or `Github Pages`? It takes a repo with simple text files with YAML
front-matter and Markdown content to publish slick low tech-liability HTML
pages like so:

    ┌──────────────────────┐          ┌────────────────────────────────┐
    │title: My Blog Post   │          │<html>                          │
    │date: 2021-05-09      │          │  <head>                        │
    │---                   │  Jekyll  │    <title>My Blog Post</title> │
    │                      │─────────►│  </head>                       │
    │# My Blog Post        │          │  <body>                        │
    │                      │          │    <h1>My Blog Post</h1>       │
    │This is my blog post. │          │    <p>This is my blog post.</p>│
    └──────────────────────┘          │  </body>                       │
                                      │</html>                         │
                                      └────────────────────────────────┘

This is a good system, but over time you might accumulate hundreds of these
files, and it can be a pain to manage them all:

      File 1    File 2    File 3    File 4    File 5    File 6
     ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐ 
     │Page 1│  │Page 2│  │Page 3│  │Page 4│  │Page 5│  │Page 6│ 
     │      │  │      │  │      │  │      │  │      │  │      │ 
     │      │  │      │  │      │  │      │  │      │  │      │ 
     └──────┘, └──────┘, └──────┘, └──────┘, └──────┘, └──────┘

The contents of these files are so lightweight, it's a shame to have to manage
them all individually. What if you could keep them all in one big file like
this:

     File 1
    ┌──────────────────────────────────────────────────────────┐
    │┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐│
    ││Page 1│  │Page 2│  │Page 3│  │Page 4│  │Page 5│  │Page 6││
    ││      │  │      │  │      │  │      │  │      │  │      ││
    ││      │  │      │  │      │  │      │  │      │  │      ││
    │└──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘│
    └──────────────────────────────────────────────────────────┘

And then you can use yamlchop/chop.py to split them out into individual files
like:

      ┌────────┐
      │YAML    │      ┌──────┐
      │---     │      │      │
      │Markdown│─────►│ HTML │
      │        │      │      │
      │        │      └──────┘
      │--------│ 
      │YAML    │      ┌──────┐  
      │---     │      │      │  
      │Markdown│─────►│ HTML │  
      │        │      │      │ 
      │        │      └──────┘ 
      │--------│
      │YAML    │      ┌──────┐ 
      │---     │      │      │ 
      │Markdown│─────►│ HTML │ 
      │        │      │      │ 
      │        │      └──────┘ 
      │--------│
      │YAML    │      ┌──────┐ 
      │---     │      │      │ 
      │Markdown│─────►│ HTML │ 
      │        │      │      │ 
      │        │      └──────┘ 
      │--------│
      │YAML    │      ┌──────┐ 
      │---     │      │      │ 
      │Markdown│─────►│ HTML │ 
      │        │      │      │ 
      │        │      └──────┘ 
      │--------│
      │YAML    │      ┌──────┐ 
      │---     │      │      │ 
      │Markdown│─────►│ HTML │ 
      │        │      │      │ 
      │        │      └──────┘ 
      └────────┘

## It Uses AI, Right? Okay, Sure.

During this process, if there are any posts for which you have given a title
(the indication to release), OpenAI will step in and write a headline, a meta
description, and a keyword field with extracted keywords. You'll need an
`openai.txt` file with your API key in it in the yamlchop repo folder.
`.gitignore` will keep it from getting committed.
    
     ┌──────────────────────┐     ┌──────────────────────┐
     │title: My Blog Post   │     │title: My Blog Post   │
     │date: 2021-05-09      │     │date: 2021-05-09      │
     │---                   │     │headline: [OpenAI]    │
     │                      │────►│description: [OpenAI] │
     │# My Blog Post        │     │keywords: [OpenAI]    │
     │                      │     │---                   │
     │This is my blog post. │     │                      │
     └──────────────────────┘     │# My Blog Post        │
                                  │                      │
                                  │This is my blog post. │
                                  └──────────────────────┘
    
## Tell Them About the Arrows!

Since the pages have a natural sequential order, that can help with Web
navigation. Imagine the previous next arrows you could build, given these
relationships:

        TOPIC A            TOPIC B            TOPIC C    
     ┌──────────┐   B   ┌──────────┐   C   ┌──────────┐ 
     │          │──────►│          │──────►│          │ 
     │          │       │          │       │          │ 
     │   HTML   │       │   HTML   │       │   HTML   │ 
     │          │   A   │          │   B   │          │ 
     │          │◄──────│          │◄──────│          │ 
     └──────────┘       └──────────┘       └──────────┘ 

In other words, our blog posts get previous and next arrows that have the exact
wording of your title tag in the links leading to each page. From a search
engine optimization standpoint, this is pretty sweet. You can see an example
and read lots of rambling at
[https://mikelev.in/blog/](https://mikelev.in/blog/).

## How do I use it?

1. Create a file called `journal.md` in the `_drafts` folder of your Jekyll
   site. This is where you'll keep all your pages.

2. Add a YAML front-matter block to the top of the file, like this:

        -------------------------------------------------------------------------------- 
        date: 2021-05-09
        [triple dash]

3. Add a Markdown block to the bottom of the file, like this:

        # My Blog Post

        This is my blog post.

4. Repeat steps 2 and 3 for each page you want to add to your site.

5. Only give YAML title fields to the posts you want published. Everything
   else will be considered a draft. OpenAI will write your headline, meta
   description and extract keywords for you and organize categories.

6. Run: `python ~/repos/yamlchop/chop.py -f ~/[path_to_jekyll_repo]/_drafts/journal.md`

7. This will git commit and push your change.

If anything needed OpenAI fields written, the automatic git commit and push
that publishes the site will not occur until the next time you run that
command, assuming you wish to have a chance to review and edit the OpenAI
fields.

## License

This software is provided under a very liberal MIT license that only requires
you to include the license text in any redistribution. See the LICENSE file for
details.
