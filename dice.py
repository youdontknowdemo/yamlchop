import argparse


aparser = argparse.ArgumentParser()
add_arg = aparser.add_argument
add_arg("-f", "--folder", required=True)
args = aparser.parse_args()

print(f"Hello World {args.folder}")
apath = args.folder
myloc = apath.split("/")[-1]
print(f"My Folder is {myloc}")
journal = "journal.md"  # Maybe get this from CLI args
file = f"/home/ubuntu/repos/hide/{myloc}/{journal}"


def split_file(filename):
    delimiter = 80 * '-'
    with open(filename) as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if delimiter in line:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


for i, chunk in enumerate(split_file(file)):
    with open(f'/home/ubuntu/repos/hide/{myloc}/files/file_{i}.txt', 'w') as f:
        f.writelines(chunk)

print("Done")


