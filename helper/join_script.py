import glob

if __name__ == "__main__":
    filenames = glob.glob("stats/tmp/tmp-*.jsonl")
    with open("computed_features.jsonl", 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())