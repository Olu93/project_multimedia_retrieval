# %%
from glob import glob
from helper.config import CLASS_FILE_COARSE
import io
from pathlib import Path
import jsonlines

def load_classes(class_file_path):
    if not class_file_path:
        return {}
    path_to_classes = Path(class_file_path)
    search_pattern = path_to_classes / "*.cla"
    class_files = list(glob(str(search_pattern), recursive=True))
    class_file_handlers = [io.open(cfile, mode="r") for cfile in class_files]
    class_file_content = [cf.read().split("\n") for cf in class_file_handlers]
    curr_class = None
    class_memberships = {}
    for cfile in class_file_content:
        for line in cfile:
            line_content = line.split()
            line_nr_of_items = len(line_content)
            if line_nr_of_items == 2:
                continue
            if line_nr_of_items == 3:
                curr_class = line_content[0]
            if line_nr_of_items == 1:
                class_memberships[f"m{line_content[0]}"] = curr_class
    return class_memberships
# %%
if __name__ == "__main__":
    data = jsonlines.Reader(io.open("stats/2020-11-02-features.jsonl", "r"))
    writer = jsonlines.Writer(io.open("stats/2020-11-02-features-mod.jsonl", "w"))
    memberships = load_classes(CLASS_FILE_COARSE)
    for line in data:
        line_id = line['name'] 
        line['label_coarse'] = memberships.get(line_id, "no_class")
        writer.write(line)
