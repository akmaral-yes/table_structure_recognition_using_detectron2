import json
import os


def save_dict(output_path, name, dict_to_save):
    """
    Save a dictionary in output_path with the given name
    """
    path = os.path.join(output_path, name)
    with open(path, 'w') as fp:
        json.dump(dict_to_save, fp, sort_keys=True,
                  indent=4, default=lambda o: o.__dict__)


def load_dict(path, name):
    """
    Load a dictionary from output_path with the given name
    """
    path = os.path.join(path, name)
    with open(path) as jfile:
        dict_loaded = json.load(jfile)
    return dict_loaded


def append_to_file(output_path, filename, line):
    """
    Write the given line to the .csv file with the given filename
    """
    csv_path = os.path.join(output_path, filename)
    with open(csv_path, 'a', newline="\n") as f:
        f.write(line)
        f.write("\n")
