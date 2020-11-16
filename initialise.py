import json
import os

def initialise():
    print("#" * 10 + " Initialisation menu " + "#" * 10)
    print('''
        1) Initialise everything\n
        2) Run normalisation\n
        3) Run feature extraction\n
        4) Set new configuration paths\n
        5) Print current configuration paths\n\n 
        Type a number 1 - 5. \n\n
    ''')
    choice = input(">> ")


def print_configuration_paths():
    with open('config.json') as f:
        data = json.load(f)
    print(f'''
    - Path to original shapes database: {data["DATA_PATH_PSB"]}\n
    - Path to normalised shapes database: {data["DATA_PATH_NORMED"]}\n
    - Path to classification files: {data["CLASS_FILE"]}\n
    - Path to coarse classification files: {data["CLASS_FILE_COARSE"]}\n
    - Path to generated features .csv file: {data["FEATURE_DATA_FILE"]}\n
    - Path to generated statistics .csv file: {data["STAT_PATH"]}\n
    ''')
