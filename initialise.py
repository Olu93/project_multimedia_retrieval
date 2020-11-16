import json

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
