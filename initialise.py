import json
import os

from feature_extractor import FeatureExtractor
from normalizer import Normalizer
from reader import PSBDataset


def initialise():
    while True:
        print("#" * 10 + " Initialisation menu " + "#" * 10)
        print('''
            1) Check paths and run pipelines
            2) Set new configuration paths
            3) Print current configuration paths
            4) Exit\n 
            Type a number 1 - 4. \n
        ''')
        choice = input(">> ")
        if choice == "1":
            initialise_everything()
        elif choice == "2":
            set_new_config()
        elif choice == "3":
            print_configuration_paths()
        elif choice == "4":
            print("Thanks, bye!")
            break
        else:
            print("\nPlease enter a valid number.")


def prompt_for_class_files(new_path):
    with open('config.json') as f:
        data = json.load(f)
    base = False
    coarse = False
    for root, dirs, files in os.walk(new_path):
        if "coarse" in root and not coarse:
            print(f"- New path to the coarse classification file found at: {root}")
            data["CLASS_FILE_COARSE"] = root
            coarse = True
        if "base" in root:
            print(f"\n\n- New path to the fine grained classification file found at: {root}")
            data["CLASS_FILE"] = root
            base = True
    if not base and not coarse:
        print("The system could not find a classification file in the path to the shape database.\n")
        choice = input("Do you wish to select classification files now? (y/n)\n>> ")
        if choice == "y":
            new_path = input("Enter the new path to the fine grained classification file.\n>> ")
            data["CLASS_FILE"] = new_path
            new_path = input("Enter the new path to the coarse classification file.\n>> ")
            data["CLASS_FILE_COARSE"] = new_path


def set_new_config():
    with open('config.json') as f:
        data = json.load(f)

    print("Old configuration:")
    print_configuration_paths()
    print("Press 7 to exit.")

    choice = input("What do you want to change? (1 - 7)\n>> ")
    if choice == "1":
        new_path = input("Enter the new path to the original shapes database.\n>> ")
        data["DATA_PATH_PSB"] = new_path
        prompt_for_class_files(new_path)
    elif choice == "2":
        new_path = input("Enter the new path to the normalised shapes database.\n>> ")
        data["DATA_PATH_NORMED"] = new_path
    elif choice == "3":
        new_path = input("Enter the new path to the fine grained classification file.\n>> ")
        data["CLASS_FILE"] = new_path
    elif choice == "4":
        new_path = input("Enter the new path to the coarse classification file.\n>> ")
        data["CLASS_FILE_COARSE"] = new_path
    elif choice == "5":
        new_path = input("Enter the new path to the generated features .csv file.\n>> ")
        data["FEATURE_DATA_FILE"] = new_path
    elif choice == "6":
        new_path = input("Enter the new path to the generated statistics .csv file.\n>> ")
        data["STAT_PATH"] = new_path
    elif choice == "7":
        return
    else:
        print("Enter a valid number.")
        return

    with open('config.json', 'w') as outfile:
        json.dump(data, outfile)
    print("Path updated.")


def initialise_everything():
    print('''This procedure can take up to hours to finish.
    The program will now run:
     - Normalisation pipeline over the shape database. (~3hrs)
     - Feature extraction over shape database. (~2hrs)\n
     Are you sure you want to continue (y/n)?\n
    ''')
    choice = input(">> ")
    if choice == "n" or choice == "no":
        return

    with open('config.json') as f:
        data = json.load(f)
    path_psd = data["DATA_PATH_PSB"]
    path_normed = data["DATA_PATH_NORMED"]
    path_feature = data["FEATURE_DATA_FILE"]
    db = PSBDataset()
    if len(os.listdir(path_psd)) == 0:
        print("No valid dataset found.\nPoint to a valid dataset.")
        return
    else:
        prompt_for_class_files(path_psd)
        choice = input("Do you wish to go back to the menu to change the current classification settings? (y/n)\n>> ")
        if choice == "n":
            return
    if not os.path.isfile(path_normed):
        print("No valid normalised dataset found.\nRunning normalisation.")
        norm = Normalizer(db)
        norm.run_full_pipeline()
    if not os.path.isfile(path_feature):
        print("No valid feature file found.\nRun feature extraction.")
        FE = FeatureExtractor(db)
        FE.run_full_pipeline()


def generate_default():
    data = {"DATA_PATH_PSB": "", "DATA_PATH_NORMED": "", "CLASS_FILE": "", "CLASS_FILE_COARSE": "",
            "FEATURE_DATA_FILE": "", "STAT_PATH": ""}
    if not os.path.isfile('config.json'):
        print("~" * 35 + "\nNo valid config settings found.\nGenerating defaults.\n" + "~" * 35)
        with open('config.json', 'w') as outfile:
            json.dump(data, outfile)

    with open('config.json') as f:
        data = json.load(f)

    to_working_directory = os.getcwd()
    to_shape_database = os.path.join(to_working_directory, "datasets", "psb")
    to_shape_normed_database = os.path.join(to_working_directory, "datasets", "normed_psb")
    to_classification_file = os.path.join(to_working_directory, "datasets", "psb", "benchmark", "classification", "v1",
                                          "base")
    to_coarse_classification_file = os.path.join(to_working_directory, "datasets", "psb", "benchmark", "classification",
                                                 "v1", "coarse1")
    to_feature_file = os.path.join(to_working_directory, "stats", "2020-11-02-features.jsonl")
    to_stat_dir = os.path.join(to_working_directory, "stats")

    if data["DATA_PATH_PSB"] == "":
        data["DATA_PATH_PSB"] = str(to_shape_database)
    if data["DATA_PATH_NORMED"] == "":
        data["DATA_PATH_NORMED"] = str(to_shape_normed_database)
    if data["CLASS_FILE"] == "":
        data["CLASS_FILE"] = str(to_classification_file)
    if data["CLASS_FILE_COARSE"] == "":
        data["CLASS_FILE_COARSE"] = str(to_coarse_classification_file)
    if data["FEATURE_DATA_FILE"] == "":
        data["FEATURE_DATA_FILE"] = str(to_feature_file)
    if data["STAT_PATH"] == "":
        data["STAT_PATH"] = str(to_stat_dir)

    with open('config.json', 'w') as outfile:
        json.dump(data, outfile)

    prompt_for_class_files(data["DATA_PATH_PSB"])

    print("\nDefault values for database and files are as follows:\n")
    print_configuration_paths()
    choice = input("\n\nDo you wish to apply further changes to the default values or "
                   "run feature extraction/normalisation pipelines(y/n)?\n>> ")
    if choice == "y":
        initialise()
    else:
        return


def print_configuration_paths():
    with open('config.json') as f:
        data = json.load(f)
    print(f'''
    1) Path to original shapes database: {data["DATA_PATH_PSB"]}
    2) Path to normalised shapes database: {data["DATA_PATH_NORMED"]}
    3) Path to classification file: {data["CLASS_FILE"]}
    4) Path to coarse classification file: {data["CLASS_FILE_COARSE"]}
    5) Path to generated features .csv file: {data["FEATURE_DATA_FILE"]}
    6) Path to generated statistics .csv file: {data["STAT_PATH"]}\n
    ''')


if __name__ == "__main__":
    generate_default()
