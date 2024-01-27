import os
import argparse


def main(args):

    for subdir, dirs, files in os.walk(args.rootdir):        
        for file in files:
            file_to_delete = os.path.join(subdir, file)
            if file_to_delete.endswith(args.extension_to_print):
                with open(file_to_delete, "r") as f:
                    print(f.read())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rootdir", type=str, required=True)
    parser.add_argument("-e", "--extension_to_print", type=str, required=True, help="print contents of all files under rootdir with this extension")
    args = parser.parse_args()

    main(args)
