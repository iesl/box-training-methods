import os
import shutil
import argparse


def main(args):

    for subdir, dirs, files in os.walk(args.rootdir):
        
        if args.deleting_dirs:
            for dir in dirs:
                dir_to_delete = os.path.join(subdir, dir)
                if dir_to_delete.endswith(args.extension_to_delete):
                    shutil.rmtree(dir_to_delete, ignore_errors=False, onerror=None)
                    print("Directory '%s' has been removed successfully" %dir_to_delete)
        
        else:
            for file in files:
                file_to_delete = os.path.join(subdir, file)
                if file_to_delete.endswith(args.extension_to_delete):
                    os.remove(file_to_delete)
                    print("File '%s' has been removed successfully" %file_to_delete)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rootdir", type=str, required=True)
    parser.add_argument("-e", "--extension_to_delete", type=str, required=True, help="delete all files under rootdir with this extension")
    parser.add_argument("--deleting_dirs", action="store_true", help="use this flag if deleting directories whose names have extension; otherwise deleting files")
    args = parser.parse_args()

    main(args)
