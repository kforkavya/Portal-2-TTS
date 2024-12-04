from imports import *

def delete_folder(folder_path):
    """
    Safely deletes a folder and its contents.
    If the folder does not exist, no error is raised.
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
