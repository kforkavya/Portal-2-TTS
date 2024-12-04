from imports import *

def delete_folder(folder_path):
    """
    Safely deletes a folder and its contents.
    If the folder does not exist, no error is raised.
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

def audio_duration(file_path):
    """Calculate the duration of an audio file in seconds."""
    with sf.SoundFile(file_path) as f:
        return len(f) / f.samplerate
