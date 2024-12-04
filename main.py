from utils import *
from config import *
from extract_and_process_audio import extract_and_process_audio

def main():
    print("Starting the audio extraction process...")
    # extract_and_process_audio()
    print("Audio Processing Complete!")
    print("Splitting data...")
    split_data()

if __name__ == "__main__":
    main()
