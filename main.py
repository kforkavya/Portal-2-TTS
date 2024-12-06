from utils import *
from config import *
from extract_and_process_audio import extract_and_process_audio

def preprocess():
    print("Starting the audio extraction process...")
    # extract_and_process_audio()
    print("Audio Processing Complete!")
    print("Splitting data...")
    split_data()

def main():
    if not (os.path.exists(TRAINING_DATA_FILE) and os.path.exists(VALIDATION_DATA_FILE)):
        preprocess()
    for data_file, manifest_file in [(TRAINING_DATA_FILE, MANIFEST_TRAIN), (VALIDATION_DATA_FILE, MANIFEST_VALIDATION)]:
        if not os.path.exists(data_file):
            data = create_data(manifest_file)
            save_data(data, data_file)
    print("Training and Validation data creation is complete!")

if __name__ == "__main__":
    main()
