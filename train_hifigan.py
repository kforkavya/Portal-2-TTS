from common_utils import *
from config import *

def main(create_mels=False, create_train_valid=False):
    # Save the mel spectrograms as .npy files
    if (not os.path.exists(MEL_DIR)) or create_mels:
        save_mel_spectrograms(AUDIO_DIR, MEL_DIR)
        print("Mels created!")

    # Create training.txt and validation.txt
    if create_train_valid:
        try:
            for manifest_file, hifigan_file in [(MANIFEST_TRAIN, HIFIGAN_TRAIN), (MANIFEST_VALIDATION, HIFIGAN_VALIDATION)]:
                data = [json.loads(line) for line in open(manifest_file, 'r')] # Reading JSON file
                if os.path.exists(hifigan_file):
                    os.remove(hifigan_file)
                with open(hifigan_file, 'w') as f:
                    for entry in data:
                        filename = entry["audio_filepath"].split('/')[1][:-4] # Removing the .wav extension
                        text = entry["text"]
                        f.write(f"{filename}|{text}\n")
        except Exception as e:
            print(f"Error create_train_valid: {e}")
            exit(1)

    # Train
    os.makedirs(CHECKPOINT_HIFIGAN_DIR, exist_ok=True)
    os.chdir(HIFIGAN_DIR)
    try:
        os.system(f"python3 train.py \
                --input_training_file {HIFIGAN_TRAIN} \
                --input_validation_file {HIFIGAN_VALIDATION} \
                --input_wavs_dir {AUDIO_DIR} \
                --input_mels_dir {MEL_DIR} \
                --checkpoint_path {CHECKPOINT_HIFIGAN_DIR} \
                --config config_v1.json")
    except Exception as e:
        print("Error!!!")
        print(e)
    finally:
        os.chdir(MAIN_DIR)

if __name__ == "__main__":
    main()
