from common_utils import *
from config import *

def main(create_train_valid=False):
    # Create training.txt and validation.txt
    if create_train_valid:
        try:
            for manifest_file, hifigan_file in [(MANIFEST_TRAIN, TACOTRON2_TRAIN), (MANIFEST_VALIDATION, TACOTRON2_VALIDATION)]:
                data = [json.loads(line) for line in open(manifest_file, 'r')] # Reading JSON file
                if os.path.exists(hifigan_file):
                    os.remove(hifigan_file)
                with open(hifigan_file, 'w') as f:
                    for entry in data:
                        filepath = assign_path(entry["audio_filepath"])
                        text = entry["text"]
                        f.write(f"{filepath}|{text}\n")
        except Exception as e:
            print(f"Error create_train_valid: {e}")
            exit(1)

    # Train
    os.makedirs(CHECKPOINT_TACOTRON2_DIR, exist_ok=True)
    logging_dir = os.path.join(CHECKPOINT_TACOTRON2_DIR, "logs")
    os.makedirs(logging_dir, exist_ok=True)
    os.chdir(TACOTRON2_DIR)
    try:
        os.system(f"python3 train.py \
               --output_directory {CHECKPOINT_TACOTRON2_DIR} \
               --log_directory {logging_dir} \
               --hparams 'training_files={TACOTRON2_TRAIN},validation_files={TACOTRON2_VALIDATION},batch_size={BATCH_SIZE}'")
    except Exception as e:
        print("Error!!!")
        print(e)
    finally:
        os.chdir(MAIN_DIR)

    # Find the latest checkpoint file
    latest_checkpoint_file = max(
        [os.path.join(CHECKPOINT_TACOTRON2_DIR, f) for f in os.listdir(CHECKPOINT_TACOTRON2_DIR) if f.startswith("checkpoint_")],
        key=(lambda x : int(x.split('_')[-1])), # Take the checkpoint of the farthest steps calculated
    default=None
    )
    if latest_checkpoint_file is None:
        print("No checkpoints found. Exiting.")
        exit(1)
    print("Checkpoint file picked is", latest_checkpoint_file)
    os.system(f'cp {latest_checkpoint_file} {TACOTRON2_TRAINED_FILE}')
    print("Saved!!!")

if __name__ == "__main__":
    main()
