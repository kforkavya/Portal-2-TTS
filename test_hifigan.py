from common_utils import *
from config import *

def main():
    # Find the latest checkpoint file
    latest_checkpoint_file = max(
        [os.path.join(CHECKPOINT_HIFIGAN_DIR, f) for f in os.listdir(CHECKPOINT_HIFIGAN_DIR) if f.startswith("g_")],
        key=(lambda x : int(x[-8:])), # Take the checkpoint of the farthest steps calculated
    default=None
    )
    if latest_checkpoint_file is None:
        print("No checkpoints found. Exiting.")
        exit(1)
    print("Checkpoint file picked is", latest_checkpoint_file)

    # Start Inference
    os.makedirs(TEST_HIFIGAN_DIR, exist_ok=True)
    os.chdir(HIFIGAN_DIR)
    try:
        os.system(f"python3 inference.py \
                --input_wavs_dir {AUDIO_DIR} \
                --output_dir {TEST_HIFIGAN_DIR} \
                --checkpoint_file {latest_checkpoint_file}")
    except Exception as e:
        print("Error!!!")
        print(e)
    finally:
        os.chdir(MAIN_DIR)

if __name__ == "__main__":
    main()
