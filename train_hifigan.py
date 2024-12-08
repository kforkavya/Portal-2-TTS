from common_utils import *
from config import *

def main(create_mels=False):
    # Save the mel spectrograms as .npy files
    if (not os.path.exists(MEL_DIR)) or create_mels:
        save_mel_spectrograms(AUDIO_DIR, MEL_DIR)



if __name__ == "__main__":
    main(create_mels=True)
