MAIN_DIR = "/home/Portal-2-TTS" # Change as per your convenience

def assign_path(relative_path):
    return MAIN_DIR + "/" + relative_path

# Directories
AUDIO_DIR = assign_path("audio")
TEMP_DIR = assign_path("temp_audio")
MEL_DIR = assign_path("mels")
TACOTRON2_DIR = assign_path("tacotron2")
HIFIGAN_DIR = assign_path("hifi-gan")
CHECKPOINT_TACOTRON2_DIR = assign_path("checkpoints_tacotron2")
CHECKPOINT_HIFIGAN_DIR = assign_path("checkpoints_hifigan")
TEST_HIFIGAN_DIR = assign_path("test_hifigan")

# Sources and blocklist
SOURCES = [
    "https://theportalwiki.com/wiki/GLaDOS_voice_lines_(Portal)", 
    "https://theportalwiki.com/wiki/GLaDOS_voice_lines_(Portal_2)"
]
BLOCKLIST = ["potato", "_ding_", "00_part1_entry-6", "_escape_"]

# Audio parameters
TARGET_SAMPLING_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
MAX_WAV_VALUE = 32768.0

# Threading parameters
MAX_THREADS = 16

# Manifest files
MANIFEST_FILE = assign_path("manifest.json")
MANIFEST_VALIDATION = assign_path("manifest_validation.json")
MANIFEST_TRAIN = assign_path("manifest_train.json")

# Data file
TRAINING_DATA_FILE = assign_path("training_data.pkl")
VALIDATION_DATA_FILE = assign_path("validation_data.pkl")
TACOTRON2_TRAINED_FILE = assign_path("tacotron2_trained.pt")

# Tacotron2 Training Parameters
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE=1e-3
NUM_EPOCHS = 100

# Config files
CONFIG_HIFIGAN = assign_path("config_hifigan.json")

# Hifigan files
HIFIGAN_TRAIN = assign_path("hifigan_training.txt")
HIFIGAN_VALIDATION = assign_path("hifigan_validation.txt")
