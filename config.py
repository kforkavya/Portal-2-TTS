# Directories
AUDIO_DIR = "audio"
TEMP_DIR = "temp_audio"
CHECKPOINT_DIR = "checkpoints"

# Sources and blocklist
SOURCES = [
    "https://theportalwiki.com/wiki/GLaDOS_voice_lines_(Portal)", 
    "https://theportalwiki.com/wiki/GLaDOS_voice_lines_(Portal_2)"
]
BLOCKLIST = ["potato", "_ding_", "00_part1_entry-6", "_escape_"]

# Audio parameters
TARGET_SAMPLING_RATE = 22050

# Threading parameters
MAX_THREADS = 16

# Manifest files
MANIFEST_FILE = AUDIO_DIR + "/" + "manifest.json"
MANIFEST_VALIDATION = "manifest_validation.json"
MANIFEST_TRAIN = "manifest_train.json"

# Data file
TRAINING_DATA_FILE = "training_data.pkl"
VALIDATION_DATA_FILE = "validation_data.pkl"

BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE=1e-3
NUM_EPOCHS = 100
