from imports import *
from config import *

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

def split_data(test_ratio=0.1):
    with open(MANIFEST_FILE, 'r') as f:
        total_data_size = len(f.readlines())
    test_data_size = int(total_data_size * test_ratio)
    os.system(f"cat {MANIFEST_FILE} | tail -n {test_data_size} > {MANIFEST_VALIDATION}")
    os.system(f"cat {MANIFEST_FILE} | head -n -{test_data_size} > {MANIFEST_TRAIN} ")

def extract_mel_spectrogram(audio_file, sr=TARGET_SAMPLING_RATE, n_fft=2048, hop_length=512, n_mels=80):
    """Extract Mel spectrogram from the audio file."""
    audio, _ = librosa.load(audio_file, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convert to dB scale
    return mel_spectrogram

def preprocess_text(text):
    """Preprocess the text by normalizing numbers and cleaning."""
    # Normalize numbers
    text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)
    # Remove unwanted characters (you can customize this based on your needs)
    text = re.sub(r"[^\w\s]", '', text)  # Remove punctuation
    text = text.lower()  # Convert text to lowercase
    return text

def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def create_data(manifest_file):
    data = []
    with open(manifest_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Extract Mel-spectrogram features from the audio file
            mel_spectrogram = extract_mel_spectrogram(item["audio_filepath"])
            data.append((mel_spectrogram, item["text"]))
    return data
