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

def extract_mel_spectrogram(audio_file, sr=TARGET_SAMPLING_RATE, n_fft=2048, hop_length=512, n_mels=80, max_wav_value=32768.0):
    """Extract Mel spectrogram matching the PyTorch tensor-based implementation."""
    # Load audio and ensure sampling rate matches
    audio, _ = librosa.load(audio_file, sr=sr)
    
    # Normalize audio to match the PyTorch implementation
    audio_norm = audio / max_wav_value
    
    # Add batch dimension to audio (like unsqueeze in PyTorch)
    audio_norm = np.expand_dims(audio_norm, axis=0)
    
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_norm[0],
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    return mel_spectrogram

def preprocess_text(text):
    """Preprocess the text by normalizing numbers and cleaning."""
    # Normalize numbers
    text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)
    # Remove unwanted characters (you can customize this based on your needs)
    text = re.sub(r"[^\w\s]", '', text)  # Remove punctuation
    text = text.lower()  # Convert text to lowercase
    return text

def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

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
