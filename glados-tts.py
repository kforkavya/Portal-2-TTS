from config import *
from common_utils import *

# Load Tacotron2 Model
def load_tacotron2(checkpoint_path):
    from my_hparams import create_hparams
    # my_hparams.py has a sys.path.append('tacotron2') hence the above line will do the job of inserting path of tacotron2/
    try:
        from model import Tacotron2
        model_var = Tacotron2(create_hparams())
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        model_var.load_state_dict(checkpoint['model_state_dict'])
        model_var.eval()
    except Exception as e:
        print("Error load_tacotron2:", e)
    finally:
        sys.path.remove('tacotron2')  # Clean up by removing the added path
    return model_var

# Load HiFi-GAN Model
def load_hifigan(checkpoint_path):
    print(sys.path)
    sys.path.insert(0, HIFIGAN_DIR) # Add HIFIGAN_DIR to the Python path temporarily
    try:
        from models import Generator
        model = Generator()
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['generator'])
        model.eval()
        model.remove_weight_norm()
    except Exception as e:
        print("Error load_hifigan:", e)
    finally:
        sys.path.remove(HIFIGAN_DIR)  # Clean up by removing the added path
    return model

# Text-to-Mel Conversion
def generate_mel(input_text, tacotron2):
    sys.path.insert(0, TACOTRON2_DIR) # Add TACOTRON2_DIR to the Python path temporarily
    try:
        from text import text_to_sequence
        sequence = text_to_sequence(preprocess_text(input_text), ['english_cleaners'])
        sequence = torch.tensor(sequence).unsqueeze(0).to(torch.device('cpu'))
        with torch.no_grad():
            mel_outputs, _, _, _ = tacotron2.inference(sequence)
    except Exception as e:
        print("Error generate_mel:", e)
    finally:
        sys.path.remove(TACOTRON2_DIR)  # Clean up by removing the added path
    return mel_outputs

# Mel-to-Audio Conversion using HiFi-GAN
def generate_audio(mel, hifigan):
    with torch.no_grad():
        y_g_hat = hifigan(mel)
        audio = y_g_hat.squeeze()  # Remove batch and channel dimensions
        audio = audio * MAX_WAV_VALUE  # Scale to int16 range
        audio = audio.cpu().numpy().astype('int16')  # Convert to numpy int16
    return audio

# Main Function
def glados_tts(text):
    # Load models
    print("Loading Tacotron2...")
    if not os.path.exists(TACOTRON2_TRAINED_FILE):
        print("No Tacotron2 checkpoint found. Exiting.")
        exit(1)
    tacotron2 = load_tacotron2(TACOTRON2_TRAINED_FILE)

    print("Loading HiFi-GAN...")
    hifigan_checkpoint = max(
        [os.path.join(CHECKPOINT_HIFIGAN_DIR, f) for f in os.listdir(CHECKPOINT_HIFIGAN_DIR) if f.startswith("g_")],
        key=(lambda x : int(x[-8:])), # Take the checkpoint of the farthest steps calculated
    default=None
    )
    if hifigan_checkpoint is None:
        print("No hifigan checkpoints found. Exiting.")
        exit(1)
    hifigan = load_hifigan(hifigan_checkpoint)

    # Generate mel spectrogram
    print(f"Generating mel spectrogram for: \"{text}\"")
    mel = generate_mel(text, tacotron2)

    # Generate audio
    print("Generating audio...")
    audio = generate_audio(mel, hifigan)

    # Save audio
    sf.write(OUTPUT_WAV_PATH, audio, samplerate=SAMPLING_RATE)
    print(f"Audio saved at {OUTPUT_WAV_PATH}")

# Entry Point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GLaDOS Text-to-Speech")
    parser.add_argument("--text", type=str, required=True, help="Input text for TTS")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file path")
    args = parser.parse_args()

    # Generate TTS
    glados_tts(args.text)
