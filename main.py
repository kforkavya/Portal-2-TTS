from common_utils import *
from config import *
from extract_and_process_audio import extract_and_process_audio
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from my_hparams import create_hparams
sys.path.append('tacotron2')
from model import Tacotron2
from loss_function import Tacotron2Loss
from data_utils import TextMelCollate

def preprocess():
    print("Starting the audio extraction process...")
    # extract_and_process_audio()
    print("Audio Processing Complete!")
    print("Splitting data...")
    split_data()

def get_data():
    if not (os.path.exists(TRAINING_DATA_FILE) and os.path.exists(VALIDATION_DATA_FILE)):
        preprocess()
        for data_file, manifest_file in [(TRAINING_DATA_FILE, MANIFEST_TRAIN), (VALIDATION_DATA_FILE, MANIFEST_VALIDATION)]:
            if not os.path.exists(data_file):
                data = create_data(manifest_file)
                save_data(data, data_file)
        print("Data creation is complete!")

    training_data = load_data(TRAINING_DATA_FILE)
    val_data = load_data(VALIDATION_DATA_FILE)
    print("Data loaded!")

    return training_data, val_data

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_num = checkpoint['epoch']

    return model, optimizer, epoch_num

def save_checkpoint(model, optimizer, epoch_num, filepath):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def main():
    # Get the data
    training_data, val_data = get_data()

    # Create hyperparameters
    hparams = create_hparams()

    # Make the dataset classes
    train_dataset = CustomDataset(training_data, hparams)
    val_dataset = CustomDataset(val_data, hparams)

    # Load the datasets
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,  num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # If additional configuration is needed
    hparams.sampling_rate = TARGET_SAMPLING_RATE

    # Initialize the Tacotron 2 model
    model = Tacotron2(hparams).to(device)

    # Loss function
    criterion = Tacotron2Loss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Make the checkpoints folder
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Staring from the last calculated epoch
    start_epoch = 0  # Default starting epoch

    # Check if there's an existing checkpoint
    latest_checkpoint = max(
        [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=os.path.getctime,
    default=None
    )

    if latest_checkpoint:
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        model, optimizer, start_epoch = load_checkpoint(latest_checkpoint, model, optimizer)
        print(f"Resuming training from epoch {start_epoch + 1}.")
    else:
        print("No existing checkpoint found. Starting training from scratch.")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"Starting epoch {epoch + 1}...")
        checkpoint_path = f"{CHECKPOINT_DIR}/tacotron2_epoch_{epoch + 1}.pt"

        # Training Loop
        model.train()
        train_loss = 0
        for mel, text in train_loader:
            mel, text = mel.to(device), text.to(device)
            optimizer.zero_grad()
            output, postnet_output, alignments = model(text)
            loss = criterion(output, postnet_output, mel)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for mel, text in val_loader:
                mel, text = mel.to(device), text.to(device)
                output, postnet_output, alignments = model(text)
                loss = criterion(output, postnet_output, mel) 
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, checkpoint_path)
        print(f"Checkpoint saved for epoch {epoch + 1}.")

if __name__ == "__main__":
    main()
