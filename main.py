from utils import *
from config import *
from extract_and_process_audio import extract_and_process_audio
from tacotron2.model import Tacotron2
from tacotron2.loss_function import Tacotron2Loss
from torch.utils.data import DataLoader
import torch.optim as optim
from custom_dataset import CustomDataset

def preprocess():
    print("Starting the audio extraction process...")
    extract_and_process_audio()
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

def main():
    # Get the data
    training_data, val_data = get_data()

    # Make the dataset classes
    train_dataset = CustomDataset(training_data)
    val_dataset = CustomDataset(val_data)

    # Load the datasets
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device is", device)

    # Initialize the Tacotron 2 model
    model = Tacotron2().to(device)

    # Loss function
    criterion = Tacotron2Loss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
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

        # Save checkpoints
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/tacotron2_epoch_{epoch + 1}.pt")

if __name__ == "__main__":
    main()
