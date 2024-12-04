from imports import *
from config import *
from utils import *

lock = Lock()

def fetch_urls_and_texts():
    """Fetch URLs of audio files and their corresponding texts from the sources."""
    print("Fetching URLs and texts...")
    urls = []
    filenames = []
    unique_filenames = []
    texts = []
    for source in SOURCES:
        response = requests.get(source)
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a"):
            url = link.get("href")
            if url and url.endswith(".wav") and all(b not in url for b in BLOCKLIST):
                filename = url.split("/")[-1]
                list_item = link.find_parent("li")
                ital_item = list_item.find_all('i')
                if ital_item:
                    text = ital_item[0].text.strip().replace('"', '')
                    if "[" not in text and "]" not in text and "$" not in text:
                        urls.append(url)
                        filenames.append(filename)
                        if filename not in unique_filenames:
                            unique_filenames.append(filename)
                            texts.append(text)
    print(f"Found {len(urls)} audio files.")
    return urls, filenames, unique_filenames, texts

def download_audio(url, filename, downloaded_files):
    """Download a single audio file, ensuring no duplicates."""
    with lock:
        if filename in downloaded_files:
            # If file already downloaded then no need to download again
            return

        downloaded_files.add(filename)  # Mark as downloaded within the lock

    try:
        response = requests.get(url)
        with open(os.path.join(TEMP_DIR, filename), "wb") as f:
            f.write(response.content)
        with lock:
            # Printing needs to have lock too
            print(f"Downloaded: {filename}")
    except Exception as e:
        with lock:
            print(f"Failed to download {url}: {e}")

def download_all(urls, filenames):
    """Download all audio files in parallel."""
    downloaded_files = set()
    with ThreadPoolExecutor(MAX_THREADS) as executor:
        tasks = [
            executor.submit(download_audio, url, filename, downloaded_files)
            for url, filename in zip(urls, filenames)
        ]
        for task in tasks:
            task.result()  # Wait for all downloads to complete

def resample_audio(input_path, output_path):
    """Resample audio to the target sampling rate."""
    audio, sr = librosa.load(input_path, sr=TARGET_SAMPLING_RATE)
    sf.write(output_path, audio, samplerate=TARGET_SAMPLING_RATE)

def create_manifest(audio_dir, texts, filenames, output_file):
    """Creates a manifest file for audio dataset."""
    total_audio_time = 0
    manifest = []

    for i in range(len(filenames)):
        audio_path = os.path.join(audio_dir, filenames[i])
        if not os.path.exists(audio_path):
            print(f"File not found: {audio_path}, skipping...")
            continue

        try:
            item = {
                "audio_filepath": audio_path,
                "text": re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), texts[i]).lower(),
                "duration": audio_duration(audio_path),
            }
            total_audio_time += item["duration"]
            manifest.append(item)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Write manifest to file
    with open(output_file, 'w') as out_file:
        for item in manifest:
            out_file.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
    
    print(f"Manifest created at {output_file}")
    print(f"Total audio time: {total_audio_time / 60:.2f} minutes")

def extract_and_process_audio():
    """Fetch, download, and resample audio files."""
    # Delete if already existed
    delete_folder(AUDIO_DIR)
    delete_folder(TEMP_DIR)

    # Create directories
    os.makedirs(AUDIO_DIR)
    os.makedirs(TEMP_DIR)

    # Fetch URLs and texts
    urls, filenames, unique_filenames, texts = fetch_urls_and_texts()

    # Download all audio files
    download_all(urls, filenames)

    # Resample audio files
    print("Resampling audio files...")
    for file in os.listdir(TEMP_DIR):
        if file.endswith(".wav"):
            input_path = os.path.join(TEMP_DIR, file)
            output_path = os.path.join(AUDIO_DIR, file)
            resample_audio(input_path, output_path)
    print("Resampling completed.")

    # Clean up temporary files
    delete_folder(TEMP_DIR)

    # Create the manifest file after downloading and resampling
    create_manifest(AUDIO_DIR, texts, unique_filenames, MANIFEST_FILE)
