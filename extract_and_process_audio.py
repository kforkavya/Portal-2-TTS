from imports import *
from config import *
from utils import *

lock = Lock()

def fetch_urls():
    """Fetch URLs of audio files from the sources."""
    print("Fetching URLs...")
    urls = []
    for source in SOURCES:
        response = requests.get(source)
        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a"):
            url = link.get("href")
            if url and url.endswith(".wav") and all(b not in url for b in BLOCKLIST):
                filename = url.split("/")[-1]
                urls.append((url, filename))
    print(f"Found {len(urls)} audio files.")
    return urls

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

def download_all(urls):
    """Download all audio files in parallel."""
    downloaded_files = set()
    with ThreadPoolExecutor(MAX_THREADS) as executor:
        tasks = [
            executor.submit(download_audio, url, filename, downloaded_files)
            for url, filename in urls
        ]
        for task in tasks:
            task.result()  # Wait for all downloads to complete

def resample_audio(input_path, output_path):
    """Resample audio to the target sampling rate."""
    audio, sr = librosa.load(input_path, sr=TARGET_SAMPLING_RATE)
    sf.write(output_path, audio, samplerate=TARGET_SAMPLING_RATE)

def extract_and_process_audio():
    """Fetch, download, and resample audio files."""
    # Delete if already existed
    delete_folder(AUDIO_DIR)
    delete_folder(TEMP_DIR)

    # Create directories
    os.makedirs(AUDIO_DIR)
    os.makedirs(TEMP_DIR)

    # Fetch URLs
    urls = fetch_urls()

    # Download all audio files
    download_all(urls)

    # Resample audio files
    print("Resampling audio files...")
    for file in os.listdir(TEMP_DIR):
        if file.endswith(".wav"):
            input_path = os.path.join(TEMP_DIR, file)
            output_path = os.path.join(AUDIO_DIR, file)
            resample_audio(input_path, output_path)

    # Clean up temporary files
    delete_folder(TEMP_DIR)
