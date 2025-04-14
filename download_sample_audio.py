import os
import urllib.request

def download_sample_audio():
    """Download a sample audio file for testing the MemoTag system."""
    # Create assets directory if it doesn't exist
    os.makedirs("assets", exist_ok=True)
    
    # URL for a sample audio file (public domain)
    sample_url = "https://cdn.freesound.org/previews/414/414345_3576657-lq.mp3"
    output_path = os.path.join("assets", "sample_speech.mp3")
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"Sample audio already exists at {output_path}")
        return output_path
    
    # Download the file
    print(f"Downloading sample audio file to {output_path}...")
    urllib.request.urlretrieve(sample_url, output_path)
    print("Download complete!")
    
    return output_path

if __name__ == "__main__":
    file_path = download_sample_audio()
    print(f"Sample audio file is available at: {file_path}")
    print("You can use this file to test the MemoTag system.")
    print("To run the system with this sample, execute one of the following:")
    print("  - Windows: run_app.bat")
    print("  - macOS/Linux: ./run_app.sh")
    print("Then upload the sample file through the web interface.") 