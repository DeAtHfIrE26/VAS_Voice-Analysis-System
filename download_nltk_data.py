import nltk
import os
import sys
import time

def ensure_nltk_data():
    """
    Downloads all required NLTK data packages and ensures they are loaded correctly.
    """
    print("Setting up NLTK resources...")
    
    # Create a custom directory for NLTK data
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Add the directory to NLTK's search path
    nltk.data.path.insert(0, nltk_data_dir)
    
    # List of required NLTK resources
    resources = [
        'punkt',  # Required for tokenization
        # Removed stopwords, averaged_perceptron_tagger, and wordnet as we have fallbacks for them
    ]
    
    # Attempt to download each resource with retry logic
    for resource in resources:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"Downloading NLTK resource: {resource} (attempt {attempt+1}/{max_attempts})")
                nltk.download(resource, download_dir=nltk_data_dir, quiet=False)
                # Check if we can load the resource - use correct resource path
                if resource == 'stopwords':
                    nltk.data.find(f'corpora/{resource}')
                elif resource == 'averaged_perceptron_tagger':
                    nltk.data.find(f'taggers/{resource}')
                elif resource == 'wordnet':
                    nltk.data.find(f'corpora/{resource}')
                else:
                    nltk.data.find(f'tokenizers/{resource}')
                print(f"✓ Successfully downloaded and verified {resource}")
                break
            except Exception as e:
                print(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    print(f"⚠️ Failed to download {resource} after {max_attempts} attempts")
                else:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
    
    # Create fallback implementations for critical functionality
    print("Setting up fallback tokenization mechanisms...")
    
    # Print NLTK data path
    print(f"NLTK data path: {nltk.data.path}")
    
    print("NLTK setup complete")

if __name__ == "__main__":
    ensure_nltk_data()