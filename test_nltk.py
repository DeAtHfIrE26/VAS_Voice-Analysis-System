import nltk
import sys

print("Testing NLTK installation and resources...")

# Try downloading resources
try:
    nltk.download('punkt', quiet=False)
    print("✓ Successfully downloaded punkt")
except Exception as e:
    print(f"✗ Failed to download punkt: {str(e)}")

try:
    nltk.download('stopwords', quiet=False)
    print("✓ Successfully downloaded stopwords")
except Exception as e:
    print(f"✗ Failed to download stopwords: {str(e)}")

try:
    nltk.download('averaged_perceptron_tagger', quiet=False)
    print("✓ Successfully downloaded averaged_perceptron_tagger")
except Exception as e:
    print(f"✗ Failed to download averaged_perceptron_tagger: {str(e)}")

try:
    nltk.download('wordnet', quiet=False)
    print("✓ Successfully downloaded wordnet")
except Exception as e:
    print(f"✗ Failed to download wordnet: {str(e)}")

# Test resource availability
try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    sample_text = "Hello world. This is a test sentence."
    words = word_tokenize(sample_text)
    sentences = sent_tokenize(sample_text)
    print(f"✓ Tokenization works! Found {len(words)} words and {len(sentences)} sentences.")
except Exception as e:
    print(f"✗ Tokenization failed: {str(e)}")

try:
    from nltk.corpus import stopwords
    stops = stopwords.words('english')
    print(f"✓ Stopwords work! Found {len(stops)} English stopwords.")
except Exception as e:
    print(f"✗ Stopwords failed: {str(e)}")

try:
    from nltk import pos_tag
    tagged = pos_tag(word_tokenize("The quick brown fox jumps over the lazy dog."))
    print(f"✓ POS tagging works! Sample: {tagged[:3]}")
except Exception as e:
    print(f"✗ POS tagging failed: {str(e)}")

# Check NLTK data path
print(f"NLTK data path: {nltk.data.path}")

# Print Python version
print(f"Python version: {sys.version}")