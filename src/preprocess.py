import json
import nltk
from nltk.tokenize import sent_tokenize

# ------------------------------------------------------------------
# NLTK setup (robust for Windows / venv)
# ------------------------------------------------------------------
nltk.download("punkt")
nltk.download("punkt_tab")


# ------------------------------------------------------------------
# Load textbook text file (expects CLEAN .txt, NOT PDF)
# ------------------------------------------------------------------
def load_text_file(path: str) -> str:
    """
    Load a textbook text file with robust encoding handling.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as file:
            content = file.read()

    # Normalize whitespace
    content = content.replace("\n", " ")
    content = " ".join(content.split())

    return content


# ------------------------------------------------------------------
# Remove references / bibliography section (highly recommended)
# ------------------------------------------------------------------
def remove_references(text: str) -> str:
    """
    Cut off references / bibliography section to avoid
    badly formatted citation text.
    """
    markers = ["references", "bibliography", "reference"]
    lower = text.lower()

    for marker in markers:
        idx = lower.find(marker)
        if idx != -1:
            return text[:idx]

    return text


# ------------------------------------------------------------------
# Sentence tokenization
# ------------------------------------------------------------------
def sentence_tokenize(text: str):
    """
    Tokenize text into sentences.
    """
    return sent_tokenize(text, language="english")


# ------------------------------------------------------------------
# Detect character-spaced garbage text (PDF artifacts)
# ------------------------------------------------------------------
def is_character_spaced(text: str) -> bool:
    """
    Detects text where most 'words' are single characters,
    e.g. 'B r o o k e r , R . A .'
    """
    tokens = text.split()
    if not tokens:
        return True

    single_char_tokens = sum(1 for t in tokens if len(t) == 1)
    return (single_char_tokens / len(tokens)) > 0.4


# ------------------------------------------------------------------
# Semantic chunking (3 sentences, stride 1)
# ------------------------------------------------------------------
def chunk_sentences(sentences, window=3, stride=1):
    """
    Create overlapping semantic chunks and filter out
    corrupted chunks.
    """
    chunks = []
    chunk_id = 0

    for i in range(0, len(sentences) - window + 1, stride):
        chunk_text = " ".join(sentences[i:i + window])

        # Skip character-spaced / corrupted chunks
        if is_character_spaced(chunk_text):
            continue

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text
        })

        chunk_id += 1

    return chunks


# ------------------------------------------------------------------
# Save chunks to JSON
# ------------------------------------------------------------------
def save_chunks_to_json(chunks, output_file):
    """
    Save semantic chunks to a JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def main():
    input_file = "C:\\Users\\kappa\\OneDrive\\capstone\\dcass\\data\\reconstructed_output.txt"   # MUST be a clean .txt file
    output_file = "C:\\Users\\kappa\\OneDrive\\capstone\\dcass\\data\\chunks.json"

    # Load and clean text
    text = load_text_file(input_file)
    text = remove_references(text)

    # Sentence tokenization
    sentences = sentence_tokenize(text)
    print(f"Total sentences after cleaning: {len(sentences)}")

    # Semantic chunking
    chunks = chunk_sentences(sentences, window=3, stride=1)
    print(f"Total valid chunks: {len(chunks)}")

    # Save output
    save_chunks_to_json(chunks, output_file)

    print("✅ Phase 1 completed successfully (clean semantic chunks)")


if __name__ == "__main__":
    main()
