import re
import string

def normalize_text(text: str) -> str:
    """
    Normalize review text:
    - Lowercase
    - Remove punctuation
    - Replace URLs and numbers
    - Trim whitespace
    - Filter out very short reviews (<10 chars)
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Replace URLs
    text = re.sub(r'http\S+|www\S+', 'URL', text)

    # Replace numbers
    text = re.sub(r'\d+', 'NUM', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Trim whitespace
    text = text.strip()

    # Filter out very short reviews
    if len(text) < 10:
        return ""

    return text


if __name__ == "__main__":
    # Example usage
    sample_reviews = [
        "This product is AMAZING!!!",
        "Check out http://example.com for details.",
        "Only cost me 123 dollars.",
        "Bad.",
        "   Lots of spaces   "
    ]

    for review in sample_reviews:
        print(f"Original: {review}")
        print(f"Normalized: {normalize_text(review)}\n")