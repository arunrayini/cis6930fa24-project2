import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def extract_features(context):
    """
    Extracting features from the redacted context.
    """
    features = {}
    words = word_tokenize(context)
    redacted_index = context.find("█")

    if redacted_index == -1:
        features['previous_word'] = ""
        features['next_word'] = ""
        features['length_redacted'] = 0
        features['context'] = context
    else:
        token_index = next((i for i, token in enumerate(words) if "█" in token), -1)
        features['previous_word'] = words[token_index - 1] if token_index > 0 else ""
        features['next_word'] = words[token_index + 1] if token_index < len(words) - 1 else ""
        features['length_redacted'] = context.count("█")
        features['context'] = context.replace("█", "")
    return features
