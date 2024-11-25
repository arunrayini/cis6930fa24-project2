from src.feature_extraction import extract_features

def predict(model, vec, context, verbose=False):
    """
    Predicting the unredacted name from the context.
    """
    features = extract_features(context)
    X = vec.transform([features])
    prediction = model.predict(X)[0]
    if verbose:
        print(f"Extracted features: {features}")
        print(f"Predicted name: {prediction}")
    return prediction
