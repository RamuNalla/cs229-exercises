def scale_features(data):
    mean = sum(data) / len(data)
    std = (sum((x - mean)**2 for x in data) / len(data))**0.5
    return [(x - mean) / std for x in data]