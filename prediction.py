import joblib

def predict(data):
    clf = joblob.load("rf_model.sav")
    return clf.predict(data)