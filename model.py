import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model():
  data = pd.read_csv('./data/Fish.csv')

  X = data.drop('Species', axis=1)  # Features
  y = data['Species']  # Target variable

  model = RandomForestClassifier()
  model.fit(X, y)

  joblib.dump(model, 'fish_classifier.pkl')


if __name__ == '__main__':
    train_model()