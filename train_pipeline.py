import csv
from ml_engine import BoltML


def fine_tune_model():
    pipeline = BoltML()

    features = []
    targets = []

    with open("osf_training_data.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            features.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
            targets.append([float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])])

    print(f"Loaded {len(features)} training frames from OSF.")

    pipeline.fine_tune_with_athlete_data(features, targets, epochs=20, prior_weight=0.6)

    print("Model successfully tuned on provided examples.")


if __name__ == "__main__":
    fine_tune_model()