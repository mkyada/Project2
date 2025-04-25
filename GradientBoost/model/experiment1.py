import numpy as np
from generate_classification_data import generate_blobs
from GradientBoost.model.GradientBoostClassifier import GradientBoostClassifier

def main():
    # generate data
    output = 'demo_data.csv'
    generate_blobs(n_samples=200, std=0.7, output=output)
    data = np.loadtxt(output, delimiter=',', skiprows=1)
    X, y = data[:, :2], data[:, 2].astype(int)

    clf = GradientBoostClassifier(
        n_estimators=50,
        learning_rate=0.2,
        max_depth=2
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = (y_pred == y).mean()
    print(f"Demo accuracy (train): {acc:.3f}")

if __name__ == '__main__':
    main()
