#!/usr/bin/env python3
"""
generate_classification_data.py

Creates a CSV of two Gaussian blobs for binary classification.
Columns: x1, x2, y
"""
import numpy as np
import argparse
import csv

def generate_blobs(n_samples=100, centers=[(-1, -1), (1, 1)], std=0.5, output='data.csv'):
    n_per = n_samples // len(centers)
    X_list, y_list = [], []
    for label, center in enumerate(centers):
        pts = np.random.randn(n_per, 2) * std + center
        X_list.append(pts)
        y_list.extend([label] * n_per)
    X = np.vstack(X_list)
    y = np.array(y_list)
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x1', 'x2', 'y'])
        for xi, yi in zip(X, y):
            writer.writerow([xi[0], xi[1], int(yi)])
    print(f"Saved {len(y)} samples to {output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n_samples', type=int, default=100)
    p.add_argument('--std', type=float, default=0.5)
    p.add_argument('--output', type=str, default='data.csv')
    args = p.parse_args()
    generate_blobs(args.n_samples, std=args.std, output=args.output)
