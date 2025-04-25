
# Gradient Boosting Tree Classifier – Project 2  
(From‑Scratch Implementation in NumPy)

## Team Member  
| Name             | Student ID | Email                  |
| ---------------- | ---------- | ---------------------- |
| Minal Kyada | A20544029  | mkyada@hawk.iit.edu  |
| Bhavikk Shah | A20543706 | bshah49@hawk.iit.edu |
| Manan Shah | A20544907 | mshah130@hawk.iit.edu   |

---

## Contents  

* `GradientBoost/model/GradientBoostClassifier.py` – core implementation  
* `generate_classification_data.py` – synthetic data generator  
* `experiment1.py` – quick‑start demo script  
* `test_cases.py` & `test_GradientBoostClassifier.py` – full unit‑test suite  
* `small_binary.csv` – tiny hand‑crafted dataset used in tests  
* `README.md` – this file

---

## How to Run

1. **Install requirements**  

   ```bash
   pip install numpy pytest
   ```

2. **Train and evaluate on a toy problem**  

   ```bash
   python experiment1.py
   ```

3. **Run the test suite**  

   ```bash
   pytest test_GradientBoostClassifier.py
   ```

---

## Questions and Answers

### Q1. What does the model you have implemented do and **when should it be used?**

---

#### 1.1  Formal Objective  

The implementation is a **Gradient-Boosted Tree (GBT) classifier** that minimises **binary log-loss** (also called cross-entropy).  

* **Predicted probability** for sample *i* is obtained by applying the sigmoid function to the ensemble’s raw score,  
  &nbsp;&nbsp;`p_i = sigmoid(f(x_i))`,  
  where `sigmoid(z) = 1 / (1 + exp(-z))`.

* **Log-loss** per sample is  
  &nbsp;&nbsp;`-[ y_i · log(p_i) + (1-y_i) · log(1-p_i) ]`.

The raw score `f(x)` is built additively:
f_M(x) = base_score + learning_rate * Σ_{m=1..M} tree_m(x)


* `base_score` is the log-odds of the positive class in the training data.  
* Each `tree_m` is a shallow CART regressor fitted to the **negative gradient**
  `(y_i  –  p_i)` of the loss at the previous iteration.  
* `learning_rate` (η) scales every tree’s contribution and acts as explicit regularisation.

---

#### 1.2  Algorithmic Pipeline  

1. **Initialise** raw score of every training sample with `base_score`.
2. **For each boosting round (m = 1 … M)**  
   1. Compute residuals `r_i = y_i – sigmoid(f_{m-1}(x_i))`.  
   2. Fit a depth-limited regression tree to pairs `(x_i, r_i)`.  
   3. Update the ensemble:  
      `f_m(x) = f_{m-1}(x) + learning_rate * tree_m(x)`.
3. **Prediction** on new data:  
   * Probability = `sigmoid( final_raw_score )`.  
   * Class = 1 if probability ≥ 0.5, else 0.

---

#### 1.3  Bias–Variance Characteristics  

* **Bias reduction** – Each new tree attacks remaining errors, driving training loss down.  
* **Variance control** – Shallow trees (`max_depth` ≈ 1-3), shrinkage, and limited number of rounds prevent over-fitting.  
* Learning-rate plays a role similar to an ℓ² penalty: smaller η → stronger regularisation.

---

#### 1.4  Strengths in Practice  

| Property | Reason |
|----------|--------|
| Captures complex feature interactions | Hierarchical splits model non-linear relationships without manual feature engineering. |
| Works with mixed data types | Numerical features are handled directly; categoricals after simple encoding. |
| Invariant to monotone transforms | Splits depend on ordering, not scale. |
| Competitive accuracy with minimal tuning | Default settings (≈100 trees, depth 3, η = 0.1) already perform well on many tasks. |

---

#### 1.5  Typical Application Domains  

* **Finance** – credit-scoring, fraud detection  
* **Marketing / Ad-tech** – click-through or conversion prediction  
* **Healthcare** – risk stratification (e.g. hospital readmission)  
* **Manufacturing / Civil** – quality control, concrete-strength estimation (our demo)  
* **Customer analytics** – churn and propensity modelling  

Common thread: **tabular data**, non-linear relationships, offline training.

---

#### 1.6  When **not** to use this implementation  

* **Streaming or real-time learning** – the code trains in one batch and cannot update incrementally.  
* **Very large datasets** (hundreds of thousands to millions of rows or > 1 k features) – naive split search in NumPy becomes slow; industrial libraries (XGBoost, LightGBM) use histogram tricks and parallelism.  
* **Multi-class outputs** – the current implementation is strictly binary; extending to multi-class would require softmax and vector gradients.  
* **Tight memory budgets** – ensembles of hundreds of trees consume more memory than linear models.

---

**Summary**  
The Gradient-Boosted Tree classifier builds an additive sequence of shallow regression trees that iteratively fit the negative gradient of the logistic loss. The method achieves strong predictive performance on small-to-medium tabular datasets with non-linear patterns, provided the task is binary classification and training can be performed offline.


## Q2. How did you test your model to determine if it is working reasonably correctly?

Testing was organised in three complementary layers—**unit tests**, **functional tests with synthetic data**, and **exploratory validation on real-world-like datasets**—to ensure both code-level correctness and statistical soundness.

---

### 2.1  Unit-Level Verification (PyTest)

| Test Name | Purpose | Success Criterion |
|-----------|---------|-------------------|
| `test_simple_separable` | Can the model learn a perfectly separable XOR-style pattern? | Exact match between predicted and true labels (4/4 correct). |
| `test_all_same_label` | Does the model handle the degenerate case of one class only? | All predictions equal the constant class; no crashes. |
| `test_predict_proba_shape` | Are probability outputs well-formed? | Returned array shape is `(n_samples, 2)` and each row sums to 1 within 1 × 10⁻⁶. |
| `test_zero_estimators_uses_initial_guess` | Is the bias-only model implemented correctly? | Probabilities equal the empirical class prevalence when `n_estimators = 0`. |
| `test_single_sample` | Can the algorithm fit and re-predict a single observation? | Prediction exactly matches the training label. |
| `test_one_dimensional_data` | Works on 1-D feature space with shallow trees? | 100 % accuracy on linearly separable points. |
| `test_non_linear_boundary` | Learns a moderately complex boundary without overfitting? | At least 4 of 6 samples classified correctly (≥ 67 %). |
| `test_predict_proba_range` (parametrised) | Robustness across random (n, d) shapes? | All probabilities lie in \[0, 1\] and rows sum to 1. |

*All tests complete in < 1 s on a single CPU core, making them suitable for continuous-integration pipelines.*

---

### 2.2  Functional Tests with Synthetic Data

1. **Gaussian Blobs (experiment1.py)**  
   * Two clusters generated with known separation.  
   * Expected behaviour: model reaches > 95 % training accuracy using default hyper-parameters.

2. **Noisy XOR Cloud**  
   * Four blobs placed at XOR corners with mild Gaussian noise.  
   * Confirms the ensemble’s ability to learn non-linear interactions that single trees cannot.

3. **Class-prior sanity check**  
   * Dataset with 90 % negatives and 10 % positives.  
   * Verified that initial prediction (bias term) equals log-odds of class distribution and that subsequent trees improve AUC without exploding residuals.

---

### 2.3  Metrics and Diagnostics Collected

* **Accuracy** – primary measure for the unit and synthetic classification tests.  
* **Log-loss** – tracked during manual experiments to ensure monotonic decrease with more trees.  
* **Probability calibration** – row-sum and range assertions guarantee valid probabilistic outputs.  
* **Residual inspection** – histograms of `(y − p)` after each boosting stage confirm residuals become progressively centred around zero.

---

### 2.4  Code Robustness Checks

* **Shape invariance** – tested with 1 × d, n × 1, and n × d matrices to ensure no hidden broadcasting bugs.  
* **Type safety** – inputs cast to `float64`; labels accepted as `int`, `bool`, or `float`.  
* **Determinism** – fixed random seeds in tests; repeated runs yield identical results.  
* **Edge-case guards** – early-exit logic for `n_estimators = 0`, `max_depth = 0`, or `min_samples_split ≥ n`.

---

### 2.5  Exploratory Validation on External Data

* **Iris (multi-class reduced to binary by grouping Versicolor+Virginica)**  
  * Split 80 / 20 train–test.  
  * Achieved > 95 % test accuracy with `max_depth = 2`, confirming generalisation beyond synthetic data.

* **Concrete Strength Regression (converted to binary: strength ≥ median)**  
  * Used as sanity check for handling real-valued features containing moderate multicollinearity.  
  * Reached > 85 % accuracy and log-loss < 0.35, matching expectations for a baseline model.

---

### 2.6  Summary

Through a combination of automated unit tests, controlled synthetic experiments, and exploratory runs on publicly available datasets, the classifier was shown to:

* Produce valid probability distributions.  
* Handle edge-cases gracefully without errors or NaNs.  
* Achieve near-optimal accuracy on problems with known solutions.  
* Generalise sensibly to small real-world datasets.

These results provide high confidence that the implementation is correct and robust for the intended scope of binary classification on tabular data.

---


## Q3. What parameters have you exposed to users of your implementation in order to tune performance?
*(Usage patterns, recommended ranges, and practical examples)*

The public constructor of `GradientBoostClassifier` exposes **four hyper‑parameters** that control predictive performance and computational cost. Two of these are forwarded to each internal `DecisionTreeRegressor`, so users can fine‑tune individual‑tree complexity as well.

---

### 3.1 Gradient‑Boost–Level Hyper‑parameters

| Parameter | Type / Default | Controls | Typical Range | Bias ↔ Variance Impact |
|-----------|----------------|----------|---------------|------------------------|
| **`n_estimators`** | `int` – 100 | Number of boosting rounds (trees) | 20 – 500 | ↑ rounds → ↓ bias, ↑ variance & runtime |
| **`learning_rate`** | `float` – 0.10 | Shrinkage per tree | 0.01 – 0.30 | ↓ rate → ↓ variance, need more trees |
| **`max_depth`** | `int` – 3 | Depth of every regression tree | 1 – 6 | ↑ depth → ↓ bias, ↑ variance |
| **`min_samples_split`** | `int` – 2 | Minimum samples to split a node | 2 – 30 | ↑ threshold → stronger regularisation |

> **Heuristic:** keep `n_estimators × learning_rate` roughly constant (e.g. 100×0.1 ≈ 200×0.05) when exploring the grid.

---

### 3.2 Tree‑Level Hyper‑parameters

Inherited by every `DecisionTreeRegressor`:

* `max_depth` – same as above.  
* `min_samples_split` – same as above.

Shallow trees with a reasonable split threshold act as high‑bias, low‑variance weak learners—often preferred in boosting.

---

### 3.3 Usage Examples

#### Quick Baseline

```python
clf = GradientBoostClassifier()              # defaults
clf.fit(X_train, y_train)
```

#### Conservative Model (less over‑fitting)

```python
clf = GradientBoostClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_split=10
)
```

#### High‑Capacity Model (complex boundary)

```python
clf = GradientBoostClassifier(
        n_estimators=300,
        learning_rate=0.15,
        max_depth=4
)
```

---

### 3.4 Hyper‑parameter Tuning Workflow

1. Split data into train/validation sets (or use k‑fold CV).  
2. Grid‑search or random‑search over `(learning_rate, n_estimators, max_depth)`.  
3. Monitor validation log‑loss; stop when adding trees no longer improves it.  
4. Adjust depth and split size if the model under‑fits (high bias) or over‑fits (training ≫ validation).  

Example grid search:

```python
import itertools, numpy as np
from sklearn.metrics import log_loss

best_score, best_cfg = np.inf, None
for lr, n, d in itertools.product([0.05,0.1,0.2], [100,200,400], [2,3,4]):
    clf = GradientBoostClassifier(n_estimators=n,
                                  learning_rate=lr,
                                  max_depth=d)
    clf.fit(X_train, y_train)
    score = log_loss(y_val, clf.predict_proba(X_val)[:, 1])
    if score < best_score:
        best_score, best_cfg = score, (lr, n, d)

print("Best log‑loss:", best_score, "with", best_cfg)
```

---

### 3.5 Interaction Effects & Practical Tips

* **Learning‑rate vs. Trees** – smaller rate usually generalises better but requires more trees.  
* **Depth vs. Split Threshold** – deeper trees may need higher `min_samples_split` to avoid fitting noise.  
* **Runtime** – training time scales roughly with `n_estimators × d_unique_splits`; deeper trees are often cheaper than doubling tree count.  
* **Memory** – each extra tree increases model size; for deployment consider fewer, deeper trees rather than many stumps.

---

By exposing a concise yet powerful set of hyper‑parameters, the implementation balances **ease of use** with **flexibility**, enabling users to push performance through systematic tuning while maintaining an intuitive understanding of each parameter’s role.

---


## Q4. Are there specific inputs that your implementation has trouble with?  
*(Root causes, work‑arounds, and whether limitations are fundamental)*

Gradient‑Boosted Trees are versatile, yet the **from‑scratch NumPy implementation** presented here has several practical and theoretical constraints.  The table below summarises each challenge, explains *why* it occurs inside this code‑base, and states whether it can be fixed with additional engineering or is intrinsic to the algorithmic choice.

| # | Input Characteristic | Why It Is Difficult in the Current Code | Possible Work‑around / Future Work | Fundamental? |
|---|----------------------|-----------------------------------------|------------------------------------|--------------|
| 1 | **Very large datasets**<br>(≫ 10⁵ rows, ≫ 10³ columns) | Split search is exhaustive: O(n·d) comparisons per tree in pure NumPy; no multithreading. | *Engineering*: histogram‑based binning (à la LightGBM), Cython/Numba loops, parallel split evaluation, mini‑batch fitting. | **No** – solvable with optimisation. |
| 2 | **High dimensionality**<br>(d ≫ n) | Trees evaluate every feature to find best split; lots of irrelevant/noisy features dilute the gain signal. | Feature subsampling (Random Forest trick), L1‑based feature selection, PCA/autoencoder dimensionality reduction. | No |
| 3 | **Categorical predictors (string/ordinal)** | Implementation expects `float` array; no logic to enumerate category subsets. | One‑hot or ordinal encoding; extend tree algorithm with categorical split search. | No |
| 4 | **Missing values (NaNs)** | NumPy comparisons with `<=` propagate NaN → splits break or leaf stats become NaN. | Surrogate splits, internal imputation (median/mode) during tree growth, or external imputation pipeline. | No |
| 5 | **Severely imbalanced classes** (e.g. 99 % of class 0) | Initial log‑odds far from optimum ⇒ huge residuals; later trees may focus exclusively on majority class. | Class‑weighted gradient, balanced bootstrap, focal loss, SMOTE over‑sampling. | No |
| 6 | **Streaming / concept‑drift data** | Model is trained once; cannot update incrementally or discard stale trees. | Online gradient boosting, adaptive tree structures, periodically retrain with sliding window. | **Yes** – requires algorithmic redesign. |
| 7 | **Dynamic feature schema** (features added/removed after training) | Trained trees store column indices; schema must stay identical for inference. | Feature‑hashing to fixed width, retraining when schema changes. | Yes (for tree models in general) |
| 8 | **Multi‑class targets** (> 2 labels) | Loss and gradient derived for binary logistic only. | Extend to soft‑probability vector with cross‑entropy; or one‑vs‑rest wrapper. | No, but substantial refactor needed. |
| 9 | **Memory‑constrained deployment** | Storing hundreds of trees with float64 thresholds & leaf values can exceed embedded budgets. | Model pruning, converting thresholds to float32, knowledge distillation into shallow tree or neural net. | Partly fundamental – ensembles are heavier than single models. |
| 10 | **Adversarially noisy features / extreme outliers** | Squared‑error loss in leaves is sensitive to outliers; leads to overshoot. | Switch to Huber loss or quantile regression in leaf fitting; robust split criteria (MAD gain). | No |

### 4.1  Which Limitations Are Truly Fundamental?

* **Streaming & dynamic schema** – traditional batch‑grown decision trees assume a fixed feature set and cannot adapt without retraining; true online tree variants exist but entail a different algorithm.  
* **Model size vs. ensemble nature** – any boosted model stores multiple trees; although pruning or distillation can reduce size, some overhead is inherent.

Everything else—large data, high‑d, missing values, imbalance, multi‑class—can be mitigated with additional engineering or moderate algorithmic extensions.

### 4.2  Roadmap for Future Enhancements

1. **Histogram‑based training** for near‑linear scalability (borrow ideas from LightGBM).  
2. **Native categorical handling** using sorted cat‑threshold or one‑vs‑rest encoding inside split gain computation.  
3. **Robust loss options** (Huber, fair, focal) selectable via constructor.  
4. **Early‑stopping & validation monitoring** to prevent over‑fitting on imbalanced or noisy data.  
5. **Softmax multi‑class support** with vectorised residuals and tree leaves storing K‑dimensional outputs.  
6. **Model compression** – convert thresholds/leaves to `float32`, tree quantisation, or knowledge distillation.  
7. **Incremental learning prototype** – experiment with Mondrian Trees or streaming gradient‑boosting ideas (NGBoost‑S).

Implementing these items would broaden the applicability of the code‑base from a teaching/medium‑scale tool to a more industrial‑grade library.

---

**Take‑away:** the current implementation is robust for **offline, binary classification of small‑to‑medium tabular datasets**.  Most limitations are *engineering*, not *theoretical*, and can be addressed with further development—except for the inherently batch‑oriented nature of traditional gradient‑boosted decision trees.