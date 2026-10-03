# AI / ML Portfolio

End-to-end machine learning and deep learning work in Python: data preparation, classical ML, NLP, neural networks in Keras and PyTorch, and small deployed apps with FastAPI and Streamlit. Each project is self-contained in its own folder with the notebook, the data or a download path, and any saved model artifacts.

---

## Highlights

| Project | Type | What it shows | Result | Location |
|---|---|---|---|---|
| **Customer Churn Prediction** | ANN classification | Keras network, early stopping, saved encoders and scaler, separate inference notebook | ~86% validation accuracy | [`churn-prediction/`](churn-prediction) |
| **Salary Prediction** | ANN regression | Same bank dataset reframed as regression (linear output, MSE loss) | RMSE ≈ $58K; salary is weakly predictable from these features | [`projects/salary_prediction/`](projects/salary_prediction) |
| **Travel Package Purchase** | Classification | Seven classifiers compared on identical splits, tuned Random Forest, per-column encoders | Tuned Random Forest selected | [`projects/tourism_prediction/`](projects/tourism_prediction) |
| **Insurance Charges** | Regression | One-hot encoding and interaction features (`age × smoker`, `bmi × smoker`), Streamlit app | Interaction features gave the best R² of three iterations | [`projects/insurance_prediction/`](projects/insurance_prediction) |
| **IMDB Sentiment Analysis** | NLP | SimpleRNN vs LSTM vs GRU, plus inference on raw text | GRU best at 85.4% test accuracy | [`deep-learning/`](deep-learning) |
| **Car Price Prediction** | Regression | Outlier clipping, leakage check, Random Forest with grid search | R² 0.926 | [`classical-ml/random_forest/`](classical-ml/random_forest) |
| **Adult Income Classification** | Classification | Decision Tree with grid search, per-column encoders, FastAPI endpoint | Tuned and deployed as an API | [`classical-ml/decision_tree/`](classical-ml/decision_tree) |
| **Bank Marketing Comparison** | Classification | `ColumnTransformer` + `Pipeline` reused across four models | KNN best, saved as `best_model.pkl` | [`classical-ml/svm/`](classical-ml/svm) |
| **Mall Customer Segmentation** | Clustering | KMeans with elbow and silhouette analysis, interpretable cluster profiles | 5 customer segments | [`classical-ml/kmeans/`](classical-ml/kmeans) |
| **PyTorch Fundamentals** | Deep learning | Tensors, training loops, multi-class classification, CNN on FashionMNIST | See [PyTorch](#pytorch) | [`pytorch/`](pytorch) |

---

## Repository Structure

```
ai-ml-portfolio/
├── churn-prediction/               # ANN classifier, encoders, scaler, inference notebook
├── projects/
│   ├── insurance_prediction/       # Regression + Streamlit app
│   ├── salary_prediction/          # ANN regression on the churn dataset
│   ├── tip_prediction/             # PyTorch regression and Iris classification
│   └── tourism_prediction/         # 7-model comparison, tuned Random Forest
├── classical-ml/
│   ├── simple_linear_regression/   # FastAPI + Streamlit deployment
│   ├── multiple_linear_regression/ # FastAPI + Streamlit deployment
│   ├── logistic_regression/        # Titanic survival, hyperparameter tuning, API + app
│   ├── knn/
│   ├── naive_bayes/
│   ├── decision_tree/              # Adult income, FastAPI endpoint
│   ├── random_forest/              # Car price prediction
│   ├── svm/                        # Bank marketing, model comparison
│   └── kmeans/                     # Clustering, mall customer segmentation
├── data-preparation/
│   ├── eda/                        # Tips, FIFA 19, Netflix
│   ├── feature_engineering/        # Encoding, scaling, time-series features
│   └── imbalanced_data/            # Resampling and SMOTE
├── deep-learning/
│   ├── rnn/                        # Embeddings, SimpleRNN, sentiment inference
│   └── lstm_gru/                   # LSTM and GRU on IMDB
├── nlp/
│   ├── nlp_basics/                 # Tokenization, stemming, lemmatization, POS tagging
│   └── nlp_vectorization/          # Bag-of-Words, TF-IDF, n-grams on SMS spam
├── pytorch/
│   ├── fundamentals/               # Tensors, workflow, classification
│   ├── computer_vision/            # FashionMNIST, CNN
│   └── models/                     # Saved state dicts
├── requirements.txt
└── README.md
```

---

## Deep Learning

### Bank customer ANNs: churn and salary

Both projects use the same bank dataset and the same preprocessing: drop ID columns, label-encode `Gender`, one-hot encode `Geography`, scale with `StandardScaler`. Every fitted object (both encoders, the scaler, the model) is saved, so inference applies exactly the transformations used in training.

| | Churn | Salary |
|---|---|---|
| Target | `Exited` (binary) | `EstimatedSalary` (continuous) |
| Architecture | Dense 64 → 32 → 1 | Dense 64 → 32 → 1 |
| Output / loss | Sigmoid / binary crossentropy | Linear / mean squared error |
| Training | Early stopping, halted at epoch 26 | Ran all 72 allowed epochs |
| Result | ~86% validation accuracy | Validation MSE ≈ 3.36B (RMSE ≈ $58K) |

The architecture is identical, but the outcomes are not. Churn has a clear signal in the features, while salary barely does, so the regression model cannot do much better however it is tuned.

### Text pipeline: embeddings to recurrent networks

Built in stages: tokenized and padded text, an embedding layer, three recurrent architectures on IMDB reviews, then inference on new text.

- **Embeddings** (`deep-learning/rnn/embedding.ipynb`): integer-encoded and padded sample sentences, then passed them through `Embedding(10000, 10)` to inspect the resulting `(batch, 8, 10)` tensors.
- **SimpleRNN** (`simple_rnn.ipynb`): `Embedding(10000, 128)` → `SimpleRNN(128)` → sigmoid, with reviews padded to 500 tokens. Validation accuracy swung between 61% and 82% across epochs while training accuracy climbed to 93%.
- **Inference** (`prediction.ipynb`): reloads the saved model, encodes raw text with the IMDB word index, and returns a label with a confidence score.
- **LSTM and GRU** (`deep-learning/lstm_gru/`): `Embedding(10000, 32)` with 64 recurrent units, reviews padded to 200 tokens, 5 epochs, batch size 128.

| Model | Parameters | Train acc | Val acc | Test acc |
|---|---|---|---|---|
| SimpleRNN (128 units, length 500, 10 epochs) | 1,313,025 | 93.4% | 78.5% | not evaluated |
| LSTM (64 units, dropout 0.2 and recurrent dropout 0.2) | ~345K | 90.5% | 82.6% | 83.0% |
| GRU (64 units, dropout 0.3 after the layer) | 338,881 | 95.9% | 86.7% | 85.4% |

This is not a perfectly controlled comparison: the SimpleRNN used a longer sequence length, wider layers, and more epochs, and the LSTM had recurrent dropout the GRU lacked. The direction is still clear. The gated models were far more stable than the plain RNN with a quarter of the parameters, and the GRU matched or beat the LSTM while training faster. Both still overfit, so more regularization would be the next step.

---

## PyTorch

Notebooks that follow Daniel Bourke's [*Learn PyTorch for Deep Learning*](https://www.learnpytorch.io) course, covering the fundamentals through computer vision. `helper_functions.py` comes from the course repository.

| Notebook | Topic | Key points |
|---|---|---|
| `fundamentals/pytorch_1.ipynb` | Tensors | Creation, dtypes, shape errors and transposes, matrix multiplication, aggregation, reshaping and stacking, indexing, NumPy interop, reproducibility, GPU transfer |
| `fundamentals/pytorch_2.ipynb` | Training workflow | Synthetic linear data with known parameters (weight 0.7, bias 0.3), a custom `nn.Module`, hand-written training and test loops, saving and loading a `state_dict`, then the same model with `nn.Linear` |
| `fundamentals/neural_network_classification.ipynb` | Classification | Multi-class model on a 4-class blob dataset, logits → softmax → argmax, accuracy, precision, recall, F1, confusion matrix |
| `computer_vision/computer_vision_pytorch.ipynb` | Computer vision | FashionMNIST, `DataLoader` batching, baseline model, timed experiments, TinyVGG-style CNN, stepping through `nn.Conv2d` shapes |

Results from the notebooks:

- **Regression workflow:** the model recovered the true parameters closely. The hand-built model learned weight 0.699 and bias 0.309, and the `nn.Linear` version learned 0.697 and 0.303.
- **Multi-class classification:** 99.5% test accuracy on the blob dataset.
- **FashionMNIST baseline** (Flatten + two linear layers, 3 epochs, 64 s on CPU): 83.4% test accuracy, 0.477 test loss.

Saved state dicts live in `pytorch/models/`. A separate PyTorch project in [`projects/tip_prediction/`](projects/tip_prediction) applies the same patterns to real data: a tip regressor (R² 0.30) and an Iris classifier (96.7% test accuracy).

---

## Classical Machine Learning

| Area | Notebooks | Covered |
|---|---|---|
| Linear models | `simple_linear_regression`, `multiple_linear_regression` | Train, evaluate (MSE, MAE, RMSE, R², adjusted R², residual analysis), serve with FastAPI, front with Streamlit |
| Logistic regression | `logistic_regression` | Titanic survival, confusion matrix and classification report, `GridSearchCV` and `RandomizedSearchCV` |
| KNN | `knn` | Worked examples of classification and regression, Euclidean vs Manhattan distance |
| Naive Bayes | `naive_bayes` | Stratified split, 5-fold stratified cross-validation, `var_smoothing` tuning |
| Decision trees | `decision_tree` | Impurity and information gain, Adult income model, feature importances, `plot_tree`, API with saved per-column encoders |
| Random forest | `random_forest` | Used-car pricing, quantile clipping, feature importances, grid search (R² 0.924 → 0.926) |
| SVM and model comparison | `svm` | Preprocessing pipeline shared by Logistic Regression, KNN, Decision Tree, and SVM |
| Clustering | `kmeans` | KMeans (elbow, silhouette), hierarchical clustering with a dendrogram, DBSCAN, customer segmentation |

## Data Preparation and NLP

- **EDA:** the `tips` dataset, FIFA 19 players, and Netflix titles. Univariate and bivariate analysis, groupby summaries, correlation heatmaps, datetime parsing, outlier checks.
- **Feature engineering:** feature types, missing-value strategies (MCAR, MAR, MNAR and imputation choices), label, one-hot, and target encoding, six scalers and transforms, and time-series features (lags, rolling windows, differences, cyclical encoding, moving averages and EMA).
- **Imbalanced data:** upsampling, downsampling, and SMOTE on synthetic 90/10 data.
- **NLP basics:** sentence and word tokenization, stemming vs lemmatization, stopword removal, POS tagging.
- **Text vectorization:** cleaning pipeline on the SMS Spam Collection, Bag-of-Words and TF-IDF with unigrams and n-grams.

---

## Tech Stack

- **Language:** Python
- **Data and visualization:** pandas, NumPy, Matplotlib, Seaborn
- **Machine learning:** scikit-learn, imbalanced-learn, kneed
- **Deep learning:** TensorFlow / Keras, PyTorch, torchvision, torchmetrics
- **NLP:** NLTK
- **Serving:** FastAPI, Streamlit, joblib
- **Tooling:** Jupyter, Git

---

## Getting Started

```bash
git clone https://github.com/bisheshdahal/ai-ml-portfolio.git
cd ai-ml-portfolio

python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash); .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
pip install -r pytorch/requirements.txt   # only needed for the PyTorch notebooks
```

Notebooks read data and models by relative path, so **run each notebook from inside its own project folder**.

Projects with a backend and a UI need the API running first. For example, from `classical-ml/simple_linear_regression/`:

```bash
uvicorn height_predictor_api:app --reload   # terminal 1
streamlit run height_predictor_app.py       # terminal 2
```

The same pattern applies to `multiple_linear_regression`, `logistic_regression`, and `decision_tree`. The insurance app (`projects/insurance_prediction/`) is self-contained and trains on startup:

```bash
streamlit run insurance_predictor_app.py
```

A few saved models need extra libraries to load, for example `pyarrow` for the bank marketing `best_model.pkl`.

---

## Engineering Practices Applied

- **Reproducible inference:** every encoder, scaler, and model used in training is persisted, and inference reuses them rather than refitting.
- **One encoder per categorical column,** so each field can be encoded and decoded independently.
- **Feature order is enforced** at prediction time to match the training columns exactly.
- **Leakage checks:** features derived from the target are removed before training.
- **Stratified splits and cross-validation** wherever class balance matters.
- **Pipelines over manual preprocessing** (`ColumnTransformer` + `Pipeline`) so several models can be compared on identical preprocessing.
- **Honest evaluation:** classification reports and confusion matrices rather than accuracy alone, and weak results (salary regression, tip regression) are reported as they are.
