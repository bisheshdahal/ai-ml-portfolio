# ML Learning Journey

This repo is my day-by-day log as I work through an ML course. Each notebook is whatever I actually coded that day, and this README gets a new section added as I go — it's meant to read like a log, not a polished write-up after the fact.

---

## Day 1 — EDA Project: Getting Started with Exploratory Data Analysis

**File:** `Day1_EDA_project.ipynb`
**Dataset:** Seaborn's built-in `tips` dataset

First real EDA day. Started with the basics — shape, `info()`, checking for nulls and duplicates — then dropped duplicate rows. Looped through the columns to see what unique values each one held, which helped me actually understand what I was working with instead of just guessing.

Engineered my first feature: `tip_percentage = (tip / total_bill) * 100`. Then moved into visualizing — histograms and KDE plots for `total_bill` and `tip`, a boxplot to catch outliers, and countplots for `sex`, `smoker`, `day`, and `size`. Wrapped up with a scatterplot of `total_bill` vs `tip` and a first attempt at a regression plot.

**Takeaway:** the EDA workflow finally clicked — understand the data, clean it, engineer something useful, then visualize, first one variable at a time and then two together.

## Day 2 — Same Dataset, Again (On Purpose)

**File:** `Day2_EDA_project.ipynb`
**Dataset:** Seaborn's built-in `tips` dataset

Deliberately repeated the whole Day 1 process on the same data — shape, info, nulls, duplicates, unique values, feature creation, the same univariate plots, same `total_bill` vs `tip` scatterplot. Nothing new conceptually, just trying to make the workflow automatic before throwing messier data at it.

**Takeaway:** repetition day. Wanted the steps to become muscle memory before Day 3.

## Day 3 — FIFA 19 EDA

**File:** `Fifa19_EDA/` folder
**Dataset:** FIFA 19 player dataset (CSV)

First time with a real, messy dataset. Had to load it with `encoding="Latin-1"` since it wasn't plain UTF-8 — a small thing but a good reminder that real data doesn't always cooperate. Checked missing values, dtypes, and ran `describe()` for a summary.

From there: an age distribution histogram, a preferred-foot countplot, and an age-vs-overall-rating scatterplot, then a full correlation heatmap across the numeric columns. Went a bit deeper too — found the top 10 highest-rated players, grouped by club to get average rating per club, checked for wage outliers with a boxplot, and looked at position and nationality distributions with bar charts. Ended with a short written insights section pulling it all together.

**Takeaway:** first time doing groupby analysis and correlation heatmaps on something real, and actually writing up findings instead of just leaving plots to speak for themselves.

## Day 4 — Netflix EDA

**File:** `Netflix_EDA/` folder
**Dataset:** Netflix titles dataset (CSV)

Started by filtering rows on conditions to look up specific titles, then checked nulls, duplicates, and unique values. Made a copy of the dataframe before touching anything — learned the hard way (mentally, at least) why you don't clean the original.

Dropped nulls, stripped whitespace from `date_added`, then converted it to datetime using `pd.to_datetime(..., format="mixed")` and pulled out day, month, and year as their own columns. Finished with a countplot of content type, a countplot of rating, a look at how rating relates to type, and a pie chart of the top ratings by percentage.

**Takeaway:** datetime parsing and extraction, and the habit of working on a copy instead of the original dataframe.

## Day 5 — Feature Engineering: Feature Types

**File:** `Day5_FeatureEngineering.ipynb`

A more conceptual day — went through the different kinds of features and how each one needs to be handled differently:

- Cleaning inconsistent categorical text (`'M'`, `'male'`, `'Fem'`, `'female'` all becoming one consistent label)
- Numerical features — just identifying and inspecting them
- Categorical features — converting to `category` dtype
- Ordinal features — manually mapping ordered categories to numbers (Junior=1, Mid=2, Senior=3)
- Binary/boolean columns
- Date/time features — pulling out day, month, year, weekday name
- Text features — engineering word count and character count
- A first look at image data with `skimage`, just inspecting shape and dimensions

**Takeaway:** not every column should be treated the same way — the right prep depends on what type of feature it actually is.

## Day 6 — Feature Engineering: Missing Values & Encoding

**File:** `Day6_FeatureEngineering.ipynb`
**Dataset:** Seaborn's built-in `titanic` dataset

Learned the three flavors of missing data — MCAR, MAR, and MNAR — and what each implies about how you're allowed to handle it. Covered dropping (`dropna()` on rows and columns, and when that's actually fine) versus imputing:

- Mean imputation for roughly normal distributions
- Median imputation when outliers are around
- Mode imputation for categorical columns
- Group-based imputation (e.g. average age by sex)
- Forward/backward fill for sequential data
- Adding a missing-indicator column as its own feature

Picked up a rough rule of thumb: under 5% missing → drop or simple impute, 5–30% → statistical imputation, over 30% → drop the feature or use something more advanced. Also covered label encoding vs one-hot encoding and the tradeoff between them (label encoding is simple but can imply a false order).

**Takeaway:** how much data is missing and what kind of feature it is should drive the decision — there's no single "correct" way to fill in gaps.

## Day 7 — Feature Engineering: Target Encoding & Scaling

**File:** `Day7_FeatureEngineering_and_Scaling.ipynb`

Learned target/mean encoding — using `groupby` to get the average of a numeric column per category, then mapping that back onto the categorical column. Useful for high-cardinality categories where one-hot encoding would blow up into too many columns.

Then spent most of the day on scaling, since features on wildly different scales can quietly dominate a model:

- **StandardScaler** — Z-score, centers to mean 0 / std 1
- **MinMaxScaler** — squeezes values into a fixed range, usually 0 to 1
- **RobustScaler** — uses median and IQR instead of mean/std, so outliers don't skew it (tested this by comparing the mean of a normal list against one with an extreme outlier thrown in)
- **MaxAbsScaler** — divides by the max absolute value, keeps the sign, range -1 to 1
- **Log transform** (`np.log1p`) — compresses skewed data like income into something closer to normal
- **Power transform (Yeo-Johnson)** — a more general fix for non-normal data, works with negative values too

**Takeaway:** which scaler to reach for depends on the shape of the data — StandardScaler for roughly normal data, MinMaxScaler for a bounded range, RobustScaler when outliers are a problem, log/power transforms for skewed distributions.

## Day 8 — Time Series Feature Engineering

**File:** `Day8_TimeSeries_FeatureEngineering.ipynb`
**Dataset:** Sales data with dates (CSV)

First time treating time as a first-class feature. Converted the date column to datetime and pulled out year, month, day, day of week, week of year (`isocalendar().week`), and quarter. Added `is_weekend` as a simple binary flag off day of week.

Built lag features (`Lag_1`, `Lag_3`, `Lag_7`) with `.shift()` so the model could see recent history alongside the current row. Added a rolling mean to smooth out short-term noise, a difference feature (`.diff()`) to capture day-over-day change, and percentage change to capture relative rather than absolute movement. Took notes on a few things I hadn't coded yet — rolling min/max, rolling std, seasonal features, cyclical encoding, and trend features — as things to come back to.

**Takeaway:** lag, rolling, and difference features are the core toolkit for letting a model "see" trends instead of just a single point in time.

## Day 9 — Time Series Feature Engineering: Cyclical Encoding, Moving Averages & EMA

**File:** `Day9_TimeSeries_FE.ipynb`
**Dataset:** Sales data with dates (CSV)

Picked up where Day 8 left off. Set `date` as the actual index this time instead of just a column, then re-extracted year, month, quarter, day of week, day, day of year, and week of year straight from the index. Added the weekend flag again the same way.

The new concept was **cyclical encoding** — instead of leaving `month` as a plain number (where December and January end up looking numerically far apart even though they're right next to each other), I encoded it with sine and cosine (`month_sin`, `month_cos`) so the "wraparound" is preserved.

Rebuilt the lag, rolling mean, diff, and percent-change features from Day 8 on the indexed data, plotted a correlation heatmap, and dropped the NaN rows the shifting/rolling left behind. Then moved on to moving averages — computed 3-day, 7-day, and 30-day windows with `shift(1).rolling(n).mean()` — and finally exponential moving averages (EMA) with `.ewm(span=n).mean()` at spans of 7 and 30.

Plotted sales against MA7 and EMA7 side by side to see the difference directly: EMA reacts faster to recent changes because it weights recent points more heavily, while a plain moving average treats every point in the window equally.

**Takeaway:** cyclical encoding fixes the "December vs. January" problem for periodic features, and EMA is the better choice when you care more about recent movement than long-run smoothing.

---
## Day 10 — Handling Imbalanced Data

**File:** `Day10_Imbalanced_Data.ipynb`
**Dataset:** Synthetic (generated with numpy / `make_classification`)

Today was about what happens when one class massively outnumbers another — a 90/10 split instead of a nice even 50/50. Started by building an imbalanced dataset by hand: 900 rows for class 0, 100 for class 1, each drawn from its own normal distribution, then concatenated into one dataframe to see the imbalance show up in `value_counts()`.

From there, covered the two classic ways to fix it:

- **Upsampling** — duplicate rows from the minority class (using `resample()` with `replace=True`) until it matches the majority class in size
- **Downsampling** — randomly drop rows from the majority class (`resample()` with `replace=False`) until it matches the minority class

Both are quick fixes, but upsampling just duplicates existing points rather than creating anything new, which is where **SMOTE** (Synthetic Minority Over-sampling Technique) comes in — it generates new synthetic minority-class points through interpolation between existing ones, instead of copying the same rows over and over. Tried it out on a synthetic dataset from `make_classification` (90/10 split), plotted the classes before and after using `SMOTE().fit_resample()`, and could see the minority class visibly fill out in the scatterplot instead of just stacking duplicate points.

**Takeaway:** upsampling and downsampling are the fast, blunt tools for class imbalance, but SMOTE is the smarter fix when you want the model to see *new* plausible minority examples instead of the same ones repeated.

---
## Day 11 — Simple Linear Regression: From Notebook to Deployed App

**File:** `1Simple Linear Regression/` folder (`SLR.ipynb`, `main.py`, `app.py`)
**Dataset:** `weight-height.csv`

First time taking a model past the notebook and actually deploying it. Trained a simple linear regression on `weight-height.csv` — used `Weight` as the single feature to predict `Height`, since a scatterplot and correlation heatmap showed a strong linear relationship between the two. Split into train/test, scaled `Weight` with `StandardScaler`, fit a `LinearRegression`, and evaluated it with MSE, MAE, RMSE, R², and adjusted R². Checked the residuals with a histogram to make sure the errors were roughly centered around zero, then saved the trained model and scaler with `joblib`.

From there, wired it up to two small apps:

- `main.py` — a FastAPI backend that loads the saved model/scaler and exposes a `POST /predict` endpoint
- `app.py` — a Streamlit frontend that takes a weight input, calls the backend, and shows the predicted height

Ran into a real bug here too: `main.py` was loading `simple_linear_regression_model.pkl`, but the notebook had actually saved it as `linear_regression_model.pkl` — a filename mismatch that crashed the backend on startup with a `FileNotFoundError`. Fixed by matching the load path to the actual saved filename.

**Takeaway:** deploying a model is a different skill from training one — FastAPI serves the predictions, Streamlit gives it a UI, and the backend has to be running before the frontend, since the UI calls it directly over HTTP. Also learned to double-check that saved filenames and loaded filenames actually match — an easy mismatch to introduce and an easy one to miss.

## Day 12 — Multiple Linear Regression: Predicting an Economic Index

**File:** `Day12 MLR/` folder (`mlr.ipynb`, `main1.py`, `app1.py`)
**Dataset:** `economy_index.csv`

Same deployment pattern as Day 11, but with two input features instead of one. Loaded `economy_index.csv` and dropped the unneeded `Unnamed: 0`, `year`, and `month` columns, leaving `interest_rate` and `unemployment_rate` as inputs and `index_price` as the target. Used a pairplot and a correlation matrix to check relationships between all three variables, then regplots of `interest_rate` against both `index_price` and `unemployment_rate` to look at those relationships individually.

Split into train/test (75/25), scaled both input features with `StandardScaler`, and trained a `LinearRegression` model on the two-feature input. Evaluated the same way as Day 11 — MSE, MAE, RMSE, R², adjusted R² — and checked residuals with a distribution plot. Saved the model and scaler with `joblib` as `mlr.pkl` and `scaler.pkl`.

Deployed it the same way as the SLR project:

- `main1.py` — FastAPI backend taking `interest_rate` and `unemployment_rate` as named JSON fields, scaling them, and returning a predicted `index_price`
- `app1.py` — Streamlit frontend with two number inputs feeding into the same `/predict` call pattern

This time paid close attention to feature order — with multiple inputs, the backend has to build the input array in the exact same column order the model was trained on (`[interest_rate, unemployment_rate]`), or predictions come out wrong with no error to warn you. Since `main1.py` uses named Pydantic fields rather than a raw list, the order stayed correct without needing to think about it too hard.

**Takeaway:** going from one feature to multiple doesn't change the deployment pattern much, but it raises the stakes on keeping feature order consistent between training and inference — a mismatch there fails silently instead of throwing an error.

---
## Day 13 — Insurance Charges Prediction

**File:** `insurance-prediction.ipynb`, `app.py`
**Dataset:** `insurance.csv`

Went through a few iterations of the same model before landing on a final version — good practice in seeing how each change actually moved the R² score. First pass dropped `region` entirely and just label-encoded `sex` and `smoker`, which gave a baseline R². Then brought `region` back in with one-hot encoding (`pd.get_dummies`, dropping the first category to avoid the dummy trap), which improved things. Final version added two interaction features — `age_smoker` and `bmi_smoker` — since age and BMI matter a lot more for cost when someone's a smoker, and that combination gave the best R² of the three.

Also did a quick overfitting/underfitting sanity check by comparing R² on the training set versus the test set — close scores on both meant the model wasn't overfitting.

This time skipped saving the model with `joblib` and instead built the Streamlit app (`app.py`) to load `insurance.csv` and retrain the `LinearRegression` model fresh every time the app runs, using the same encoding and interaction-feature steps as the notebook. The app takes age, sex, BMI, children, smoker status, and region as inputs, builds a matching one-row dataframe, reorders its columns to match the training data exactly, and predicts charges.

**Takeaway:** one-hot encoding plus a couple of well-chosen interaction features can meaningfully beat a plain label-encoded baseline, and when a matching-column dataframe is built by hand for prediction, the column order has to line up exactly with training or the prediction is silently wrong.

---
## Day 14 — Logistic Regression: Titanic Survival

**File:** `logistic.ipynb`, `main.py`, `app.py`
**Dataset:** Seaborn's built-in `titanic` dataset

First classification model of the journey, after a run of regression projects. Loaded the Titanic dataset, did the usual EDA pass — shape, info, nulls, countplots of `survived` against `sex` and `class`, an age histogram — then trimmed down to the columns that actually mattered: `survived`, `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`. Filled missing `age` with the median and missing `embarked` with the mode, then label-encoded `sex` and `embarked` into numbers.

Trained a `LogisticRegression` on an 80/20 split with scaled features, and evaluated with accuracy, a confusion matrix, and a full classification report instead of just accuracy alone — since for survival prediction, false negatives and false positives matter differently. Saved the model and scaler with `joblib`.

Deployed it the same way as the earlier regression projects — `main.py` as a FastAPI backend returning both a survival prediction and a probability, `app.py` as a Streamlit frontend with sliders and dropdowns for passenger details.

**Takeaway:** classification needs a different evaluation lens than regression — accuracy alone can hide a lot, so confusion matrices and classification reports matter more here than they did for the regression projects.

## Day 15 — Hyperparameter Tuning: GridSearch & RandomizedSearch

**File:** `Hyperparameter.ipynb`
**Dataset:** `load_breast_cancer` (from scikit-learn)

A dedicated day to actually learn hyperparameter tuning properly instead of hand-picking values. Loaded the breast cancer dataset, split and scaled it, then tried two different search strategies:

- **`GridSearchCV`** on a `KNeighborsClassifier`, searching over a small grid of `n_neighbors` and `weights` values — exhaustively tries every combination in the grid, which works fine when the search space is small.
- **`RandomizedSearchCV`** on a `RandomForestClassifier`, sampling random combinations from distributions (`randint` for `n_estimators`, `min_samples_split`, `max_depth`) instead of trying every possible combination — necessary once the search space gets too large for a full grid search to be practical.

Pulled out `best_params_` and `best_estimator_` from each search and scored the tuned models on the test set.

**Takeaway:** grid search is exhaustive and fine for small search spaces, while randomized search samples instead of exhausting — the practical choice once there are too many hyperparameter combinations to try them all.

---
## Day 16 — K-Nearest Neighbors: The Core Idea

**File:** `knn.ipynb`
**Dataset:** Small worked-by-hand examples (no real dataset yet)

A conceptual day before jumping into actual KNN code — worked through the algorithm by hand on two toy examples to make sure the mechanics actually made sense first.

For **KNN classification**, used a tiny height/weight/gender table and a new person (168cm, 62kg): picked K=3, calculated distance to every training point, took the 3 nearest neighbors, and let them vote on the label — 2 said Male, 1 said Female, so the prediction was Male.

For **KNN regression**, did the same thing with a house size/price table and a new house at size 1250: found the 3 nearest houses by size (1000, 1200, 1300), and instead of voting, averaged their prices to get the prediction (58.33 lakhs).

Also noted the two distance metrics that matter for the "find nearest neighbors" step — Euclidean distance (straight-line, like a flight path) versus Manhattan distance (only horizontal/vertical movement, like a taxi navigating a grid of streets).

**Takeaway:** KNN classification and KNN regression use the exact same first three steps (choose K, measure distance, find the neighbors) — they only diverge at the last step, where classification votes and regression averages.

---
## Day 17 — Naive Bayes Classification

**File:** `naivebayes.ipynb`
**Dataset:** Seaborn's built-in `iris` dataset

First time using Naive Bayes, and the first classification project with proper cross-validation baked in. Loaded the Iris dataset, checked shape/info/describe and class balance with `value_counts()`, then visualized it with a pairplot colored by species and a correlation heatmap across the numeric features.

Split into train/test with `stratify=y` this time, so the class ratio stays the same in both sets instead of risking an uneven split by chance. Scaled the features, then trained a `GaussianNB` model and checked training vs. test accuracy, a full classification report, and a confusion matrix (plotted with `ConfusionMatrixDisplay` this time instead of a raw heatmap).

Went further than previous days on validation: ran 5-fold `StratifiedKFold` cross-validation to get an average accuracy across folds rather than trusting a single train/test split, then used `GridSearchCV` to tune `var_smoothing` — the one real hyperparameter Naive Bayes has — over a log-spaced range of values. Pulled out the best model from the grid search, saved it and the scaler with `joblib`, then reloaded both to confirm a fresh prediction still worked correctly.

**Takeaway:** stratified splitting and stratified cross-validation matter more on datasets where class balance is easy to accidentally break, and even a "simple" model like Naive Bayes has a hyperparameter worth tuning rather than leaving at its default.

---
## Day 18 — Decision Trees: Theory

**File:** `DT.ipynb`
**Dataset:** N/A — reference notes, no code this time

A pure theory day before writing any Decision Tree code. This notebook is a set of reference notes/slides rather than executable cells — covering the core ideas behind how a decision tree actually splits data (impurity, information gain, how the tree decides which feature to split on at each node) before jumping into building one.

**Takeaway:** wanted the "why" behind decision tree splitting settled first, so that Day 19's actual code wouldn't just be copy-pasting `DecisionTreeClassifier()` without understanding what it's doing under the hood.

## Day 19 — Decision Tree Classification: Adult Income Prediction

**File:** `adult.ipynb`, `main.py`
**Dataset:** UCI Adult / Census Income dataset (`adult.csv`)

First project predicting income bracket (`<=50K` vs `>50K`) instead of a continuous number or a simple binary outcome. Loaded the dataset with proper column names (the raw CSV has no header row), then found that missing values weren't stored as actual NaNs — they showed up as the string `' ?'`, so had to explicitly replace those with `np.nan` before `isnull()` would catch them. Filled the missing categorical values with the column mode.

Label-encoded every categorical column and stored each encoder in a dictionary keyed by column name, so the exact same encoding could be reapplied later on new data. Trained a `DecisionTreeClassifier`, checked accuracy/classification report/confusion matrix, then pulled out feature importances and plotted the top few levels of the tree itself with `plot_tree()` to see which features it was actually splitting on first.

Ran `GridSearchCV` over `criterion`, `max_depth`, `min_samples_split`, and `min_samples_leaf` to tune the tree instead of using default settings, and saved both the tuned model and the full encoder dictionary with `joblib`.

Deployed with a FastAPI backend (`main.py`) that takes a person's details as JSON, renames the fields to match the training data's hyphenated column names (`education_num` → `education-num`, etc.), runs each categorical field through its saved encoder — rejecting anything not seen during training — and returns the predicted income bracket.

**Takeaway:** real-world data doesn't always mark missing values as actual nulls — sometimes it's a placeholder string that needs to be caught explicitly first. Also, saving one encoder per column (not just one shared encoder) is what makes it possible to correctly decode each field back to its original categories at prediction time.

---
## Day 20 — Support Vector Machines & Comparing Multiple Models

**File:** `SupportVectorMachine.ipynb`, `best_model.pkl`
**Dataset:** Bank marketing dataset (`bank.csv`)

Biggest project structurally so far — instead of training one model, built a proper `ColumnTransformer` + `Pipeline` preprocessing setup (median imputation + scaling for numeric columns, mode imputation + one-hot encoding for categorical ones) and reused it across four different classifiers to compare them fairly: Logistic Regression, KNN, Decision Tree, and SVM.

Did some feature engineering before the split — `contacted_before` as a flag for whether `pdays` was -1 (never contacted), and `balance_per_age` as a simple ratio feature. Tuned Logistic Regression, KNN, and Decision Tree each with their own `GridSearchCV`, then compared all three on test accuracy in a results table. KNN came out on top and got saved as `best_model.pkl`.

Ran into a real bug in the SVM grid: `'classifier__kernel':['Linear','rbf']` — scikit-learn's kernel names are lowercase and case-sensitive, so `'Linear'` isn't valid and would throw immediately. It hadn't caused a crash yet only because the SVM grid search itself was commented out (`#Dont run`, likely because it's the slowest of the four on this dataset). Fixed the typo to `'linear'` and confirmed with a real `GridSearchCV` run that it no longer errors, so it's ready to include next time the SVM step actually gets run.

Also found that loading `best_model.pkl` fresh needs `pyarrow` installed alongside scikit-learn — it wasn't in `requirements.txt` and the load failed without it. Once installed, the model loads and predicts correctly; ran it on a new-customer example and got a clean prediction with a probability back.

**Takeaway:** wrapping preprocessing in a `Pipeline`/`ColumnTransformer` instead of doing it by hand made it trivial to plug four different classifiers into the exact same preprocessing and compare them on equal footing. Also learned that a saved model can carry a hidden dependency (like `pyarrow` here) that only shows up when loading it somewhere fresh — worth checking `requirements.txt` covers everything a `.pkl` actually needs, not just what trained it.

## Day 21 — Random Forest Regression: Car Selling Price Prediction

**File:** `cd.ipynb`
**Dataset:** CarDekho used car listings (`cardekho.csv`)

First regression project using an ensemble model instead of a linear one. Dropped identifier-style columns that wouldn't generalize (`Unnamed: 0`, `car_name`, `model` — too many unique values to be useful features), then cleaned the data: dropped nulls, removed duplicates, and trimmed the top and bottom 1% of `selling_price` as outliers using quantile clipping.

Explored the cleaned data with a handful of plots — a selling price distribution, average price by fuel type (electric came out highest, LPG lowest), selling price against vehicle age and against km driven, and a correlation heatmap across the numeric columns. `max_power` and `engine` size showed the strongest positive correlation with price, while `mileage` correlated negatively — makes sense, since higher-mileage-per-liter cars tend to be smaller/cheaper ones.

Label-encoded the categorical columns (`brand`, `seller_type`, `fuel_type`, `transmission_type`), then caught a leakage risk before training: had engineered a `price_per_km` feature, but since it's derived directly from `selling_price` (the target), it got dropped from the feature set before the split rather than left in.

Trained a `RandomForestRegressor` (100 trees) and evaluated with MAE, RMSE, and R² — landed at an R² of 0.924 on the first pass. Plotted actual vs. predicted prices to check how tight the fit was, and pulled feature importances, which confirmed `max_power` and `vehicle_age` were doing most of the work in the model's predictions. Ran `GridSearchCV` over `n_estimators`, `max_depth`, and `min_samples_split` to tune further, which nudged R² up to 0.926, then saved the tuned model with `joblib` and reloaded it to confirm a fresh prediction still worked.

**Takeaway:** feature importance plots are a good sanity check that a model is leaning on the features that actually make sense (power and age driving car price, not something spurious) — and any feature computed directly from the target needs to be dropped before training, not just excluded by accident.

---
## Day 22 — Model Comparison: Predicting Travel Package Purchases

**File:** `tourism1.ipynb`, `best_model.pkl`, `scaler.pkl`, `encoders.pkl`
**Dataset:** Travel package customer data (`Travel.csv`)

The biggest model comparison so far — seven classifiers (KNN, Logistic Regression, Naive Bayes, Decision Tree, SVM, Random Forest, AdaBoost) run head-to-head to predict whether a customer would take a travel package. Dropped `CustomerID`, imputed missing numeric values with the median and categorical ones with the mode, then explored the target balance, age distribution by outcome, monthly income vs. purchase, and a correlation heatmap across the numeric columns.

Label-encoded the categorical columns, did a stratified train/test split so the target ratio stayed consistent across both sets, and scaled the features for the models that needed it (KNN, Logistic Regression, SVM) while leaving tree-based models on the raw values. Ran all seven through the same accuracy comparison, then took the best of them (a tuned Random Forest via `GridSearchCV`) and evaluated it properly with a classification report and confusion matrix.

Caught a subtle bug while reviewing this one: an accidental second `train_test_split` call later in the notebook re-split the data **without** stratifying, using a different random_state pairing than before — which silently desynced the already-scaled training data from the new labels. KNN, Logistic Regression, and SVM were technically training on mismatched features/labels without erroring, just quietly producing untrustworthy accuracy numbers. Removed the duplicate split so everything downstream consistently uses the one stratified split.

Also fixed how categorical encoding got saved — the original version reused a single `LabelEncoder` object across every column, which meant only the last column's mapping was ever recoverable. Switched to saving one encoder per column in a dictionary (`encoders.pkl`), so a new customer's raw category values can actually be encoded correctly later instead of guessed. Finished the "predict on a new customer" cell, which had been left as an empty placeholder, with a real example row that gets encoded and passed to the saved Random Forest model.

**Takeaway:** a bug that doesn't throw an error is the most dangerous kind — the mismatched split here didn't crash anything, it just quietly corrupted results for three of the seven models. Also, when saving encoders for later reuse, one encoder per categorical column (not one shared object) is what actually makes it possible to encode new data consistently down the line.

---
## Day 23 — Clustering: KMeans, Hierarchical, and DBSCAN

**File:** `Clustering.ipynb`
**Dataset:** Synthetic blobs (`make_blobs`), Iris, synthetic moons (`make_moons`)

First unsupervised learning day — no target variable this time, just finding structure in the data on its own. Started with **KMeans** on synthetic blob data: scaled the features, ran the elbow method by plotting WCSS (within-cluster sum of squares) across k=2 to 10, and confirmed the elbow point visually and with `KneeLocator` from the `kneed` library instead of eyeballing it alone. Cross-checked with silhouette scores across the same range of k, then fit the final model and predicted on held-out test data.

Moved on to **hierarchical clustering** using the Iris dataset — scaled the features, reduced to 2 dimensions with PCA for visualization, and plotted a dendrogram with `scipy`'s `linkage` (Ward's method) to see how points merge into clusters at different distance thresholds. Used `AgglomerativeClustering` to cut the tree into 2 clusters based on that.

Finished with **DBSCAN** on moon-shaped synthetic data — a good example of why density-based clustering exists, since KMeans assumes roughly spherical clusters and would struggle on this crescent shape. DBSCAN groups points by density instead, so it handles the non-convex moon shapes correctly without needing to specify a cluster count upfront.

**Takeaway:** different clustering algorithms make different assumptions about cluster shape — KMeans assumes round/spherical clusters, hierarchical clustering builds a nested structure you can cut at any level, and DBSCAN doesn't assume a shape at all, just density, which is why it's the right tool for something like the moons dataset.

## Day 24 — KMeans in Practice: Mall Customer Segmentation

**File:** `mall.ipynb`
**Dataset:** `Mall_Customers.csv`

Applied the KMeans workflow from Day 23 to a real business use case — segmenting mall customers by `Annual Income` and `Spending Score` to find natural customer groups a marketing team could actually target. Checked for nulls (none), selected just those two features, and scaled them before clustering.

Ran the elbow method across k=1 to 10 and settled on k=5 as the natural cluster count. Fit the final KMeans model, added the cluster label back onto the original dataframe, and checked a silhouette score to quantify how well-separated the clusters actually were. Inverse-transformed the cluster centers back to the original income/spending scale so they'd be interpretable as real numbers instead of scaled ones, then plotted the customers colored by cluster with the centroids marked on top.

Finished by printing out per-cluster summaries — customer count, average income, and average spending score for each of the 5 groups — which is the part that actually turns "5 clusters" into something a business could act on (e.g. high income / low spending customers are a very different marketing target than low income / high spending ones).

**Takeaway:** clustering only becomes useful once you translate the cluster numbers back into something interpretable — inverse-transforming the centroids and summarizing each cluster's real-world averages is what makes the segmentation actually usable instead of just an abstract plot.

---

## Day 25 — NLP Fundamentals: Tokenization, Stemming, Lemmatization, POS Tagging

**File:** `Natural_language_procsessing.ipynb`
**Dataset:** Short custom text corpora and paragraphs (no external dataset)

First dedicated NLP day, laying out the roadmap before writing any of it: raw text → cleaning (tokenize, stem/lemmatize, remove stopwords) → classical vectorization (BoW, TF-IDF) → semantic embeddings (Word2Vec) → sequence models (RNN/LSTM/GRU) → Transformers/attention → pretrained models like BERT.

Started with **tokenization** at three levels — sentence (`sent_tokenize`), word (`word_tokenize`), and punctuation-aware (`wordpunct_tokenize`, which splits contractions like `"How're"` differently than the standard word tokenizer) — plus `TreebankWordTokenizer` as another word-level option with its own punctuation-handling rules.

Moved to **stemming**, comparing three stemmers on the same word list (`eating`, `programming`, `finalized`, etc.): `PorterStemmer` (classic, sometimes overly aggressive — e.g. reducing `"sportingly"` to a non-word stem), `RegexpStemmer` (rule-based, strips a custom suffix pattern like `ing$|s$|e$|able$`), and `SnowballStemmer` (an improved version of Porter that handles some edge cases like `"fairly"` better). Directly compared Porter vs. Snowball output side by side on the same words to see where they diverge.

Contrasted stemming with **lemmatization** using `WordNetLemmatizer`, which returns actual dictionary words instead of chopped-off stems — but only when given the correct part-of-speech tag (`pos='v'` for verbs, etc.), since the default POS assumption changes the output.

Built a full **preprocessing pipeline** on a paragraph: sentence tokenize → word tokenize → filter stopwords → stem (one pass) and separately word tokenize → filter stopwords → lemmatize (another pass), rejoining each sentence back into a string afterward. Noted `stopwords.words('nepali')` also works via NLTK's multilingual stopword lists.

Covered **POS tagging** with `nltk.pos_tag()` on a Nepal-themed paragraph, referencing the standard Penn Treebank tag set (NN, VB, JJ, RB, PRP, DT, etc.). Attempted **named entity recognition** with `nltk.ne_chunk()` next, but `tree.draw()` kept crashing the kernel, so that cell was left commented out rather than fixed — a known Tkinter/rendering issue with `ne_chunk`'s tree visualization, not a code logic problem. Wrote a POS-tag-to-WordNet-tag mapping helper function (`get_wordnet_pos`) in preparation for POS-aware lemmatization, though it wasn't wired into a full pipeline yet.

Closed with the **theory** behind the next vectorization step: Bag-of-Words (counting word frequency into a fixed-vocabulary vector) and TF-IDF (Term Frequency × Inverse Document Frequency), working through the IDF formula by hand with toy numbers (`log(1000/990)`, `log(100/50)`) to build intuition for why rare words get weighted higher than common ones before implementing it in code.

**Takeaway:** stemming and lemmatization solve the same problem differently — stemming is fast but crude (can produce non-words), lemmatization is slower but linguistically correct, and it only works right if you tell it the correct part of speech first. Also, BoW and TF-IDF are worth understanding by hand with toy numbers before trusting `sklearn`'s vectorizer to do it — knowing *why* rare words get weighted higher makes the vectorizer output much less of a black box later.

---
## Day 26 — Text Vectorization: SMS Spam Detection Prep

**File:** `vectorization.ipynb`
**Dataset:** SMS Spam Collection (`SMSSpamCollection`)

Applied Day 25's theory to a real dataset — loaded ~5,500 labeled SMS messages (ham vs. spam) and built a full text-cleaning pipeline: stripped out anything that wasn't a letter with regex, lowercased everything, tokenized, removed stopwords, and stemmed what was left, appending each cleaned message to a running `corpus` list. Along the way, caught and fixed a performance issue — the stopword set was originally being rebuilt from `stopwords.words('english')` on every single loop iteration, which is wasteful; built it once before the loop instead.

Vectorized the cleaned corpus two ways: `CountVectorizer` first as a simple binary bag-of-words (capped to the top 1000 features), then again with n-grams (2–3 word phrases instead of single words) to capture short phrases the single-word version would miss. Followed up with `TfidfVectorizer`, both on single words and on bigrams, to compare weighted term-importance vectors against the raw counts.

**Takeaway:** vectorizer choice and n-gram range both change what the resulting features actually capture — single-word counts miss phrases entirely, so bigram/trigram ranges are worth trying even though they blow up the vocabulary size fast (hence capping `max_features`).

---


## Day 27 — Deep Learning: Bank Customer Churn Prediction (ANN)

**File:** `churn.ipynb` (training), `testing.ipynb` (inference)
**Dataset:** `Churn_Modelling.csv`

First neural network project. Dropped ID columns (`RowNumber`, `CustomerId`, `Surname`), label-encoded `Gender`, one-hot encoded `Geography`, scaled features with `StandardScaler` — saved all three (`label_encoder_gender.pkl`, `onehot_encoder_geo.pkl`, `scaler.pkl`) right after fitting.

Built a `Sequential` ANN (64 → 32 ReLU → 1 sigmoid), compiled with Adam + binary crossentropy, trained with `EarlyStopping` on `val_loss`. Stopped around epoch 26 at ~86% val accuracy. Saved as `churn_model.h5`.

`testing.ipynb` loads the saved model + encoders + scaler, runs a new customer through the same preprocessing steps, and predicts churn probability.

**Takeaway:** save every artifact the pipeline touches (encoders, scaler, model) — reusing instead of refitting is what makes inference match training.

---

## Day 28 — Deep Learning: Estimated Salary Prediction (ANN Regression)

**File:** `regression.ipynb`
**Dataset:** `Churn_Modelling.csv`

Same bank dataset as Day 27, but flipped the task — predicting `EstimatedSalary` (regression) instead of `Exited` (classification), keeping `Exited` in as a feature this time. Same preprocessing: dropped IDs, label-encoded `Gender`, one-hot encoded `Geography`, added a correlation heatmap to check relationships before training.

Built the same 64 → 32 ReLU architecture, but a single linear output unit and MSE loss instead of sigmoid + binary crossentropy. Trained with `EarlyStopping` on `val_loss`; it ran the full 72 allowed epochs before stopping, landing at a final val MSE of ~3.36B (RMSE ≈ $58K). Saved as `regressionmodel.h5`.

**Takeaway:** salary turned out to be a much weaker regression target — a ~$58K RMSE on estimated salaries is a sign the encoded features here (geo, gender, balance, etc.) don't actually explain salary well, unlike churn which had a cleaner classification signal. Same architecture, same pipeline, very different result — a reminder that model choice matters less than whether the target is actually predictable from the given features.

## Day 28 — Introduction to PyTorch: Building Networks From Scratch

**File:** `pytorch.ipynb`
**Dataset:** Seaborn's built-in `tips` dataset (regression), `load_iris` from scikit-learn (classification)

First time using PyTorch after several days of Keras/TensorFlow — a different way of building the same kind of models, writing the network as an actual Python class instead of stacking layers in a `Sequential` call. Two mini-projects in one sitting, one regression and one classification, to see the pattern apply to both.

**Regression — `TipPredictor`:** predicted `tip` from `total_bill` and `size`. Split and scaled the data as usual, then converted everything to PyTorch tensors (`torch.FloatTensor`), reshaping the target with `.view(-1, 1)` to match what the network expects. Built `TipPredictor` as a class subclassing `nn.Module`, with an `nn.Sequential` stack going `2 → 32 → 16 → 8 → 1` with ReLU activations between each layer, and a `forward()` method defining how input flows through it. Trained manually with a `for` loop over 300 epochs — forward pass, compute `MSELoss`, zero the gradients, backpropagate, step the optimizer (`Adam`) — instead of a single `.fit()` call like Keras. Evaluated with MAE/MSE/RMSE/R² and tested it on a new customer (bill $25.50, party of 2).

**Classification — `IrisClassifier`:** same pattern, different shape and loss function. Network went `4 → 32 → 16 → 3` (4 input features, 3 output classes for the 3 iris species), used `CrossEntropyLoss` instead of `MSELoss`, and target labels went into `torch.LongTensor` instead of `FloatTensor` since classification targets are class indices, not continuous values. Trained the same manual loop, then evaluated with accuracy, a confusion matrix, and a classification report — and ran a prediction on a single new flower sample.

**Takeaway:** PyTorch makes you write out the training loop by hand — forward pass, loss, `zero_grad()`, `backward()`, `step()` — instead of hiding it behind `.fit()`, which is more code but makes it much clearer what's actually happening at each step of training. The regression-vs-classification difference mostly comes down to three things: the final layer's output size, the loss function (`MSELoss` vs `CrossEntropyLoss`), and the target tensor's dtype (`Float` vs `Long`).

---


*New day, new entry — this file gets a new section added as I go.*

