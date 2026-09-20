#!/usr/bin/env bash
# ============================================================
# reorganize_repo.sh
#
# Reorganizes the ML-Course repo into a clean, professional
# folder structure using `git mv` (preserves file history).
# "Day#" labels are fully removed from all names.
#
# HOW TO USE:
#   1. Commit or stash any pending changes first (working tree
#      must be clean).
#   2. Copy this file into the ROOT of your repo
#      (~/Study_work Material/ML-Course-)
#   3. Run:  bash reorganize_repo.sh
#   4. Review with `git status`, then:
#        git add -A
#        git commit -m "Reorganize repo into professional structure"
#        git push origin main
#
# NOTE: This script does NOT touch __pycache__ or the tulip
# image dataset — those should already be handled by .gitignore.
# ============================================================

set -e  # stop on first error

echo "Creating new folder structure..."
mkdir -p data-preparation/eda/fifa19_eda
mkdir -p data-preparation/eda/netflix_eda
mkdir -p data-preparation/feature_engineering/timeseries_feature_engineering_1
mkdir -p data-preparation/feature_engineering/timeseries_feature_engineering_2
mkdir -p data-preparation/imbalanced_data
mkdir -p classical-ml/simple_linear_regression
mkdir -p classical-ml/multiple_linear_regression
mkdir -p classical-ml/knn
mkdir -p classical-ml/logistic_regression
mkdir -p classical-ml/decision_tree
mkdir -p classical-ml/naive_bayes
mkdir -p classical-ml/random_forest/plots
mkdir -p classical-ml/svm
mkdir -p classical-ml/kmeans
mkdir -p nlp/nlp_basics
mkdir -p nlp/nlp_vectorization
mkdir -p deep-learning/ann/churn_classification
mkdir -p deep-learning/ann/regression
mkdir -p deep-learning/ann/pytorch
mkdir -p deep-learning/cnn
mkdir -p deep-learning/rnn
mkdir -p deep-learning/lstm_gru
mkdir -p pytorch/fundamentals
mkdir -p pytorch/computer_vision
mkdir -p pytorch/custom_dataset
mkdir -p pytorch/models
mkdir -p projects/insurance_prediction
mkdir -p projects/tourism_prediction/plots

echo "Moving Deep Learning / ANN..."
git mv "Deep Learning/ANN/Churn + Testing/Churn_Modelling.csv" deep-learning/ann/churn_classification/churn_modelling.csv
git mv "Deep Learning/ANN/Churn + Testing/churn.ipynb" deep-learning/ann/churn_classification/churn.ipynb
git mv "Deep Learning/ANN/Churn + Testing/churn_model.h5" deep-learning/ann/churn_classification/churn_model.h5
git mv "Deep Learning/ANN/Churn + Testing/label_encoder_gender.pkl" deep-learning/ann/churn_classification/label_encoder_gender.pkl
git mv "Deep Learning/ANN/Churn + Testing/onehot_encoder_geo.pkl" deep-learning/ann/churn_classification/onehot_encoder_geo.pkl
git mv "Deep Learning/ANN/Churn + Testing/scaler.pkl" deep-learning/ann/churn_classification/scaler.pkl
git mv "Deep Learning/ANN/Churn + Testing/testing.ipynb" deep-learning/ann/churn_classification/testing.ipynb
git mv "Deep Learning/ANN/Pytorch/pytorch.ipynb" deep-learning/ann/pytorch/pytorch.ipynb
git mv "Deep Learning/ANN/Regression/Churn_Modelling.csv" deep-learning/ann/regression/churn_modelling.csv
git mv "Deep Learning/ANN/Regression/regression.ipynb" deep-learning/ann/regression/regression.ipynb
git mv "Deep Learning/ANN/Regression/regressionmodel.h5" deep-learning/ann/regression/regression_model.h5

echo "Moving Deep Learning / CNN..."
git mv "Deep Learning/CNN/sample_images.png" deep-learning/cnn/sample_images.png
git mv "Deep Learning/CNN/sign.ipynb" deep-learning/cnn/sign_language_classification.ipynb
git mv "Deep Learning/CNN/sign_mnist_test.csv" deep-learning/cnn/sign_mnist_test.csv
git mv "Deep Learning/CNN/sign_mnist_train.csv" deep-learning/cnn/sign_mnist_train.csv

echo "Moving Deep Learning / LSTM GRU..."
git mv "Deep Learning/LSTM GRU/gru.ipynb" deep-learning/lstm_gru/gru.ipynb
git mv "Deep Learning/LSTM GRU/lstm.ipynb" deep-learning/lstm_gru/lstm.ipynb
git mv "Deep Learning/LSTM GRU/lstm_training_history.png" deep-learning/lstm_gru/lstm_training_history.png
git mv "Deep Learning/LSTM GRU/review_length_distribution.png" deep-learning/lstm_gru/review_length_distribution.png

echo "Moving Deep Learning / RNN..."
git mv "Deep Learning/RNN/embedding.ipynb" deep-learning/rnn/embedding.ipynb
git mv "Deep Learning/RNN/prediction.ipynb" deep-learning/rnn/prediction.ipynb
git mv "Deep Learning/RNN/simplernn.ipynb" deep-learning/rnn/simple_rnn.ipynb
git mv "Deep Learning/RNN/simplernn_model.h5" deep-learning/rnn/simple_rnn_model.h5

echo "Moving EDA projects (removing Day# labels)..."
git mv "EDA_Projects/Day1_EDA_project.ipynb" data-preparation/eda/eda_project_1.ipynb
git mv "EDA_Projects/Day2_EDA_project.ipynb" data-preparation/eda/eda_project_2.ipynb
git mv "EDA_Projects/Fifa19_EDA/Day3_Fifa19_EDA.ipynb" data-preparation/eda/fifa19_eda/fifa19_eda.ipynb
git mv "EDA_Projects/Fifa19_EDA/Fifa19.csv" data-preparation/eda/fifa19_eda/fifa19.csv
git mv "EDA_Projects/Netflix_EDA/Day4_Netflix_EDA.ipynb" data-preparation/eda/netflix_eda/netflix_eda.ipynb
git mv "EDA_Projects/Netflix_EDA/netflix_data.csv" data-preparation/eda/netflix_eda/netflix_data.csv

echo "Moving Feature Engineering (removing Day# labels)..."
git mv "Feature Engineering/Day5_FeatureEngineering.ipynb" data-preparation/feature_engineering/feature_engineering_1.ipynb
git mv "Feature Engineering/Day6_FeatureEngineering.ipynb" data-preparation/feature_engineering/feature_engineering_2.ipynb
git mv "Feature Engineering/Day7_FeatureEngineering_and_Scaling.ipynb" data-preparation/feature_engineering/feature_engineering_scaling.ipynb
git mv "Feature Engineering/Day8 TimeSeries FeatureEngineering/Day8_TimeSeries_FeatureEngineering.ipynb" data-preparation/feature_engineering/timeseries_feature_engineering_1/timeseries_feature_engineering_1.ipynb
git mv "Feature Engineering/Day8 TimeSeries FeatureEngineering/Day8_sales.csv" data-preparation/feature_engineering/timeseries_feature_engineering_1/sales.csv
git mv "Feature Engineering/Day9 TimeSeries FeatureEngineering/Day9_TimeSeries_FE.ipynb" data-preparation/feature_engineering/timeseries_feature_engineering_2/timeseries_feature_engineering_2.ipynb
git mv "Feature Engineering/Day9 TimeSeries FeatureEngineering/Day9_data.csv" data-preparation/feature_engineering/timeseries_feature_engineering_2/timeseries_data.csv

echo "Moving Imbalanced Data (removing Day# label)..."
git mv "Imbalanced Data/Day10_Imbalanced_Data.ipynb" data-preparation/imbalanced_data/imbalanced_data.ipynb

echo "Moving Simple Linear Regression..."
git mv "Machine Learning/1Simple Linear Regression/SLR.ipynb" classical-ml/simple_linear_regression/simple_linear_regression.ipynb
git mv "Machine Learning/1Simple Linear Regression/app.py" classical-ml/simple_linear_regression/height_predictor_app.py
git mv "Machine Learning/1Simple Linear Regression/linear_regression_model.pkl" classical-ml/simple_linear_regression/linear_regression_model.pkl
git mv "Machine Learning/1Simple Linear Regression/main.py" classical-ml/simple_linear_regression/height_predictor_api.py
git mv "Machine Learning/1Simple Linear Regression/requirements.txt" classical-ml/simple_linear_regression/requirements.txt
git mv "Machine Learning/1Simple Linear Regression/scaler.pkl" classical-ml/simple_linear_regression/scaler.pkl
git mv "Machine Learning/1Simple Linear Regression/weight-height.csv" classical-ml/simple_linear_regression/weight_height.csv

echo "Moving KNN..."
git mv "Machine Learning/5KNN/knn.ipynb" classical-ml/knn/knn.ipynb

echo "Moving Multiple Linear Regression (removing Day# label)..."
git mv "Machine Learning/Day12 MLR/app1.py" classical-ml/multiple_linear_regression/index_price_app.py
git mv "Machine Learning/Day12 MLR/economy_index.csv" classical-ml/multiple_linear_regression/economy_index.csv
git mv "Machine Learning/Day12 MLR/main1.py" classical-ml/multiple_linear_regression/index_price_api.py
git mv "Machine Learning/Day12 MLR/mlr.ipynb" classical-ml/multiple_linear_regression/mlr.ipynb
git mv "Machine Learning/Day12 MLR/mlr.pkl" classical-ml/multiple_linear_regression/mlr_model.pkl
git mv "Machine Learning/Day12 MLR/scaler.pkl" classical-ml/multiple_linear_regression/scaler.pkl

echo "Moving Decision Tree..."
git mv "Machine Learning/Decision Tree/DT.ipynb" classical-ml/decision_tree/decision_tree.ipynb
git mv "Machine Learning/Decision Tree/adult.csv" classical-ml/decision_tree/adult.csv
git mv "Machine Learning/Decision Tree/adult.ipynb" classical-ml/decision_tree/adult_income_classification.ipynb
git mv "Machine Learning/Decision Tree/adult_label_encoders.pkl" classical-ml/decision_tree/adult_label_encoders.pkl
git mv "Machine Learning/Decision Tree/best_decision_tree_model.pkl" classical-ml/decision_tree/best_decision_tree_model.pkl
git mv "Machine Learning/Decision Tree/main.py" classical-ml/decision_tree/income_classifier_api.py

echo "Moving KMeans..."
git mv "Machine Learning/Kmeans/Clustering.ipynb" classical-ml/kmeans/clustering.ipynb
git mv "Machine Learning/Kmeans/Mall_Customers.csv" classical-ml/kmeans/mall_customers.csv
git mv "Machine Learning/Kmeans/mall.ipynb" classical-ml/kmeans/mall_customer_segmentation.ipynb

echo "Moving Logistic Regression..."
git mv "Machine Learning/Logistic Regression/Hyperparameter.ipynb" classical-ml/logistic_regression/hyperparameter_tuning.ipynb
git mv "Machine Learning/Logistic Regression/app.py" classical-ml/logistic_regression/titanic_app.py
git mv "Machine Learning/Logistic Regression/logistic.ipynb" classical-ml/logistic_regression/logistic_regression.ipynb
git mv "Machine Learning/Logistic Regression/main.py" classical-ml/logistic_regression/titanic_api.py
git mv "Machine Learning/Logistic Regression/model.pkl" classical-ml/logistic_regression/model.pkl
git mv "Machine Learning/Logistic Regression/scaler.pkl" classical-ml/logistic_regression/scaler.pkl

echo "Moving Naive Bayes..."
git mv "Machine Learning/Naive Bayes Model/naive_bayes_model.pkl" classical-ml/naive_bayes/naive_bayes_model.pkl
git mv "Machine Learning/Naive Bayes Model/naivebayes.ipynb" classical-ml/naive_bayes/naive_bayes.ipynb
git mv "Machine Learning/Naive Bayes Model/scaler.pkl" classical-ml/naive_bayes/scaler.pkl

echo "Moving Random Forest..."
git mv "Machine Learning/Random Forest/Plot_7_feature_importances.png" classical-ml/random_forest/plots/plot7_feature_importances.png
git mv "Machine Learning/Random Forest/cardekho.csv" classical-ml/random_forest/cardekho.csv
git mv "Machine Learning/Random Forest/cd.ipynb" classical-ml/random_forest/car_price_prediction.ipynb
git mv "Machine Learning/Random Forest/plot1_selling_price_distribution.png" classical-ml/random_forest/plots/plot1_selling_price_distribution.png
git mv "Machine Learning/Random Forest/plot2_avg_price_by_fuel_type.png" classical-ml/random_forest/plots/plot2_avg_price_by_fuel_type.png
git mv "Machine Learning/Random Forest/plot3_selling_price_vs_vehicle_age.png" classical-ml/random_forest/plots/plot3_selling_price_vs_vehicle_age.png
git mv "Machine Learning/Random Forest/plot4_selling_price_vs_km_driven.png" classical-ml/random_forest/plots/plot4_selling_price_vs_km_driven.png
git mv "Machine Learning/Random Forest/plot5_correlation_heatmap.png" classical-ml/random_forest/plots/plot5_correlation_heatmap.png
git mv "Machine Learning/Random Forest/plot_6_actual_vs_pred.png" classical-ml/random_forest/plots/plot6_actual_vs_pred.png

echo "Moving SVM..."
git mv "Machine Learning/SVM/SupportVectorMachine.ipynb" classical-ml/svm/support_vector_machine.ipynb
git mv "Machine Learning/SVM/bank.csv" classical-ml/svm/bank.csv
git mv "Machine Learning/SVM/best_model.pkl" classical-ml/svm/best_model.pkl

echo "Moving NLP..."
git mv "NLP/NLP Basics/Natural_language_procsessing.ipynb" nlp/nlp_basics/natural_language_processing.ipynb
git mv "NLP/NLP Vectorization/SMSSpamCollection" nlp/nlp_vectorization/sms_spam_collection
git mv "NLP/NLP Vectorization/vectorization.ipynb" nlp/nlp_vectorization/vectorization.ipynb

echo "Moving Projects (removing Day# label)..."
git mv "Projects/Day13 Insurance Prediction/app.py" projects/insurance_prediction/insurance_predictor_app.py
git mv "Projects/Day13 Insurance Prediction/insurance-prediction.ipynb" projects/insurance_prediction/insurance_prediction.ipynb
git mv "Projects/Day13 Insurance Prediction/insurance.csv" projects/insurance_prediction/insurance.csv

git mv "Projects/Tourism Project/Travel.csv" projects/tourism_prediction/travel.csv
git mv "Projects/Tourism Project/best_model.pkl" projects/tourism_prediction/best_model.pkl
git mv "Projects/Tourism Project/eda_1_target_distribution.png" projects/tourism_prediction/plots/eda_1_target_distribution.png
git mv "Projects/Tourism Project/eda_2_age_distribution.png" projects/tourism_prediction/plots/eda_2_age_distribution.png
git mv "Projects/Tourism Project/eda_3_monthly_income_boxplot.png" projects/tourism_prediction/plots/eda_3_monthly_income_boxplot.png
git mv "Projects/Tourism Project/eda_4_correlation_matrix.png" projects/tourism_prediction/plots/eda_4_correlation_matrix.png
git mv "Projects/Tourism Project/encoders.pkl" projects/tourism_prediction/encoders.pkl
git mv "Projects/Tourism Project/model_comparison.png" projects/tourism_prediction/plots/model_comparison.png
git mv "Projects/Tourism Project/scaler.pkl" projects/tourism_prediction/scaler.pkl
git mv "Projects/Tourism Project/tourism.ipynb" projects/tourism_prediction/tourism_prediction.ipynb
git mv "Projects/Tourism Project/tuned_rf_confusion_matrix.png" projects/tourism_prediction/plots/tuned_rf_confusion_matrix.png

echo "Moving Pytorch..."
git mv "Pytorch/Computer_Vision/computer_vision_pytorch.ipynb" pytorch/computer_vision/computer_vision_pytorch.ipynb
git mv "Pytorch/Computer_Vision/helper_functions.py" pytorch/computer_vision/helper_functions.py
git mv "Pytorch/Custom Dataset/Custom_datasets.ipynb" pytorch/custom_dataset/custom_datasets.ipynb
git mv "Pytorch/helper_functions.py" pytorch/helper_functions.py
git mv "Pytorch/models/01_pytorch_workflow_model_1.pth" pytorch/models/pytorch_workflow_model_1.pth
git mv "Pytorch/models/pytorch_2_model.pth" pytorch/models/pytorch_2_model.pth
git mv "Pytorch/neural_network_classification.ipynb" pytorch/fundamentals/neural_network_classification.ipynb
git mv "Pytorch/pytorch_1.ipynb" pytorch/fundamentals/pytorch_1.ipynb
git mv "Pytorch/pytorch_2.ipynb" pytorch/fundamentals/pytorch_2.ipynb
git mv "Pytorch/requirements.txt" pytorch/requirements.txt

echo "Removing now-empty old top-level folders (if any remain)..."
find "Deep Learning" "EDA_Projects" "Feature Engineering" "Imbalanced Data" "Machine Learning" "NLP" "Projects" "Pytorch" -type d -empty -delete 2>/dev/null || true

echo ""
echo "Done. Run 'git status' to review, then:"
echo "  git add -A"
echo "  git commit -m \"Reorganize repo into professional structure\""
echo "  git push origin main"
