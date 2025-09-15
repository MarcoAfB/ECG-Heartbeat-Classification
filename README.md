# ECG-Heartbeat-Classification

## Project Overview
This project develops and optimizes a Deep Learning model to classify heartbeats into five arrhythmia categories, using the MIT-BIH Arrhythmia dataset available on Kaggle (https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

The primary motivation is to create a robust and reliable classifier that can serve as a decision support tool in medical diagnostics. The automatic and accurate detection of arrhythmias is a fundamental step toward the early treatment of potentially serious cardiac conditions.

This work distinguishes itself from other public approaches by methodologically solving three critical and frequently encountered problems: the use of inadequate metrics, the choice of models that ignore the temporal nature of the data, and the negligence of class imbalance.

## The Problem with Common Approaches
An analysis of public notebooks on Kaggle for this same challenge revealed recurring methodological flaws. This project was built to specifically correct these points:

1. **Inadequate Metric (Accuracy):** Many projects use Accuracy as their primary metric. In a dataset with imbalanced classes like this one (where the "Normal" class is the majority), a model can achieve high accuracy simply by predicting the most common class, while completely failing to detect rare arrhythmias, which are clinically more important.

2. **Non-Temporal Models:** Models like Multilayer Perceptrons (MLP) or traditional algorithms treat each point of the ECG signal as a dependent feature, but not necessarily in a temporal way (they might consider that variables i and j are related, but not that the relationship is sequential), thus ignoring the sequence and temporal dependency that defines a heartbeat.

3. **Lack of Data Balancing:** Training a model on imbalanced data without proper handling leads to a bias toward the majority class, resulting in a classifier that is useless in practice for detecting anomalies.

## Our Methodological Approach
To build a robust solution, we adopted a rigorous and justified methodology at each step:

1. **Evaluation Metric:** F1-Score
Instead of Accuracy, we chose the F1-Score as our primary metric. The F1-Score is the harmonic mean of Precision and Recall, making it an ideal metric for imbalanced problems. A high F1-Score ensures that the model not only makes correct predictions but is also capable of finding the minority classes (rare arrhythmias), minimizing both false positives and false negatives.

2. **Model Architecture:** LSTM (Long Short-Term Memory)
Recognizing that an ECG signal is a time series, the natural choice was a Recurrent Neural Network (RNN). Specifically, we used an LSTM, which is designed to learn long-term dependencies in sequential data, capturing the complete morphology of each heartbeat much more effectively than static models.

3. **Handling Class Imbalance**
The most effective strategy identified was Over-sampling. We kept all records from the majority class and increased the minority classes through resampling until all classes had the same number of examples as the "Normal" class. This was done on the training set to avoid contaminating the sets used to validate the models.

4. **Hyperparameter Optimization:** Bayesian Search
To find the best model architecture, we used an advanced optimization approach with Ray Tune and Optuna. Bayesian Search is a smarter strategy than random or grid search because it uses the results from previous runs to inform which hyperparameter combinations to test next. This allowed us to explore the search space more efficiently, optimizing parameters such as:

- Number of LSTM layers

- Size of hidden layers

- Dropout rate

- Learning rate

- Batch size

- Use of Bidirectional layers

- Use of a dynamic learning rate schedule

5. **Feature Engineering:** Signal Analysis
During the Exploratory Data Analysis, it was observed that most of the discriminative information in the signals seemed to be concentrated in the first 140 features. This led to a key experiment: training the optimized model on a "truncated" dataset, which proved to be highly effective.

## Results and Conclusion
The combination of all these techniques resulted in a high-performance model. The final strategy, which combined the model with 140 features and balancing via over-sampling, was the clear winner.

F1-Score on the Test Set: 0.925

This result demonstrates an excellent balance in classifying all classes, including the minority ones.

Considerations on Computational Cost and Project Decisions
A crucial aspect of the project was managing computational cost. The ideal balancing approach (over-sampling) creates a very large dataset. Training a single model with this approach took almost 2 hours.

On the other hand, the hyperparameter optimization phase required training dozens of models. Performing this search on the ideal dataset would have been computationally infeasible.

Therefore, we made a pragmatic decision: for the search with Ray Tune, we used a faster balancing technique (under + over-sampling), which generated a smaller dataset. This allowed us to test 33 hyperparameter configurations in approximately 3 hours. After finding the best configuration, we trained it on the ideal dataset (over-sampling) to generate the final model. This strategy balanced the need for an exhaustive search with resource limitations.
