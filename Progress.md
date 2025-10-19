Inital Training run

Metric	Training Value	Validation Value
Loss	0.2543	0.2575
Accuracy	92.78%	92.78%
AUC	0.6142	0.6251

Metrics Explanation

Here is a breakdown of your final training results (first train - 10/19/2025):

Metric     Training Value    Validation Value   Interpretation   

Loss       0.2543            0.2575             The Cost Function. 
This is a measure of error (how far the prediction is from the actual label). A lower value is better. The training and validation losses are very close, indicating the model is not overfitting—it performs almost as well on unseen validation data as it does on the data it trained on.

Accuracy   92.78%            92.78%             The Misleading Metric. 
This is the percentage of correct predictions (did the user click or not?). This metric is misleadingly high because your dataset is highly imbalanced (e.g., $9$ out of $10$ user-asset pairs are "no-click"). The model can achieve $92\%$ accuracy just by predicting "no-click" every time. Ignore this metric for recommender systems.

AUC        0.6142            0.6251             The Key Metric. 
Area Under the ROC Curve. This measures the model's ability to distinguish between a positive instance (a clicked asset) and a negative instance (a non-clicked asset), regardless of data imbalance. This is your real performance score.
* 0.5: Random guessing.
* 0.6251: Better than random. It indicates your model has learned some genuine signal and can successfully rank a positive item above a random negative item 62.51% of the time. This is a solid starting point!