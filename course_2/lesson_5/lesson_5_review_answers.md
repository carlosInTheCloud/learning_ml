# Course 2 — Lesson 5 — Review (answers)

1. **Answer:** The amount of impurity, chaos, or uncertinty present. Entropy measures unpredictibility. A perfectly mixed dataset has high entropy. (maximum uncertinty), while a completely uniform dataset has zero entropy (no uncertinty).

2. **Answer:** Zero. There is zero suprises or uncertinty.

3. **Answer:** To account for the proportion of data that falls into each child bucket, preventing a tiny, pure bucket to falsly inflate the split value.

4. **Answer:** It finds the midpoint in the dataset, then evaluates each data point against it asking if the data is > or < than the midpoint. This "brute force" method ensures every possible mathematical boundary in the training data is evaluated for maximum gain.

5. **Answer:** It always makes the logical optimal choice at the current node without looking ahead to see if a worse split now might lead to a better split later. Greedy algorithm optimize for the immediate moment. It calculates the information gain for the current step and locks it in, completely blind of the rest of the tree.

6. **Answer:** CART strickly enforces binary splits (two branch per node), while ID3 allows for multi-way splits based on the number of categories. CART's nature produces deeper, more generalized trees, whereas ID3 can shatter data quickly by branching in many directions at once.

7. **Answer:** It will perfectly memorize the training data, but suffer from high variance, causing to fail on new unseen data. It is an overfitted tree. It has learned the noise of the dataset as opposed to learning the pattern.

8. **Answer:** It converts into a final leaf node, predicting the majority class of the training examples that originally fell into that bucket.

9. **Answer:** **C.** By capping the depth, the algorithm is forced to stop early before it can create hyper-specific, overfitted rules.

10. **Answer:** It creates a new dataset for each tree by randomly sampling rows from the original data with replacements. Sampling with replacement means some original rows will appear multiple times, and some will be left out entirely, creating a unique perspective for each tree.

11. **Answer:** To de-correlate the trees, preventing a highly dominant feature from being chosen as the root node by every single tree in the forest. If one feature is incredibily strong, every tree will use it first, making all trees virtually identical. Forcing it to ignore it, occasionally ensures the forest learns diverse rules.

12. **Answer:** **C.** Bagging uses the average of many independent models to smooth out overfitted noise. Boosting actively targets errors in a chain to improve accuracy.

13. **Answer:** It increases the weight (importance) of the specific data points that Tree 1 predicted incorrectly. By heavily weighing the mistakes of Tree 1, Tree 2 to is forced to dedicate its splits as solving the hardest, most misunderstood data points.

14. **Answer:** It passes the example through every tree and returns the majority vote (mode) across all the individual predictions. If 800 trees predict "bonk" and "200" predict "no bonk", the forest casts a majority vote and outputs "bonk".

15. **Answer:** Severe overfitting (high variance), because they will eventually memorize the noise of the training data while trying to correct microscopic errors. Boosting ruthelessly chases errors if left unchecked. It will build trees solely to predict extreme outliers, ruining generalization.

16. **Answer:** Because information theory originated in computer science and telecomunications, where information is measured in binary 'bits' 1s and 0s. Using based 2 allows the resulting entropy to be read literally as the number of bits required to encode the uncertainty.

17. **Answer:** ((0.4/2) + (0.6/2) = 1 - (0.2 + 0.3) = 0.5

18. **Answer:** Because it is impossible for a single human to to simultaneously read, trace, and interpret the routing logic of thousands of independent trees interacting over via majority vote. While a single tree is highly intepretable, an ensemble sacfrifices that interpretability for raw predictive accuracy.

19. **Answer:** **D.** This is a classic pre-prunning threshold that prevents the algorithm from drilling down into micro-clusters of data, avoiding overfitting.

20. **Answer:** Random forests are highly parallelizable (faster to train), and are famously resistent to overfitting with almost no hyperparameter tuning required. XGBoosts requires sequential training (slower) and meticulous tunnig to prevent catastrophic overfitting. Random forests are 'plug and play' and highly stable.
