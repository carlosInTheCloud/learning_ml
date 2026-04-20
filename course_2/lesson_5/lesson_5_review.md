# Course 2 — Lesson 5 — Review questions

1. In Information Theory, what does Shannon Entropy mathematically quantify within a given dataset?

2. What is the calculated Shannon Entropy of a binary classification dataset where every single example belongs to the exact same class (e.g., 100% ‘Bonk’)?

3. When computing Information Gain, why must we take a weighted average of the child nodes’ entropies before subtracting from the parent node’s entropy?

4. How does the CART algorithm dynamically formulate splitting questions for a continuous variable, such as Cadence?

5. What is a characteristic of the ‘Greedy algorithm’ as used in decision tree induction?

6. What is the fundamental difference in tree architecture between the CART algorithm and the ID3 algorithm?

7. If an unconstrained Decision Tree is allowed to grow until every leaf node has an entropy of 0.0, what will almost certainly happen to the model?

8. In the context of Decision Tree pruning, what physically happens to a parent node during ‘Reduced Error Post-Pruning’ when its children are chopped off?

9. Which of the following is a ‘Pre-Pruning’ technique used to prevent a tree from overfitting during the training phase?

   - **A.** Applying the sigmoid function to the leaf outputs  
   - **B.** Removing leaves that do not improve validation accuracy  
   - **C.** Setting a maximum limit on the depth of the tree  
   - **D.** Multiplying the feature weights by a decay penalty (L2)

10. An ensemble method relies heavily on ‘Bootstrapping’ (the ‘B’ in Bagging). What does bootstrapping physically do to the training data?

11. In a Random Forest algorithm, what is the primary purpose of introducing ‘Feature Randomness’ (limiting the subset of features available at each split)?

12. Which of the following statements correctly differentiates Bagging from Boosting?

    - **A.** Bagging is strictly used for Decision Trees, while boosting is only used for neural networks.  
    - **B.** Bagging alters the weights of misclassified data points, while boosting randomly samples the data.  
    - **C.** Bagging trains independent models in parallel, while Boosting trains models sequentially to reduce bias.  
    - **D.** Bagging builds models sequentially, while Boosting builds them simultaneously in parallel.

13. When a Boosting algorithm like AdaBoost finishes training Tree 1, what specific mathematical adjustment does it make before training Tree 2?

14. For binary classification, how does a Random Forest physically output its final, aggregate prediction when given a new test example?

15. Unlike standard single Decision Trees, Gradient Boosting algorithms (like XGBoost) are particularly prone to which specific failure state if their sequential chain grows too long?

16. Why is the base-2 logarithm ($\log_2$) standardly used in the Shannon Entropy formula instead of the natural logarithm ($\ln$) or base-10 ($\log_{10}$)?

17. Assume a dataset has 100 rows. A Decision Tree considers a split that places 50 rows in the left bucket (Entropy = 0.4) and 50 rows in the right bucket (Entropy = 0.6). What is the Weighted Entropy of these children?

18. In a highly complex, 1,000-tree Random Forest, why is the resulting model considered a ‘Black Box’ compared to a single Decision Tree?

19. When defining the stopping criteria for tree induction, setting `min_samples_split = 20` ensures which of the following mechanical behaviors?

    - **A.** The information gain must mathematically exceed 20 bits.  
    - **B.** The tree is forced to grow until every tree has exactly 20 samples.  
    - **C.** The tree must process exactly 20 features before stopping.  
    - **D.** If a node contains 19 or fewer data points, the algorithm will instantly be converted into a leaf node, refusing to calculate any further splits.

20. If XGBoost is generally more accurate than a Random Forest, why might a Data Scientist intentionally choose to deploy a Random Forest instead in a production environment?
