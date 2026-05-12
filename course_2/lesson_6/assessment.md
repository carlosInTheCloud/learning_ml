# Lesson 6 — Assessment

## Scenario 1: The architectural fork

You have been hired by a major credit card company to build a fraud detection model. You are given a massive, highly complex dataset containing **15 million historical transactions**. The fraud patterns are heavily intertwined with legitimate transactions, meaning they are **absolutely not linearly separable** in their raw 2D/3D state.

1. **Which specific SVM architectural path (Primal or Dual) must you choose and why?**
2. **Because of this path choice, what is the exact mathematical limitation you hit regarding the kernel trick?**
3. **As an architect, outline the specific, two-step production pipeline you will build to bypass this limitation and achieve a non-linear boundary on 15 million rows.**

---

## Scenario 2: The mathematics of sparsity

During a peer code review, a junior data scientist looks at your deployment script. They are confused because your original training dataset was an **800 MB CSV** file, but the serialized SVM model (`.pkl` file) you are pushing to the AWS production server is only **1.5 MB**.

Walk the junior developer through the **exact mathematical chain reaction** that caused this compression.

You must:

- Start your explanation at the **loss function** the algorithm uses.
- Connect it to the **$\alpha$ multipliers**.
- Finish by explaining what the **inference loop** actually does in production.

---

## Scenario 3: Variance and the box constraint

Your deployed SVM is exhibiting severe **high variance (overfitting)**. It has drawn a **razor-thin margin** and perfectly memorized a cluster of bad sensor readings from the training data.

1. **Which global hyperparameter must you adjust to fix this, and in which direction (higher or lower)?**
2. **Mathematically, explain the box constraint and how this hyperparameter physically strips the power away from those rogue outlier data points.**
3. **What is the geometric result of this change on the "street"?**
