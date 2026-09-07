---
type: Guideline
title: Postponed Topics
description: Subjects deliberately excluded from the current program, with the reasoning, and the seed of a follow-up program.
tags: [scope, deferred, roadmap]
status: living
generated: { by: human:carlos.espinosa, at: 2026-09-07T13:32:57Z }
trigger: always_on
---

# Postponed Topics

A running list of subjects deliberately kept out of [overall_program.md](overall_program.md). These are **decisions, not oversights**, and this file is the seed of a follow-up program to be assembled once the current one is complete.

This is distinct from the *extension* marking inside the program. An extension is a deferrable leaf that still lives in the curriculum; an entry here is out of the curriculum entirely, for now.

**This file grows.** When a subject surfaces during the work and is judged out of scope, it is appended here with the same reasoning — what it is, why it was deferred, what it depends on, and when to revisit. The follow-up program is built from whatever has accumulated by the time the current program is finished.

---

## Reinforcement Learning

**What it is.** Learning from interaction with an environment through delayed rewards rather than from a labeled dataset: Markov decision processes, Bellman equations, value and policy iteration, temporal-difference learning, Q-learning, policy gradients, actor-critic.

**Why postponed.** It is a different learning paradigm, not a technique within supervised learning — the agent's own actions change the distribution it subsequently sees, which breaks the i.i.d. assumption underlying most of parts 2 through 8. Covering it properly means roughly 20–25 subtopics at this program's depth, a part comparable in size to Unsupervised Learning.

**Why postponing is safe.** Nothing in parts 1–11 depends on it. It is the most cleanly separable subject in machine learning, and the current program is close to ideal preparation for it: MDPs need probability and expectation (part 1.3), value function approximation needs gradient descent and neural networks (parts 1.4 and 9), and policy gradients rest on the log-derivative trick, which follows directly from maximum likelihood estimation (part 1.3.6). Finishing this program leaves no gaps to fill before starting it.

**What is given up in the meantime.** Robotics and control, game-playing agents, sequential recommendation, and the RL stage of LLM alignment. That last gap is narrower than it appears: part 9.5 covers pretraining and fine-tuning, and DPO — which has largely displaced PPO for preference tuning — is supervised-shaped rather than RL-shaped.

**When to revisit.** After part 9, when there is direct experience to judge the appetite against, rather than a guess made in advance.

---

## Multi-Armed Bandits

**What it is.** The shallow end of reinforcement learning: exploration versus exploitation, regret bounds, upper confidence bound, Thompson sampling.

**Why postponed.** Deferred alongside reinforcement learning to keep the decision whole. On its own it is small — four or five subtopics — and needs only probability and some concentration inequalities.

**Why it is worth its own entry.** It is the most immediately practical slice of the subject (A/B testing, exploration in recommender systems) and the natural on-ramp to full reinforcement learning. If only one of the two is ever taken up, this is the one with the better ratio of usefulness to cost.

**When to revisit.** Either as the first topic of a reinforcement learning follow-up, or on its own if a practical need for it arrives sooner.
