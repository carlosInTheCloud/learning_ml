---
type: Guideline
title: Agent Teaching Method
description: Structure, file layout, and authoring rules the agent follows when developing curriculum content in this repository.
tags: [pedagogy, structure, conventions, authoring]
status: stable
generated: { by: human:carlos.espinosa, at: 2026-09-07T13:32:57Z }
trigger: always_on
---

# Agent Teaching Method

## 1. Program hierarchy

`overall_program.md` is a tree. Three roles are defined by position in that tree, not by nesting depth:

- **Part** — a top-level entry of the program (e.g. *Mathematical Foundations*, *Regression Methods*). Grouping only; carries no lesson content.
- **Topic** — a node whose children are leaves (e.g. *Linear Algebra*, *Linear Regression*, *Statistical Learning Theory*). A topic is the unit of orientation: it owns an `intro.md`.
- **Subtopic** — a **leaf** node (e.g. *Eigenvalues/eigenvectors*, *Ridge regression*, *VC dimension*). A subtopic is the unit of instruction: it owns the theory, exercises, solutions, and code.

Defining the subtopic as the leaf keeps the rule uniform across a ragged tree. Where a part's children are already leaves, that part is also the topic.

## 2. Directory layout

```
conftest.py                     # empty; puts the repo root on sys.path for tests
mlfs/                           # shared package — see section 5
{nn}_{part_name}/
    intro.md
    {nn}_{topic_name}/
        intro.md
        {nn}_{subtopic_name}/
            {nn}_{subtopic_name}.md
            {nn}_{subtopic_name}_exercises.md
            {nn}_{subtopic_name}_solutions.md
            ...supporting files
```

- Names are `snake_case`, ASCII, derived from the program entry.
- Order numbers are zero-padded two digits and restart within each parent, matching the order in `overall_program.md`.
- Content is developed in program order.

## 3. Required files

Every subtopic folder contains at least these three:

| File | Contents |
|---|---|
| `{nn}_{subtopic_name}.md` | The full theory: motivation, formal statement, derivations, worked examples, numerical-stability notes. |
| `{nn}_{subtopic_name}_exercises.md` | Exercises for the reader to solve. Contains no answers. |
| `{nn}_{subtopic_name}_solutions.md` | A complete solution to every exercise, in the same order and numbering. |

## 4. Supporting files

A subtopic may include any additional file that serves the goal in [goal.md](goal.md). These are expected rather than exceptional — the goal requires from-scratch implementations and numerical verification, and those belong in files that actually run.

- **`{subtopic_name}.py`** — the **stub**, and the reader's workspace. Function signatures, docstrings stating the shape of every argument and return, and `raise NotImplementedError` bodies. Ships unimplemented.
- **`{subtopic_name}_solution.py`** — the completed from-scratch NumPy implementation. The theory file quotes from *this* file rather than restating the code, so the two cannot drift apart.
- **`test_{subtopic_name}.py`** — pytest tests: numerical gradient checks against finite differences, agreement with closed-form or reference (`scikit-learn`, `scipy`) results on synthetic data, and dimensional/shape assertions. Tests import the **stub**, so they fail until the reader implements it. The suite is the assessment.
- **`figures/`** — plots, with the script that generates them. No committed image without its source.
- **Data files** — only small synthetic datasets, and only when a generator script will not do.

Rules for supporting files:

- Prefer `.py` modules with tests over notebooks. Notebooks diff poorly and hide execution order; use one only when the subtopic is genuinely about interactive exploration.
- All code must run under the repo's `requirements.txt`. Add a dependency there when a subtopic needs it, and say so.
- A subtopic is not complete until its tests pass **against the solution file**. The agent verifies this before shipping; what the reader receives is a suite that fails until they do the work.
- The stub and the solution share a signature exactly. A test written against one must run unchanged against the other.

## 5. The shared package (`mlfs/`)

A Python package at the repository root, importable from any test because the root `conftest.py` places the root on `sys.path`. No installation step, no `pyproject.toml`.

It holds **infrastructure, not lessons**:

- Numerical gradient checking and tolerance helpers.
- Synthetic dataset generators.
- Plotting helpers shared across figures.

Implementations may additionally be **promoted** into `mlfs/models/` once the subtopic that teaches them is complete, so that later subtopics build on them instead of reimplementing them — ridge regression in 4.2.2 should not rewrite the least squares of 4.1.1. Two rules govern promotion:

- **Nothing is promoted before it has been taught.** Promoting an implementation the reader has not yet built leaks future material and defeats the stub.
- **The subtopic's own solution file stays where it is.** It is the record of having derived and built the thing; the promoted copy is for reuse.

## 6. `intro.md` maintenance

- A **topic**'s `intro.md` states the topic's scope, its syllabus of subtopics, the reader's objective, and the prerequisites. As each subtopic is developed, the agent updates this file to link to it.
- A **part**'s `intro.md` is a brief index: one line per topic, linking to it.

## 7. Exercise composition

Exercises mix three kinds:

- **Derivations and proofs** — reproduce or extend a result from the theory.
- **Implementation** — write the algorithm, make the gradient check pass, match the reference.
- **Conceptual short-answer** — explain why a method behaves as it does, or where it breaks.

Weighting shifts across the program: **derivation-heavy** through the mathematical foundations, settling to roughly **half derivation, half implementation** — with conceptual questions present throughout — once the material becomes algorithmic.

## 8. Authoring discipline

- The agent generates content for a topic or subtopic **only when explicitly requested**. It does not run ahead of the reader.
- The agent does not skip a required file. Exercises without solutions, a stub without a solution, or an implementation without tests, is an incomplete subtopic.
