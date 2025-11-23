# SAMFEO-C: Constrained RNA Sequence Design

**A specialized version of the SAMFEO framework that supports RNA design under specific IUPAC nucleotide constraints.**

## 🧬 Overview

This repository contains **SAMFEO-C**, an extension of the SAMFEO (Structure-Aware Multifrontier Ensemble Optimization) algorithm. While the original SAMFEO optimizes sequences for a target structure freely, SAMFEO-C allows users to impose **IUPAC nucleotide constraints**.

This ensures that the designed sequences not only fold into the desired secondary structure but also strictly adhere to specific sequence requirements (e.g., locking specific bases, allowing only Purines at certain positions, etc.).

## 🚀 Key Features

* **IUPAC Support:** Fully supports standard IUPAC ambiguity codes (e.g., `N`, `R`, `Y`, `S`, `W`, etc.).
* **Constrained Mutation:** The evolutionary mutation operators have been modified to never violate the user-provided constraints during optimization.
* **Online Interactive Mode:** Designed for pipeline integration or interactive usage via standard input.

## 💻 Usage

Run the tool in online constrained mode using the following command:

```bash
python main.py --online -c --step 2500
```

### Input Format

The program expects input in **pairs of lines** via standard input (stdin):
1.  **Line 1:** The target secondary structure (dot-bracket notation).
2.  **Line 2:** The IUPAC constraint string (must be the same length as the structure).

**Example Interaction:**

```text
((...))     <-- You type the structure
NNNRAAA     <-- You type the constraints
```

*In this example, the first 3 bases can be anything (`N`), the 4th must be a Purine (`R`), and the last 3 must be Adenine (`A`).*

### IUPAC Code Reference

| Code | Bases | Meaning |
| :--- | :--- | :--- |
| **A, C, G, U** | Single | Exact Match |
| **N** | A, C, G, U | Any Nucleotide |
| **R** | A, G | Purine |
| **Y** | C, U | Pyrimidine |
| **S** | G, C | Strong Interaction |
| **W** | A, U | Weak Interaction |
| **K** | G, U | Keto |
| **M** | A, C | Amino |
| **B** | C, G, U | Not A |
| **D** | A, G, U | Not C |
| **H** | A, C, U | Not G |
| **V** | A, C, G | Not U |
| **. or -** | None | Gap |

## 📦 Requirements

* Python 3.x
* numpy
* pandas
* **ViennaRNA** (Python bindings must be installed and accessible)

## 🔗 References

If you use this code or the SAMFEO method, please cite the original paper:

**[1] Zhou, T., Dai, N., Li, S., Ward, M., Mathews, D.H. and Huang, L., 2023. RNA design via structure-aware multifrontier ensemble optimization. Bioinformatics, 39(Supplement_1), pp.i563-i571.**
