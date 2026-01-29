# MM-Spectrum: Modality-Aware Mixture-of-Experts for Multimodal Spectroscopy-to-Structure Translation


MM-Spectrum is a research codebase for **molecular structure prediction and conditional generation** from **multiple spectroscopic modalities** (e.g., ¹H NMR, ¹³C NMR, IR, MS).  
This repository is built on top of the widely used benchmark **Multimodal Spectroscopic Dataset** and its official baselines, and extends them with **modality-aware Mixture-of-Experts (MoE)** design, improved training dynamics, and reproducible evaluation under **full-modality** and **partial-modality** (e.g., dual-modality) settings.

---

## Table of Contents

- [1. Highlights](#1-highlights)
- [2. Benchmark Lineage](#2-benchmark-lineage)
- [3. Method Overview (MM-Spectrum)](#3-method-overview-mm-spectrum)
- [4. Repository Structure](#4-repository-structure)
- [5. Dataset](#5-dataset)
- [6. Environment Setup](#6-environment-setup)
- [7. Data Preparation & Preprocessing](#7-data-preparation--preprocessing)
- [8. Training](#8-training)
- [9. Inference & Decoding](#9-inference--decoding)
- [10. Evaluation Protocols](#10-evaluation-protocols)
- [11. Reproducibility](#11-reproducibility)
- [12. Notes on `onmt_local/`](#12-notes-on-onmt_local)
- [13. Citation](#13-citation)
- [14. License & Acknowledgements](#14-license--acknowledgements)

---

## 1. Highlights

**MM-Spectrum** targets realistic multimodal spectroscopy modeling challenges:

- **Multimodal spectra → structure** translation (SMILES / canonical forms).
- **Full-modality** training + **partial-modality inference** (dual-modality subsets, missing modalities).
- **Modality-aware MoE routing** to mitigate *modality imbalance* and improve expert specialization.
- **Interpretable training diagnostics**: routing utilization, layer×expert occupancy, balance metrics, aux-loss curves.
- **Benchmark-compatible** scripts and data formats for reproducibility.

---

## 2. Benchmark Lineage

This project is an extension of the benchmark codebase:

- **Benchmark Repo (Official)**: `rxn4chemistry/multimodal-spectroscopic-dataset`  
  It provides:
  - a standardized dataset release + download scripts,
  - data generation scripts (`benchmark/generate_input.py`),
  - baseline training scripts (including transformer baselines built with OpenNMT),
  - evaluation conventions for spectroscopy-to-structure and structure-to-spectra tasks.

MM-Spectrum keeps the benchmark’s **dataset format and preprocessing interfaces**, but replaces/extends the transformer backbone with **MoE-enhanced modules** (implemented in `onmt_local/`).

---

## 3. Method Overview (MM-Spectrum)

### 3.1 Problem Setting

Given one or more spectral modalities (e.g., NMR/IR/MS), the model predicts the target molecular representation (e.g., SMILES).  
We evaluate:

- **Full-modality**: all modalities present at inference.
- **Partial-modality**: only subsets are available (e.g., NMR+IR, IR+MS, etc.).
- **Cross-source / heterogeneous modality distributions**: different sources can yield different modality quality, resolution, sparsity, and coverage.

### 3.2 Key Idea: Modality-Aware MoE for Spectral Tokens

MM-Spectrum introduces a **modality-aware MoE feed-forward block** into the transformer:

- **Routing conditioned on modality identity** (explicit modality embeddings / biases).
- **Specialized experts** for different modality statistics and token distributions.
- Optional regularizers and logging hooks to analyze:
  - token-to-expert assignment distribution,
  - per-layer expert occupancy,
  - load-balancing, entropy, coefficient of variation,
  - auxiliary losses and stability over steps.

> **Implementation note**: All MoE-related core changes are in `onmt_local/`, which is uploaded separately and treated as the “method code”.

---

## 4. Repository Structure

A recommended layout (adapt to your actual repository tree):


