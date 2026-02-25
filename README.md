# 🎵 Messy Mashup — Robust Music Genre Classification [🔗](https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project/)

<p align="center">
  <b>Deep Learning Project | Audio Signal Processing | Music Information Retrieval</b>
</p>

---

## 🚀 Project Overview

**Messy Mashup** is a deep learning project focused on robust music genre classification under realistic and noisy mixing conditions.

Unlike traditional clean-track classification, this challenge requires models to generalize across:

- Cross-song stem recombination  
- Tempo synchronization adjustments  
- Instrument balance variations  
- Additive environmental and synthetic noise  

The objective is to accurately predict the genre of noisy mashup tracks using robust audio representation learning techniques.

---

## 🎯 Problem Statement

The training data consists of clean instrument-separated stems from 10 genres.  
The test set contains **noisy mashups** created by:

- Mixing stems from different songs of the same genre  
- Applying tempo adjustments for rhythmic alignment  
- Adding environmental noise samples  

The goal is:

> **Predict the correct genre label for each mashup track.**

---

## 📊 Evaluation Metric

Submissions are evaluated using **Macro F1 Score** across 10 genre classes.

- F1 score computed independently per genre  
- Averaged equally  
- Robust to class imbalance  

Higher Macro F1 indicates better overall classification performance.

---

# 📂 Dataset Structure
<code>messy_mashup/
├── ESC-50-master/    # Environmental noise dataset
├── genres_stems/     # Clean instrument-separated training data
├── mashups/          # Noisy test mashups (3020 files)
├── sample_submission.csv
└── test.csv
</code>

---

## 🎼 Training Data — `genres_stems/`

- 10 genres  
- 100 songs per genre  
- Each song split into 4 stems:
    1. drums.wav
    2. vocals.wav
    3. bass.wav
    4. other.wav


### 🎵 Genres Included

1. blues
2. classical
3. country
4. disco
5. hiphop
6. jazz
7. metal
8. pop
9. reggae
10. rock

These clean stems allow learning genre-specific musical structure at the instrument level.

---

## 🔊 Noise Dataset — `ESC-50-master/`

- 2,000 environmental sound clips  
- 50 labeled sound classes  
- Used for noise augmentation and robustness training  

---

## 🎚 Test Set — `mashups/`

- 3,020 unlabeled audio files  
- Each mashup is created by:

  - Selecting stems from different songs of the same genre  
  - Applying tempo synchronization  
  - Mixing stems into a single audio track  
  - Adding one or more random noise samples  

These mashups are intentionally noisy and musically varied, making genre classification significantly more challenging.

---

## 📤 Submission Format

Required CSV structure:

<code>id,genre
0001,blues
0002,classical
0003,country</code>

---

## 📦 Dataset Summary

- Total Dataset Size: ~25.88 GB
- Training Stem Files: 4000
- Test Mashups: 3020

---
