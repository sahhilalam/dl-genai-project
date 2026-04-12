# 🎵 Messy Mashup — Robust Music Genre Classification  
[🔗 Kaggle Competition](https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project/)  

<p align="center">
  <b>Deep Learning Project | Audio Signal Processing | Music Information Retrieval</b>
</p>

---

## 🚀 Live Demo

👉 **Try the deployed model here:**  
🔗 https://huggingface.co/spaces/sahilalam75/dl-project-audio-classification  

Upload an audio clip and get instant genre predictions.

---

## 📌 Project Overview

**Messy Mashup** is a deep learning project focused on **robust music genre classification under noisy and complex real-world conditions**.

Unlike traditional clean-track classification, this project tackles:

- Cross-song stem recombination  
- Tempo synchronization distortions  
- Instrument-level mixing variations  
- Heavy environmental noise injection  

The goal is to build a model that **generalizes well despite extreme domain shifts**.

---

## 🎯 Problem Statement

- Training Data: Clean, instrument-separated stems (drums, vocals, bass, other)  
- Test Data: Noisy mashups created by:
  - Mixing stems from different songs  
  - Applying tempo alignment  
  - Adding environmental noise  

> 🎯 **Objective:** Predict the correct genre label for each mashup track.

---

## 📊 Evaluation Metric

- **Macro F1 Score**
  - Computed per class  
  - Averaged equally across 10 genres  
  - Robust to class imbalance  

---

## 📂 Dataset Structure

```
messy_mashup/
├── ESC-50-master/    # Environmental noise dataset
├── genres_stems/     # Clean training stems
├── mashups/          # Noisy test data (3020 files)
├── sample_submission.csv
└── test.csv
```

---

## 🎼 Training Data — `genres_stems/`

- 10 genres  
- 100 songs per genre  
- Each song split into 4 stems:
    1. drums.wav
    2. vocals.wav
    3. bass.wav
    4. other.wav

---

## 🎵 Genres

- blues  
- classical  
- country  
- disco  
- hiphop  
- jazz  
- metal  
- pop  
- reggae  
- rock  

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

## ⚙️ Preprocessing Pipeline

### Key Challenges
- Sampling rate mismatch (44.1kHz → 22.05kHz)
- Loudness differences (RMS shift)
- Silence in training vs continuous test audio
- Domain gap between clean stems and noisy mashups

### Final Pipeline

1. **Random Stem Mixing**
   - Combine drums, bass, vocals, other from different songs

2. **Downsampling**
   - Convert all audio → **16kHz**

3. **Gain Augmentation**
   - Random scaling per stem

4. **Cropping**
   - Train: Random 5s crop  
   - Validation: Center crop  

5. **Noise Injection**
   - 30% probability using ESC-50

6. **Normalization**
   ```
   y = y / (max(|y|) + 1e-9)
   ```

7. **Feature Extraction**
   - Log-Mel Spectrogram

---

## 🧠 Models Implemented

### 🔹 Transfer Learning Models
- ResNet18 (partial freeze)
- ResNet18 (fully unfrozen)
- **ResNet50 (best performing)**
- EfficientNet

### 🔹 Custom CNN (ScratchCNN)
- Multi-block Conv2D architecture  
- BatchNorm + ReLU + MaxPooling + Dropout  
- AdaptiveAvgPool2d for dimensionality reduction  

---

## 🏆 Best Model

| Model        | Macro F1 | Notes |
|-------------|--------|------|
| **ResNet50_1** | **0.9096** | Best performance |
| ResNet_2     | 0.9091 | Fully unfrozen |
| EfficientNet | 0.8764 | Fast & balanced |
| ScratchCNN   | 0.8555 | Lightweight |

---

## 🔍 Inference Strategy

- Split audio into **5-second windows**
- Predict each segment
- **Average probabilities** across segments

---

## 📈 Key Insights

- Treating spectrograms as images works extremely well  
- Transfer learning > training from scratch  
- Data preprocessing & augmentation were critical  
- Bridging the domain gap was the biggest challenge  

---

## ⚡ Tech Stack

- PyTorch (torch, torchvision, torchaudio)  
- NumPy  
- Pandas  
- Librosa  
- Scikit-learn  
- Matplotlib & Seaborn  
- Weights & Biases (wandb)  
- Huggingface & Gradio (deployment)   

---

## 📦 Requirements

See `requirements.txt` for exact dependency versions.

---

## 📜 Report

Full project report included in the repository.

---

## 🧠 Final Thought

This project shows that:

> *A strong data pipeline + transfer learning can outperform complex architectures trained from scratch.*
