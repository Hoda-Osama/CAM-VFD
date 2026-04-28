# CAM-VFD: Cross-Attention Multimodal Video Forgery Detection

Official implementation of the paper:

**"CAM-VFD: Cross-Attention Multimodal Video Forgery Detection"**
Hoda Osama Elkhodary et al.

---

## 📌 Overview

CAM-VFD is a multimodal deepfake detection framework that models **cross-modal contradiction** between:

* Appearance (CLIP)
* Motion (VideoMAE)
* Depth (MiDaS)

Unlike traditional fusion methods, CAM-VFD uses **directional cross-attention**, where appearance features query motion and depth to detect inconsistencies.

---

## 🧠 Key Features

* Cross-modal contradiction modeling
* Cross-attention fusion (appearance → motion/depth)
* Robust to:

  * compression
  * noise
  * blur
  * adversarial attacks (FGSM, PGD)
* Supports large-scale benchmarks:

  * GenVidBench
  * GenVideo

---

## 🏗️ Architecture

![Framework](assets/architecture.png)

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/CAM-VFD.git
cd CAM-VFD

pip install -r requirements.txt
```

---

## 📂 Dataset

We evaluate on:

* GenVidBench
* GenVideo

⚠️ Datasets are not included due to licensing restrictions.

Please follow official dataset instructions:

* [GenVidBench link]
* [GenVideo link]

---

## 🚀 Training

```bash
python training/train.py --config configs/default.yaml
```

---

## 📊 Evaluation

```bash
python evaluation/test.py --config configs/genvideo.yaml
```

---

## 🛡️ Robustness Testing

```bash
python evaluation/robustness.py
python evaluation/adversarial.py
```

---

## 📈 Results

| Dataset     | Accuracy | F1-score | AUROC  |
| ----------- | -------- | -------- | ------ |
| GenVidBench | 95.31%   | -        | -      |
| GenVideo    | 93.43%   | 90.63%   | 96.56% |

---

## 🔬 Citation

If you use this work, please cite:



---

## ⚠️ Disclaimer

This project is intended for research purposes only.
Do not use for unethical surveillance or privacy violations.

---

## 📬 Contact

Hoda Osama Elkhodary
Email: [hudaoelkhodary@gmail.com](mailto:hudaoelkhodary@gmail.com)
