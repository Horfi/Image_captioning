# Image Caption Generator

A real-time web application that generates descriptive captions for uploaded images using a CNN–Transformer architecture.

![App Preview](https://github.com/user-attachments/assets/3a3be285-286e-416f-88fa-49812d2e7609)
![Captioning Demo](https://github.com/user-attachments/assets/92258e2f-1673-4006-8bad-5f640d67ced3)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Implementation Details](#implementation-details)
5. [Current Status & Future Work](#current-status--future-work)
6. [Tech Stack](#tech-stack)
7. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running Locally](#running-locally)
   - [Docker Deployment](#docker-deployment)
8. [Training Process](#training-process)
9. [Project Structure](#project-structure)
10. [License](#license)

---

## Project Overview

This application combines computer vision and natural language processing to deliver an end-to-end image captioning solution. Users can upload images through an intuitive interface and receive fluent, human-like descriptions in real time.

### Objectives
- Create a production-ready image captioning pipeline
- Provide an intuitive, responsive web UI
- Deliver low-latency inference via a FastAPI backend
- Support containerized deployment for easy scaling

---

## Key Features
- **Drag & Drop Interface:** Seamless image upload via drag-and-drop or file browser
- **Real-Time Processing:** Instant caption generation with live feedback
- **Responsive Design:** Works across desktop and mobile devices
- **RESTful API:** Dedicated endpoint (`/api/caption`) for service integration

---

## Architecture

The image captioning system uses a hybrid CNN-LSTM architecture with the following components:

![Architecture Diagram](architecture_diagram.png)

1. **Image Encoder:**
   - Uses InceptionV3 (pretrained on ImageNet) as the backbone CNN
   - Extracts high-level visual features (2048-dimensional vector)
   - Projects features to embedding space (128-dimensional)

2. **Caption Decoder:**
   - Embedding layer for word representation
   - LSTM layer (256 units) for sequence modeling
   - Concatenated image and text features for context-aware generation
   - Dense output layer for vocabulary prediction

3. **System Design:**
   - **Frontend:** React application for user interaction
   - **Backend:** FastAPI server for image processing and caption generation
   - **Model:** TensorFlow/Keras implementation of the neural network

This architecture, as shown in the diagram, effectively combines visual understanding with natural language generation.

---

## Implementation Details

Key aspects of the implementation include:

- **Vocabulary Management:** Dynamic tokenization with special tokens (`<start>`, `<end>`, `<pad>`, `<unk>`)
- **Training Pipeline:** Parallel data processing for efficient dataset creation
- **Mixed Precision:** Automatic mixed precision training for GPU acceleration
- **Loss Function:** Custom masked loss to properly handle variable-length sequences
- **Metrics:** Masked accuracy calculation to focus on non-padding tokens
- **Inference Pipeline:** Beam search for improved caption quality

---

## Current Status & Future Work

**Current Status:**
- Local deployment only
- ~48% training accuracy and ~38% validation accuracy on Flickr8k dataset
- Basic caption generation functionality

**Planned Improvements:**
- **Attention Mechanisms:** Implementing Bahdanau attention to improve focus on relevant image regions
- **Transformer Integration:** Replacing LSTM with Transformer decoder for better sequence modeling
- **Dataset Expansion:** Scaling to larger datasets (Flickr30k or MSCOCO)
- **Fine-Tuning:** Adjusting CNN encoder weights for domain adaptation
- **Beam Search:** Implementing better decoding strategies
- **Accuracy Target:** Aiming for at least 70% validation accuracy

The screenshots show the current web interface in action, demonstrating the upload functionality and caption generation capabilities.

---

## Tech Stack

| Component      | Technologies                               |
|----------------|-------------------------------------------|
| Frontend       | React, Fetch API, Tailwind CSS             |
| Backend        | Python, FastAPI, Uvicorn                   |
| ML Framework   | TensorFlow 2.x, Keras                      |
| Image Processing | Pillow, TensorFlow Image                 |
| Deployment     | Docker, Docker Compose                     |
| Optimization   | TensorFlow XLA, Mixed Precision Training   |

---

## Getting Started

### Prerequisites
- **Python** >= 3.8
- **Node.js** >= 14
- **Docker** & **Docker Compose** (optional)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/image-caption-app.git
cd image-caption-app
```

#### Backend Setup
```bash
cd backend
python3 -m venv venv
# Activate virtual environment
# macOS/Linux
source venv/bin/activate
# Windows PowerShell
.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

#### Frontend Setup
```bash
cd ../frontend
npm install
```

### Running Locally
```bash
# Backend
cd backend
uvicorn app:app --reload

# Frontend
cd ../frontend
npm start
```
Open your browser at `http://localhost:3000`

### Docker Deployment
```bash
cd docker
docker-compose up --build
```
Visit `http://localhost:3000` after services start.

---

## Training Process

Our current model was trained on a subset (5,000 images) of the Flickr8k dataset for 10 epochs with the following approach:

- **Data Preprocessing:**
  - Caption cleaning and normalization
  - Vocabulary filtering (removing words with <5 occurrences)
  - Parallel processing for efficiency

- **Training Strategy:**
  - Adam optimizer with learning rate scheduling
  - Early stopping to prevent overfitting
  - Model checkpointing to save best weights
  - Mixed precision training where supported

- **Performance Optimization:**
  - Parallelized data loading and preprocessing
  - TensorFlow XLA compilation when available
  - Optimized dataset pipeline with prefetching and caching

---

## Project Structure

```
image-caption-app/
├── backend/            # FastAPI server & model code
│   ├── app.py          # API endpoints
│   ├── models/         # Model definition and training scripts
│   │   ├── caption_model.py    # Core model implementation
│   │   └── train_model.py      # Training pipeline
│   └── utils/          # Preprocessing utilities
├── frontend/           # React application
│   ├── public/         # Static assets
│   └── src/            # Components & styles
├── data/               # Datasets (excluded from repo)
├── docker/             # Dockerfiles & compose configuration
└── README.md           # Project documentation
```

---

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
