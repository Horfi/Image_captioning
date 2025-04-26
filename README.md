# Image Caption Generator Web App

A real-time web application that generates descriptive captions for uploaded images using a state-of-the-art CNN–Transformer model.

![App Preview](https://github.com/user-attachments/assets/3a3be285-286e-416f-88fa-49812d2e7609)
![Captioning Demo](https://github.com/user-attachments/assets/92258e2f-1673-4006-8bad-5f640d67ced3)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Running Locally](#running-locally)
   - [Docker Deployment](#docker-deployment)
6. [Training the Model](#training-the-model)
7. [Project Structure](#project-structure)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

---

## Project Overview

This application combines computer vision and natural language processing to deliver an end-to-end image captioning solution. Users upload an image and receive a fluent, human-like description in real time.

### Objectives
- Showcase a production-ready image captioning pipeline
- Provide an intuitive, responsive web UI
- Offer low-latency inference via a FastAPI backend
- Support containerized deployment with Docker

---

## Key Features
- **Drag & Drop Upload:** Easy image selection via drag-and-drop or file browser
- **Real-Time Captioning:** Instant caption generation on the client
- **Responsive UI:** Live image preview and caption display
- **RESTful API:** FastAPI endpoint (`/api/caption`) for seamless integration

---

## Architecture

![Architecture Diagram](architecture_diagram.png)

1. **Frontend (React)**
   - File upload and preview component
   - Fetch API calls to the backend
   - Dynamic caption rendering

2. **Backend (FastAPI)**
   - `/api/caption` endpoint for image uploads and caption responses
   - Model loaded at startup for edge latency

3. **Model (TensorFlow + Keras)**
   - **Encoder:** InceptionV3 CNN for feature extraction
   - **Decoder:** Transformer-based sequence generator

---

## Tech Stack

| Layer          | Technology                     |
| -------------- | ------------------------------ |
| Frontend       | React, Fetch API, CSS          |
| Backend        | Python, FastAPI, Uvicorn       |
| ML Framework   | TensorFlow, Keras              |
| Image Processing | Pillow                       |
| Deployment     | Docker, Docker Compose         |

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
python3 -m venv venv            # or `py -3.11 -m venv venv`
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
npm run convert-model
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

## Training the Model

We trained on 5 000 Flickr8k images for 10 epochs, reaching ~48% training accuracy and ~38% validation accuracy. To enhance performance:

- Integrate an attention mechanism (e.g., Bahdanau or Transformer attention)
- Increase model capacity (additional layers, larger hidden dimensions)
- Scale dataset to Flickr30k or MSCOCO
- Fine-tune the CNN encoder weights
- Experiment with beam search or nucleus sampling during decoding

```bash
cd backend/models
python train_model.py
```

---

## Project Structure

```
image-caption-app/
├── backend/            # FastAPI server & model code
│   ├── app.py          # API endpoints
│   ├── models/         # Model definition, training & checkpoints
│   └── utils/          # Preprocessing utilities
├── frontend/           # React application
│   ├── public/         # Static assets
│   └── src/            # Components & styles
├── data/               # Datasets (excluded from repo)
├── docker/             # Dockerfiles & compose configuration
└── README.md           # Project overview and instructions
```

---

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Inspired by TensorFlow image captioning tutorials
- Based on CNN–Transformer architecture research papers

