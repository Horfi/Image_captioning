# Image Caption Generator Web App

A real-time web application that generates descriptive captions for uploaded images using a state-of-the-art CNN‑Transformer model.

![App Preview](https://github.com/user-attachments/assets/3a3be285-286e-416f-88fa-49812d2e7609)
![Captioning Demo](https://github.com/user-attachments/assets/92258e2f-1673-4006-8bad-5f640d67ced3)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
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

The Image Caption Generator Web App provides an end‑to‑end AI solution that combines computer vision and natural language processing. Users can upload an image and receive a human‑like descriptive caption in real time.

<img src="architecture_diagram.png" alt="Architecture diagram" width="600" />

### Goals
- Demonstrate a production‑ready pipeline for image captioning
- Offer a responsive, intuitive web UI
- Serve captions via a RESTful API built with FastAPI
- Support scalable deployment using Docker

---

## Features

- **Drag & Drop Upload**: Seamless image uploads via drag‑and‑drop or file picker
- **Instant Captioning**: Real‑time caption generation
- **Responsive UI**: Dynamic preview and caption display
- **REST API**: FastAPI backend exposes `/api/caption` endpoint

---

## Architecture

1. **Frontend** (React)
   - File upload component
   - Image preview and caption display
   - Fetch API integration

2. **Backend** (FastAPI)
   - `/api/caption`: accepts image uploads, returns generated caption
   - Model loading at startup for low‑latency inference

3. **Model** (TensorFlow)
   - CNN (InceptionV3) encoder for image feature extraction
   - Transformer decoder for sequence generation

---

## Tech Stack

| Layer       | Technology             |
|-------------|------------------------|
| Frontend    | React, Fetch API, CSS  |
| Backend     | Python, FastAPI, Uvicorn |
| ML Framework| TensorFlow, Keras      |
| Image I/O   | Pillow                 |
| Deployment  | Docker, Docker Compose |

---

## Getting Started

### Prerequisites
- **Python** ≥ 3.8
- **Node.js** ≥ 14
- **Docker** & **Docker Compose** (optional)

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/image-caption-app.git
    cd image-caption-app
    ```

2. **Backend setup**

    ```bash
    cd backend
    python3 -m venv venv               # or py -3.11 -m venv venv
    source venv/bin/activate           # Windows: .\venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. **Frontend setup**

    ```bash
    cd ../frontend
    npm install
    ```

### Running Locally

1. **Start backend**

    ```bash
    cd backend
    uvicorn app:app --reload
    ```

2. **Start frontend**

    ```bash
    cd frontend
    npm run convert-model
    npm start
    ```

3. **Open** `http://localhost:3000`

### Docker Deployment

Ensure Docker daemon is running, then:

```bash
cd docker
docker-compose up --build
```

Access the app at `http://localhost:3000` once containers are healthy.

---

## Training the Model

1. Place your dataset (e.g., Flickr8k) in `data/`.
2. Follow the download instructions in `data/Link_to_download_data.txt`.
3. Run:

    ```bash
    cd backend/models
    python train_model.py
    ```

After training, saved weights appear in `backend/models/`.

---

## Project Structure

```
image-caption-app/
├── backend/            # FastAPI server & model code
│   ├── app.py          # API endpoints
│   ├── models/         # Model definition & training
│   └── utils/          # Preprocessing utilities
├── frontend/           # React application
│   ├── public/
│   └── src/
├── data/               # Datasets (excluded from repo)
├── docker/             # Dockerfiles & compose
└── README.md
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Inspired by TensorFlow’s image captioning tutorials
- CNN‑Transformer architecture reference papers

