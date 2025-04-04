# Real-Time Image Caption Generator Web App

This project is a web application that generates descriptive captions for uploaded images in real-time. It combines Computer Vision (using CNN) and Natural Language Processing (using Transformer) to create a complete, end-to-end AI solution.

## Project Overview

The Image Caption Generator uses a CNN-Transformer architecture to process images and generate relevant captions. The application provides a user-friendly interface for uploading images and viewing the generated captions.

### Key Features

- Upload images via drag-and-drop or file selection
- Real-time caption generation using a CNN-Transformer model
- Responsive UI with image preview
- RESTful API backend for caption generation

## Tech Stack

### Backend
- **Language & Framework:** Python with FastAPI
- **Deep Learning:** TensorFlow for the CNN-Transformer model
- **Image Processing:** Pillow for preprocessing

### Frontend
- **Framework:** React
- **HTTP Client:** Native Fetch API for API calls
- **Styling:** Custom CSS

### Deployment
- Local development with Uvicorn and React development server
- Docker support for containerized deployment

## Project Structure

```
image-caption-app/
├── backend/
│   ├── app.py                   # API endpoints & model serving
│   ├── models/
│   │   ├── caption_model.py     # CNN-Transformer architecture
│   │   ├── train_model.py       # Training script
│   │   └── model_weights.h5     # Saved model weights (after training)
│   ├── utils/
│   │   └── preprocess.py        # Image preprocessing utilities
│   ├── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── App.js               # Main application component
│   │   ├── index.js             # Entry point
│   │   └── styles.css           # Application styles
│   ├── package.json
├── data/                        # Dataset files (not included in repo)
├── docker/                      # Docker configuration files
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
├── docker-compose.yml
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+ (for backend)
- Node.js 14+ (for frontend)
- Docker and Docker Compose (optional, for containerized deployment)

### Installation and Setup

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/image-caption-app.git
cd image-caption-app
```
**Create a Virtual Environment:**

   python -m venv venv

   works 100% for Python 3.11.9
      python3.11 -m venv venv

**Activate the Virtual Environment:**
   
   Windows: 
      .\venv\Scripts\Activate.ps1

   Mac/Linux
      source venv/bin/activate


2. **Set up the backend**

```bash
cd backend
pip install -r requirements.txt
```

3. **Set up the frontend**

```bash
cd frontend
npm install
```

### Running the Application Locally

1. **Start the backend server**

```bash
cd backend
uvicorn app:app --reload
```

2. **Start the frontend development server**

```bash
cd frontend
npm start
```

3. **Access the application**
   
   Open your browser and navigate to `http://localhost:3000`

### Running with Docker

```bash
docker-compose up --build
```

## Training the Model

To train the model with your own dataset:

1. Download and place the Flickr8k dataset (or another image captioning dataset) in the `data/` directory
2. In data folder there is a file named  "Link_to_download_data.txt" navigate to it and there u can find the instruction where and how to downlaod the data
3. Navigate to the backend/models directory
4. Run the training script:

```bash
cd backend/models
python train_model.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was inspired by the TensorFlow Image Captioning with Visual Attention tutorial
- The model architecture is based on the CNN-Transformer approach for image captioning