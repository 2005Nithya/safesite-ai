# SafeSite AI - Construction Site Safety Detection

Welcome to the SafeSite AI documentation page. This is your hub for information about our intelligent construction site safety detection system.

## Quick Links

- 🏠 [Home](/)
- 📖 [Documentation](#documentation)
- 🔧 [Setup Guide](#setup-guide)
- 🚀 [Deployment](#deployment)
- 📞 [Support](#support)

## Documentation

### What is SafeSite AI?

SafeSite AI is a state-of-the-art computer vision system built with Python and Flask that uses YOLOv8 neural networks to detect safety hazards on construction sites in real-time.

### Core Features

- **Real-time Video Processing**: Analyze live CCTV feeds and detect safety violations instantly
- **Image Analysis**: Upload construction site images for safety violation detection
- **Analytics Dashboard**: Track safety metrics, trends, and historical data
- **Multi-user Support**: Secure authentication and role-based access control
- **Cloud-Ready**: Pre-configured for deployment on Netlify and Render

## Setup Guide

### Prerequisites

- Python 3.8+
- Git
- Virtual Environment (recommended)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/2005Nithya/safesite-ai.git
   cd safesite-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## Deployment

### Netlify Deployment

SafeSite AI is configured for Netlify deployment. The `netlify.toml` file contains the necessary configuration.

### Render Deployment

Use the `render.yaml` file for deploying to Render platform.

### Docker Deployment

A `Dockerfile` is included for containerized deployment:

```bash
docker build -t safesite-ai .
docker run -p 5000:5000 safesite-ai
```

## Project Structure

```
safesite-ai/
├── app.py                 # Main Flask application
├── detection.py           # Detection logic
├── video_detection.py     # Video processing
├── serve.py               # Server configuration
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── netlify.toml           # Netlify configuration
├── render.yaml            # Render configuration
├── templates/             # HTML templates
├── static/                # CSS and JavaScript files
├── public/                # Public assets
├── tests/                 # Test files
└── dataset/               # YOLOv8 dataset
```

## Models

SafeSite AI uses YOLOv8 (You Only Look Once v8) pre-trained models for detection. Models are stored as `.pt` files:

- `best.pt` - Main detection model
- `best (1).pt` - Alternative model
- `dataset/last.pt` - Latest training checkpoint

## Technology Stack

- **Backend**: Python, Flask
- **Computer Vision**: YOLOv8, OpenCV
- **Frontend**: HTML, CSS, JavaScript
- **Database**: JSON-based history
- **Deployment**: Netlify, Render, Docker

## API Endpoints

The application provides REST API endpoints for:

- Image detection
- Video feed analysis
- Historical data retrieval
- User authentication
- Analytics dashboard data

## Support

For issues, questions, or contributions:

- 📧 Check existing [GitHub Issues](https://github.com/2005Nithya/safesite-ai/issues)
- 🐛 Report bugs with detailed information
- 💡 Suggest features or improvements

## License

This project is part of the SafeSite AI initiative.

---

**Last Updated**: August 2024  
**Repository**: [github.com/2005Nithya/safesite-ai](https://github.com/2005Nithya/safesite-ai)
