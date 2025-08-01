# Myoral

Myoral is a deep learning-powered backend for intelligent oral photo analysis. Users upload oral images via the frontend app; the backend automatically performs tooth segmentation, caries (cavity) detection, gum recession analysis, and returns diagnostic images and detailed reports. The system supports user registration, login, profile management, photo uploads, and includes comprehensive model training and data processing tools.

## Project Structure

```
Myoral/
├── routers/           # FastAPI routes (user, analysis, upload, etc.)
├── models/            # Trained model files (not included)
├── Training/          # Training scripts, dataset splitting, mask visualization
├── masks/             # Generated segmentation masks
├── uploads/           # Uploaded user images
├── users/             # User avatars and resources
├── dataset/           # Raw/processed datasets (not included)
├── myenv/             # Python virtual environment
├── .gitignore
├── requirements.txt
└── ...
```

## Main Features

- **User Management:** Registration (email/phone), login (JWT token), profile update, avatar upload, token authentication.
- **Oral Photo Analysis:** Upload oral images for AI-based segmentation and diagnosis (teeth, caries, gum recession, etc.), with diagnostic images and structured reports.
- **Model Inference:** Integrated YOLOv8 segmentation model, supports automatic GPU/CPU switching.
- **Data Management:** Unified storage and management of photos, masks, user info.
- **Training & Data Tools:** Custom dataset splitting, mask visualization, model training and fine-tuning.
- **RESTful API:** CORS support, suitable for frontend-backend separation.

## Quick Start

1. **Environment Setup**

   Python 3.8+ is recommended. Create a virtual environment:

   ```sh
   python -m venv myenv
   # On Windows:
   myenv\Scripts\activate
   # On macOS/Linux:
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**

   Copy `.env.example` to `.env` and fill in database, secret key, and other info.

3. **Prepare Model Files**

   Place your trained model files (e.g., `6_26best.pt`) in the `models/` directory.

4. **Start the Backend Server**

   ```sh
   uvicorn main:app --reload
   ```

5. **API Documentation**

   Visit [http://localhost:8000/docs](http://localhost:8000/docs) in your browser for all endpoints and testing.

## Core API Endpoints

All endpoints are prefixed with `/v1`:

- `POST   /v1/register`         — User registration (email/phone)
- `POST   /v1/login`            — User login, returns JWT token
- `PUT    /v1/users/{userId}`   — Update user info
- `GET    /v1/users/{userId}`   — Get user info
- `POST   /v1/photoUpload`      — Upload oral photo
- `GET    /v1/readPhoto/{photoId}` — Get original photo
- `POST   /v1/analysis`         — Upload and analyze photo, returns diagnostic image and report
- `GET    /v1/analysis/mask/{userId}` — Get segmentation mask (supports latest/all/specific time)
- See the `routers/` directory and `/docs` for more

## Oral Analysis & AI Diagnosis

- Tooth instance segmentation with automatic numbering and annotation
- Detection of caries (cavities), gum recession, and other oral issues
- Outputs numbered mask images and structured JSON reports
- Supports automatic PDF report generation

## Database Design

- Uses PostgreSQL and SQLAlchemy ORM
- Includes user table (`Users`), photo table (`photos_path`), etc.
- Supports email verification, token management, registration method distinction

## Training & Data Tools

- `train.py`: YOLOv8 segmentation model training script, supports custom hyperparameters
- `split_dataset.py`: Automatic train/validation split
- `visualize_mask.py`: Mask visualization for data inspection

## Directory & File Notes

- `models/`: All trained model files (must be downloaded/trained separately)
- `masks/`: Segmentation mask images generated by analysis
- `uploads/`: User-uploaded original photos
- `users/`: User avatars and resources
- `dataset/`: Raw/processed datasets (not distributed with code)

## Contribution

Feel free to open issues or submit pull requests! For custom training, new features, or frontend integration, refer to the `routers/` and `Training/` directories.

## Notes

- This project is for academic and research use only. Commercial use is prohibited.
- Model files and datasets must be prepared separately and are not included in the repository.
- Keep your secret keys and database credentials secure.

---

**Myoral — Intelligent Oral Analysis Backend for Better Oral Health Management**