# MoodTune AI

MoodTune AI is a Streamlit web app that recommends Spotify songs from an uploaded image. It combines image captioning, scene recognition, color psychology, face/smile analysis, OCR signals, and a smart mood scoring engine to turn a visual moment into a music vibe.

## Features

- Upload image or capture from webcam
- OpenCV color, brightness, contrast, face, and smile analysis
- Optional BLIP image captioning and CLIP scene recognition
- Optional DeepFace emotion detection
- OCR text extraction with pytesseract when Tesseract is installed
- Smart Mood Engine with mood score breakdown
- Color psychology mapping, including blue calm, red energetic, yellow happy, and dark tones emotional/cinematic
- Spotify Web API recommendations through Spotipy
- Spotify track links, embedded player, album covers, preview audio when Spotify provides it, and match percentages
- Multi-language controls for English, Hindi, Malayalam, Tamil, and Korean songs
- Recommendation styles: Trending, Classic, Indie, Lo-fi, and Viral
- Similar mood playlist search
- SQLite history of previous searches
- Lightweight visual mood embeddings for similar image recommendations
- Export playlist as PDF
- Optional Spotify playlist creation through OAuth

## Folder Structure

```text
MoodTune AI/
|-- app.py
|-- image_analyzer.py
|-- mood_engine.py
|-- spotify_recommender.py
|-- database.py
|-- utils.py
|-- requirements.txt
|-- requirements-local.txt
|-- requirements-ai.txt
|-- requirements-advanced.txt
|-- .env.example
`-- README.md
```

At runtime, the app also creates:

```text
moodtune_history.db
```

## Installation

1. Create a virtual environment.

```bash
python -m venv .venv
```

2. Activate it.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install the default dependencies.

```bash
pip install streamlit numpy Pillow
```

The default `requirements.txt` is intentionally comments-only for Streamlit Community Cloud. This avoids build failures from optional packages on the free cloud container.

For local or larger-host experiments with BLIP captioning and CLIP scene recognition, install:

```bash
pip install -r requirements-ai.txt
```

For local OpenCV, OCR, and richer PDF export support, install:

```bash
pip install -r requirements-local.txt
```

For local experiments with DeepFace emotion detection, install the optional advanced stack:

```bash
pip install -r requirements-advanced.txt
```

4. Copy the environment template.

```bash
copy .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

5. Run the app.

```bash
streamlit run app.py
```

## Spotify API Setup

1. Go to the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
2. Create an app.
3. Copy the Client ID and Client Secret.
4. Open the app settings and add this Redirect URI for local development:

```text
http://localhost:8501
```

5. Put the credentials in `.env`.

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8501
SPOTIFY_MARKET=IN
```

Client credentials are enough for search, album covers, previews, track links, and playlist discovery. OAuth is used only when exporting a generated playlist to a Spotify account.

## Streamlit Community Cloud Deployment

Deploy the app from Streamlit Community Cloud, not Vercel.

1. Go to `https://share.streamlit.io`.
2. Choose your GitHub repository and branch.
3. Set the entrypoint file to `app.py`.
4. In Advanced settings, choose Python 3.11 or 3.12. If you already created the app with another Python version, delete and redeploy it to change Python versions.
5. Add Spotify credentials in Secrets:

```toml
SPOTIFY_CLIENT_ID = "your_client_id"
SPOTIFY_CLIENT_SECRET = "your_client_secret"
SPOTIFY_REDIRECT_URI = "https://your-app-name.streamlit.app"
SPOTIFY_MARKET = "IN"
```

The default cloud deployment intentionally excludes `spotipy`, `opencv-python-headless`, `pandas`, `reportlab`, `pytesseract`, `torch`, `transformers`, and `deepface` to reduce build failures on small free containers.

## AI Model Notes

MoodTune AI is designed to start even when large AI models are not downloaded yet.

- The default `requirements.txt` is comments-only for Streamlit Community Cloud and uses lightweight NumPy/Pillow color and scene heuristics from Streamlit's base environment.
- If you install `requirements-local.txt`, the app can use OpenCV face/smile detection, OCR, and richer PDF export.
- If you install `requirements-ai.txt`, the app can use `Salesforce/blip-image-captioning-base` for image captions.
- If you install `requirements-ai.txt`, the app can use `openai/clip-vit-base-patch32` for zero-shot scene recognition.
- If those models are unavailable, the app falls back to OpenCV and color/scene heuristics.
- DeepFace is optional and lives in `requirements-advanced.txt`. If it cannot load, the app still detects faces and smiles with OpenCV.
- OCR requires both the `pytesseract` Python package and the Tesseract system executable.

The first model-backed run can take longer because HuggingFace downloads model weights.

## How Recommendations Work

1. `image_analyzer.py` extracts caption, scene labels, face data, colors, OCR text, activity hints, social context, and aesthetic style.
2. `mood_engine.py` scores the image across moods such as happy, calm, energetic, romantic, emotional, motivational, party, nostalgic, lonely, cinematic, and friendship.
3. `spotify_recommender.py` builds Spotify search queries from the detected vibe, preferred language, preferred genre, and recommendation style.
4. `database.py` stores the mood profile, song recommendations, image thumbnail, and visual mood embedding in SQLite.

## Troubleshooting

- Error installing requirements on Streamlit Cloud: use the default `requirements.txt`, choose Python 3.12 or 3.11 in Advanced settings, and check the build logs for the first package that failed.
- No Spotify covers or embeds: check `.env` locally or Streamlit Secrets in Cloud, then restart the app.
- Preview audio missing: Spotify does not provide `preview_url` for every track.
- BLIP or CLIP is slow on first run: the model is downloading or running on CPU.
- OCR returns nothing: install the Tesseract executable and ensure it is on your system PATH.
- DeepFace install issues: the app works without it because OpenCV face/smile detection is the fallback.

## Project Quality Notes

The app is intentionally modular:

- UI and workflow live in `app.py`.
- Image understanding lives in `image_analyzer.py`.
- Mood scoring lives in `mood_engine.py`.
- Spotify search/export lives in `spotify_recommender.py`.
- Persistence lives in `database.py`.
- Shared formatting, image, CSS, and PDF helpers live in `utils.py`.
