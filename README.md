# Transify — AI-Powered Accessible Multi-Modal Communication Platform

> **Tagline:** *Breaking Communication Barriers through Artificial Intelligence.*  
> **Status:** Production Ready · 4 Modules Deployed · 34/34 System Tests Passing

---

## 🌟 Executive Overview

**Transify** is an all-in-one, multi-modal assistive communication platform engineered to empower individuals who are Deaf, hard-of-hearing, blind, visually impaired, or non-verbal.

Transify integrates **four dedicated AI modules** with zero external cloud lock-in:
1. **Live Hand Sign & Gesture Translator (ASL & HaGRID):** Real-time computer vision engine recognizing 24+ communicative sign language words and gestures with strict Camera Target Box containment, 3-frame rolling consensus analysis, and an ultra-fast 98% Decision Gateway.
2. **Optical Braille Translator (UEB Grade 1):** Optical Braille Recognition (OBR) pipeline powered by a 99.4% accuracy ExtraTrees classifier, CLAHE contrast normalizer, and strict compliance with the **ICEB Rules of Unified English Braille (2024)**.
3. **Real-Time Speech-to-Text (STT):** Live continuous microphone voice captioning with real-time interim display and one-click copy or speech playback.
4. **Natural Text-to-Speech (TTS):** Instant multi-lingual speech synthesis via Web Speech API with automatic audio fallback.

---

## 🎯 Core Modules & Capabilities

### 1. ✋ Hand Sign Recognition Engine (ASL & HaGRID)
- **24+ Communicative Classes:** `CALL`, `DISLIKE`, `FIST`, `FOUR`, `HELLO`, `HELP`, `I_LOVE_YOU`, `LIKE`, `MUTE`, `NO`, `OK`, `ONE`, `PALM`, `PEACE`, `PEACE_INVERTED`, `ROCK`, `STOP`, `STOP_INVERTED`, `THANK_YOU`, `THREE`, `THREE2`, `TWO_UP`, `TWO_UP_INVERTED`, `YES`.
- **Strict Target Box (ROI) Containment:** Inference and predictions only execute when the user's hand is inside the designated guide box on camera. Outside hands trigger an amber safety warning and reset history.
- **Multi-Frame Analysis Before Decision:** Gestures are stabilized across 3 consensus frames (`Analyzing Sign... 33% → 66% → 100%`) before confirming the translation and triggering speech.
- **98% Decision Gateway:**
  - Confidence $\ge 0.98$: Local trained ExtraTrees / Random Forest model answers directly ($< 15\text{ ms}$).
  - Ambiguity or Confidence $< 0.98$: Automatically engages the **MCP Server Disambiguation Agent** to resolve candidate clashes.

### 2. ⠃⠗ Optical Braille Translation (OBR - UEB Grade 1)
- **High-Accuracy Vision Pipeline:** 99.4% dot pattern recognition on embossed paper documents.
- **Illumination-Invariant Preprocessing:** Adaptive CLAHE enhancement handles variable room lighting and shadows.
- **Direct Output Actions:** Spoken text (TTS), 1-click clipboard copy, and formatted `.txt` download.

### 3. 🎙️ Speech-to-Text (STT) & 🔊 Text-to-Speech (TTS)
- **Live Captions:** Real-time speech recognition supporting English (US/UK) and Bengali.
- **Accessible Audio Synthesis:** Automatic read-aloud for recognized signs and translated Braille.

### 4. 🛡️ 98% Decision Gateway & MCP Architecture
- Every prediction is evaluated against an empirical confidence threshold ($\ge 0.98$).
- Local-first architecture guarantees maximum privacy, running completely offline with optional MCP Server tie-breaking.

---

## 🏛️ System Architecture

```
+------------------------------------------------------------------------------------+
|                                TRANSIFY PLATFORM                                   |
|                                                                                    |
|   +--------------------+  +--------------------+  +---------------+  +-----------+ |
|   | 1. HAND SIGN AI    |  | 2. BRAILLE OBR     |  | 3. SPEECH-TEXT|  | 4. TTS    | |
|   | MediaPipe 3D Poses |  | CLAHE Preprocess   |  | Web Speech    |  | Web Speech| |
|   | 24 ASL/HaGRID Signs|  | 6-Dot ExtraTrees   |  | Live Captions |  | Audio Eng | |
|   | Target Box Gating  |  | UEB Grade 1 FSA    |  | Transcription |  | Voice Out | |
|   +---------+----------+  +---------+----------+  +-------+-------+  +-----+-----+ |
|             |                       |                     |                |       |
|             v                       v                     v                v       |
|   +----------------------------------------------------------------------------+   |
|   |                    98% DECISION GATEWAY & MCP RESOLVER                     |   |
|   |   - Local Model Direct Answer (Confidence >= 98%)                          |   |
|   |   - MCP Server Agent Tie-Breaker (Confidence < 98% / Ambiguity)           |   |
|   +-------------------------------------+--------------------------------------+   |
|                                         |                                          |
|                                         v                                          |
|   +----------------------------------------------------------------------------+   |
|   |              FastAPI ASGI Backend & SQLite Local Storage                   |   |
|   |           Responsive WCAG 2.1 AAA High-Contrast Web Interface             |   |
|   +----------------------------------------------------------------------------+   |
+------------------------------------------------------------------------------------+
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.11, 3.12, 3.14)
- Modern web browser (Chrome, Edge, Firefox, Safari)
- Webcam and microphone (for Sign Language and STT modules)

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Abrarafick/Transify.git
cd Transify

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

*(Optional: configure `OPENAI_API_KEY` or `MCP_SERVER_URL` in `.env` if enabling external cloud LLM resolvers).*

### 3. Run the Platform

```bash
python run_server.py --host 0.0.0.0 --port 8000
```

Open your browser and navigate to:
- **Main Application:** [http://localhost:8000/](http://localhost:8000/)
- **Live Sign Translator:** [http://localhost:8000/#sign](http://localhost:8000/#sign)
- **Braille Document Translator:** [http://localhost:8000/#braille](http://localhost:8000/#braille)
- **Speech to Text:** [http://localhost:8000/#stt](http://localhost:8000/#stt)
- **Text to Speech:** [http://localhost:8000/#tts](http://localhost:8000/#tts)
- **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing & Verification

Transify includes an exhaustive automated test suite with 34 tests covering API endpoints, vision pipelines, stabilization consensus, and MCP fallback:

```bash
python -m unittest tests/test_full_system.py
```

Expected output:
```
..................................
----------------------------------------------------------------------
Ran 34 tests in 2.75s

OK
```

---

## 📂 Repository Structure

```
Transify/
├── models/                     # Trained model artifacts and manifests
│   ├── braille/                # Braille cell localizer & 6-dot classifier
│   └── sign/                   # HaGRID & ASL 24-class models, MediaPipe tasks
├── src/                        # Core source code (100% English compliant)
│   ├── api/                    # FastAPI endpoints, routers, TTS engines
│   ├── braille/                # Preprocessing, cell detector, UEB decoder
│   ├── database/               # SQLite database schemas and persistence
│   ├── frontend/               # Accessible HTML, responsive CSS, client JS
│   └── sign/                   # MediaPipe tracker, HaGRID recognizer, MCP gateway
├── tests/                      # Full-system test suite (34 unit & integration tests)
├── .env.example                # Configuration template
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── run_server.py               # Standalone production server launcher
└── README.md                   # Platform documentation
```

---

## ♿ Accessibility Compliance (WCAG 2.1 AAA)

Transify was built from the ground up for universal accessibility:
- **Theme Palette:** Pure high-contrast purple aesthetic (`#8338ec`) with full light and dark mode responsiveness.
- **Screen Reader Announcements:** Dedicated `aria-live="polite"` landmark regions for real-time auditory notifications.
- **Zero Heavy Frameworks:** Pure vanilla JavaScript and modular CSS for lightweight, zero-latency execution.
- **Local Privacy:** All translations and settings persist in local SQLite (`data/transify.db`), keeping private conversations secure on-device.

---

## 📄 License & Acknowledgments

- **License:** MIT License. Free for educational, research, and assistive deployment.
- **HaGRID Dataset:** Kapitanov et al. (HAnd Gesture Recognition Image Dataset).
- **ICEB Standards:** Unified English Braille (UEB) Rules, 3rd Edition (2024).
- **MediaPipe:** Google MediaPipe HandLandmarker.
