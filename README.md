# Twin Peaks Surveillance

[![Support me on Patreon](https://img.shields.io/badge/Patreon-Support%20my%20work-FF424D?style=flat&logo=patreon&logoColor=white)](https://www.patreon.com/AndersBjarby)

Drop a video into a folder and have characters from Twin Peaks watch and comment on it. The app extracts keyframes, runs them through GPT-4o vision, and has one of the show's characters narrate the footage — voiced with ElevenLabs.

## How it works

- Watches the `in/` directory for new video files (using `watchdog`)
- Extracts keyframes with FFmpeg
- Sends frames to GPT-4o for analysis
- A character (Agent Cooper, Gordon Cole, the Log Lady, the Giant, Albert, and others — defined in `characters/`) reacts in their own voice and personality
- Generates ElevenLabs speech and writes results to the `out/` directory

## Setup

Requires `OPENAI_API_KEY` and `ELEVENLABS_API_KEY` in a `.env` file, plus FFmpeg installed on the system.

```bash
pip install -r requirements.txt
python movie_processor.py
```

Then drop a video file into the `in/` folder.

## Tech

Python, watchdog, OpenAI (GPT-4o vision), ffmpeg-python, OpenCV, ElevenLabs TTS.
