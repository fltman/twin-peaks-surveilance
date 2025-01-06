import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
from openai import OpenAI
from datetime import datetime
import json
import random
import subprocess
import base64
from pathlib import Path
import requests
from dotenv import load_dotenv
import ffmpeg  # Replace moviepy with ffmpeg-python

load_dotenv()

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Add API keys from environment variables
config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
config['elevenlabs_api_key'] = os.getenv('ELEVENLABS_API_KEY')

if not config['openai_api_key'] or not config['elevenlabs_api_key']:
    raise ValueError("Missing required API keys in .env file")

# Initialize OpenAI
openai_client = OpenAI(api_key=config['openai_api_key'])

class Character:
    def __init__(self, name, personality, voice_id):
        self.name = name
        self.personality = personality
        self.voice_id = voice_id

def extract_keyframes(video_path, num_frames=8):
    """Extract keyframes using FFmpeg"""
    temp_dir = Path("temp_frames")
    print(f"\nKeyframe extraction:")
    print(f"Creating temporary directory: {temp_dir.absolute()}")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Get video info using ffmpeg-python's probe
        print(f"Probing video file: {video_path}")
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        duration = float(probe['format']['duration'])
        print(f"Video duration: {duration:.2f}s")
        print(f"Video info: {json.dumps(video_info, indent=2)}")
        
        # Calculate timestamps
        timestamps = [i * (duration / (num_frames)) for i in range(num_frames)]
        print(f"Extracting frames at timestamps: {[f'{t:.2f}s' for t in timestamps]}")
        
        keyframes = []
        for i, timestamp in enumerate(timestamps):
            output_path = temp_dir / f"frame_{i}.jpg"
            print(f"\nProcessing frame {i+1}/{num_frames}:")
            print(f"Output path: {output_path.absolute()}")
            
            try:
                # Extract frame using timestamp-based seeking
                print(f"Extracting frame at {timestamp:.2f}s...")
                stream = ffmpeg.input(video_path, ss=timestamp)
                stream = ffmpeg.output(stream, str(output_path), 
                                    vframes=1, 
                                    loglevel='error',
                                    **{'qscale:v': 2})
                
                # Print FFmpeg command
                print(f"FFmpeg command: {' '.join(stream.get_args())}")
                
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
                
                if output_path.exists() and output_path.stat().st_size > 0:
                    with open(output_path, 'rb') as f:
                        frame_data = f.read()
                        keyframes.append(base64.b64encode(frame_data).decode())
                    print(f"Frame size: {output_path.stat().st_size} bytes")
                else:
                    print(f"Warning: Failed to extract frame {i+1}/{num_frames} at {timestamp:.2f}s")
            
            except ffmpeg.Error as e:
                print(f"FFmpeg error on frame {i+1}/{num_frames}: {e.stderr.decode()}")
            except Exception as e:
                print(f"Error processing frame {i+1}/{num_frames}: {str(e)}")
        
        if not keyframes:
            raise Exception("Failed to extract any keyframes from video")
        
        print(f"Successfully extracted {len(keyframes)}/{num_frames} frames from video")
        print(f"Frames saved in: {temp_dir.absolute()}")
        return keyframes
        
    except Exception as e:
        print(f"Error in extract_keyframes: {str(e)}")
        raise

def analyze_frames_with_gpt4(frames, character, video_duration):
    print(f"\nAnalyzing frames with GPT-4:")
    print(f"Character: {character.name}")
    print(f"Personality: {character.personality[:100]}...")
    print(f"Number of frames: {len(frames)}")
    
    # Calculate target response length
    chars_per_second = 10
    target_length = int(video_duration * chars_per_second)
    print(f"Video duration: {video_duration:.2f}s")
    print(f"Target response length: {target_length} characters")
    
    # Create content array with all frames
    content = [
        {
            "type": "text",
            "text": f"You will be shown {len(frames)} sequential frames from a video. Describe what happens in these frames according to this personality: {character.personality}. Your response should be approximately {target_length} characters long."
        }
    ]
    
    # Add all frames to the content array
    for i, frame_data in enumerate(frames, 1):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_data}"
            }
        })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    print("\nSending request to GPT-4...")
    print(f"Message text: {content[0]['text']}")
    print(f"Number of images in request: {len(frames)}")
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000  # Increased token limit for multiple frames
    )
    
    narrative = response.choices[0].message.content
    print(f"\nResponse received:")
    print(f"Narrative: {narrative}")
    
    return narrative

def add_voiceover_to_video(video_path, text, voice_id, output_path):
    """Generate voiceover using ElevenLabs API and add it to the video"""
    print(f"\nGenerating voiceover:")
    print(f"Voice ID: {voice_id}")
    print(f"Text length: {len(text)} characters")
    print(f"Text preview: {text[:200]}...")
    
    # Create temporary directory if it doesn't exist
    temp_dir = Path("temp_audio")
    temp_dir.mkdir(exist_ok=True)
    
    # Save audio to temporary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_audio_path = temp_dir / f'voice_{timestamp}.mp3'
    
    try:
        # ElevenLabs API endpoint
        eleven_labs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        print(f"Making request to: {eleven_labs_url}")
        
        # Request headers
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": config['elevenlabs_api_key']
        }
        
        # Request data with voice settings
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.3,
                "similarity_boost": 0.95,
                "style": 0.47,
                "use_speaker_boost": True
            }
        }
        
        # Make request to ElevenLabs API
        response = requests.post(eleven_labs_url, json=data, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"ElevenLabs API error: {response.text}")
        
        try:
            # Save the audio response
            print(f"Saving audio to: {temp_audio_path}")
            with open(temp_audio_path, 'wb') as f:
                f.write(response.content)
            print(f"Audio file size: {temp_audio_path.stat().st_size} bytes")
            
            try:
                print("\nCombining video and audio:")
                print(f"Input video: {video_path}")
                print(f"Input audio: {temp_audio_path}")
                print(f"Output path: {output_path}")
                
                # Use subprocess to run ffmpeg directly
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', str(temp_audio_path),
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0',
                    '-shortest',
                    '-y',
                    str(output_path)
                ]
                
                print(f"FFmpeg command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"FFmpeg error: {result.stderr}")
                    raise Exception("Failed to combine video and audio")
                    
                print(f"Successfully created video with voiceover: {output_path}")
                print(f"Output file size: {Path(output_path).stat().st_size} bytes")
                
            except Exception as e:
                print(f"Error combining video and audio: {str(e)}")
                raise
                
        except Exception as e:
            print(f"Error combining video and audio: {str(e)}")
            raise
            
    finally:
        print(f"Temporary audio file saved at: {temp_audio_path}")

def is_file_ready(file_path, check_time=5, check_interval=0.5):
    """
    Check if a file is ready for processing by monitoring its size over time.
    
    Args:
        file_path: Path to the file to check
        check_time: Total time to monitor in seconds
        check_interval: Time between size checks in seconds
    
    Returns:
        bool: True if file size is stable, False if still growing
    """
    initial_size = Path(file_path).stat().st_size
    checks = int(check_time / check_interval)
    
    for _ in range(checks):
        time.sleep(check_interval)
        current_size = Path(file_path).stat().st_size
        
        if current_size != initial_size:
            return False
        initial_size = current_size
    
    return True

def cleanup_temp_directories():
    """Clean up temporary directories and their contents"""
    temp_dirs = ['temp_frames', 'temp_audio']
    for dir_name in temp_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"\nCleaning up {dir_path}:")
            for file in dir_path.glob('*'):
                try:
                    file.unlink()
                    print(f"Deleted: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
            try:
                dir_path.rmdir()
                print(f"Removed directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")

def cleanup_input_file(file_path):
    """Move or delete processed input file"""
    try:
        input_path = Path(file_path)
        if input_path.exists():
            # Option 1: Delete the file
            input_path.unlink()
            print(f"Deleted input file: {input_path}")
            
            # Option 2: Move to a processed folder
            # processed_dir = Path(config['watch_directory']) / 'processed'
            # processed_dir.mkdir(exist_ok=True)
            # new_path = processed_dir / input_path.name
            # input_path.rename(new_path)
            # print(f"Moved input file to: {new_path}")
    except Exception as e:
        print(f"Error cleaning up input file {file_path}: {e}")

class VideoHandler(FileSystemEventHandler):
    def __init__(self, characters):
        self.characters = characters
        self.processing_files = set()
        
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(('.mp4', '.mov', '.avi')):
            return
        
        file_path = event.src_path
        
        # Skip if we're already processing this file
        if file_path in self.processing_files:
            return
            
        try:
            self.processing_files.add(file_path)
            
            # Wait for initial file creation
            time.sleep(1)
            
            # Check if file is ready for processing
            print(f"Waiting for file to be completely written: {file_path}")
            if not is_file_ready(file_path):
                print(f"File is still being written to: {file_path}")
                return
                
            print(f"Processing file: {file_path}")
            
            # Process the video
            video_path = event.src_path
            character = random.choice(self.characters)
            
            # Extract keyframes and get video duration
            probe = ffmpeg.probe(video_path)
            video_duration = float(probe['format']['duration'])
            keyframes = extract_keyframes(video_path)
            
            # Analyze frames with GPT-4, passing the duration
            narrative = analyze_frames_with_gpt4(keyframes, character, video_duration)
            
            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            # Replace spaces with underscores in filename
            safe_name = base_name.replace(' ', '_')
            output_path = os.path.join(
                config['output_directory'],
                f"{safe_name}_{character.name}_{timestamp}.mp4"
            )
            
            # Add voiceover
            add_voiceover_to_video(video_path, narrative, character.voice_id, output_path)
            
            print(f"Completed processing: {file_path}")
            
            # Clean up after successful processing
            cleanup_temp_directories()
            cleanup_input_file(file_path)
            
        except Exception as e:
            print(f"Error processing video {event.src_path}: {str(e)}")
            # Clean up temp files even if processing failed
            cleanup_temp_directories()
        finally:
            self.processing_files.remove(file_path)

def load_characters():
    """Load character configurations from JSON files in the characters directory"""
    characters_dir = Path("characters")
    characters = []
    
    if not characters_dir.exists():
        raise FileNotFoundError("Characters directory not found")
    
    for char_file in characters_dir.glob("*.json"):
        try:
            with open(char_file, 'r') as f:
                char_data = json.load(f)
                # Try personality first, fall back to prompt if personality doesn't exist
                personality = char_data.get('personality', char_data.get('prompt'))
                if not personality:
                    raise ValueError(f"Neither 'personality' nor 'prompt' found in {char_file}")
                    
                characters.append(Character(
                    name=char_data['name'],
                    personality=personality,
                    voice_id=char_data['voice_id']
                ))
        except Exception as e:
            print(f"Error loading character from {char_file}: {str(e)}")
    
    if not characters:
        raise ValueError("No valid character configurations found")
    
    return characters

def main():
    try:
        # Clean up any leftover temporary files from previous runs
        cleanup_temp_directories()
        
        characters = load_characters()
        print(f"Loaded {len(characters)} characters")
        
        # Create output directory if it doesn't exist
        output_dir = Path(config['output_directory'])
        output_dir.mkdir(exist_ok=True)
        
        # Set up folder monitoring
        event_handler = VideoHandler(characters)
        observer = Observer()
        observer.schedule(event_handler, config['watch_directory'], recursive=False)
        observer.start()
        
        print(f"Monitoring directory: {config['watch_directory']}")
        print(f"Output directory: {config['output_directory']}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    
    except Exception as e:
        print(f"Error initializing: {str(e)}")
        return

if __name__ == "__main__":
    main() 