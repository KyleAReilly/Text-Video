import pyttsx3  # For text-to-speech
import whisper  # For transcribing audio to subtitles
from datetime import timedelta  # To format timestamps for SRT
import os  # For file and directory operations
import random  # To shuffle video files
import subprocess  # To run FFmpeg commands externally
import logging  # For improved logging
import shutil  # To check for dependency availability
from typing import Optional
import tempfile  # For managing temporary files

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_dependencies():
    """Ensure all required dependencies are installed."""
    required = ["ffmpeg", "ffprobe"]
    for cmd in required:
        if not shutil.which(cmd):
            raise EnvironmentError(f"Dependency {cmd} is not installed.")


def list_voices():
    """Lists all available voices on the system."""
    logger.info("Listing available voices...")
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for i, voice in enumerate(voices):
        print(f"{i}: {voice.name} ({voice.languages})")


def text_to_speech_with_voice(text: str, voice_index: int = 0, output_file: str = "output.mp3") -> None:
    """Converts text to speech using a specific voice and saves to a file."""
    logger.info("Converting text to speech...")
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    if 0 <= voice_index < len(voices):
        logger.info(f"Using voice: {voices[voice_index].name}")
        engine.setProperty('voice', voices[voice_index].id)
    else:
        logger.warning("Invalid voice index. Using default voice.")

    try:
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        logger.info(f"Audio saved to {output_file}")
    except Exception as e:
        logger.error(f"Error during text-to-speech conversion: {e}")
        raise


def format_timestamp(seconds: float) -> str:
    """Convert seconds into SRT-compatible timestamp format (HH:MM:SS,ms)."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def transcribe_to_srt(audio_path: str, srt_path: str, model_name: str = "base") -> None:
    """Convert an audio file to subtitles and save them as an SRT file."""
    logger.info("Transcribing audio to subtitles...")
    try:
        logger.info("Loading Whisper model...")
        model = whisper.load_model(model_name)
        logger.info("Transcribing audio...")
        result = model.transcribe(audio_path)

        logger.info(f"Saving subtitles to {srt_path}...")
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(result["segments"], start=1):
                start_time = format_timestamp(segment["start"])
                end_time = format_timestamp(segment["end"])
                text = segment["text"].strip()
                srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
        logger.info("Subtitles successfully saved.")
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise


def get_audio_duration(audio_path: str) -> float:
    """Get the duration of an audio file using FFmpeg's ffprobe."""
    logger.info("Calculating audio duration...")
    command = [
        "ffprobe", "-i", audio_path, "-show_entries", "format=duration",
        "-v", "quiet", "-of", "csv=p=0"
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        logger.info(f"Audio duration: {duration:.2f} seconds")
        return duration
    except Exception as e:
        logger.error(f"Error calculating audio duration: {e}")
        raise


def preprocess_video(input_video: str) -> str:
    """Standardize video for concatenation and return temporary file path."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    command = [
        "ffmpeg", "-y", "-i", input_video, "-vf", "scale=1920:1080,fps=30",
        "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-preset", "fast", "-an", temp_file
    ]
    try:
        logger.info(f"Preprocessing video: {input_video}")
        subprocess.run(command, check=True)
        logger.info(f"Preprocessed video saved to: {temp_file}")
        return temp_file
    except Exception as e:
        logger.error(f"Error preprocessing video: {e}")
        raise

def prepare_video_clips(video_folder: str, audio_duration: float, output_video: str) -> bool:
    """Prepares video clips to match the audio duration and concatenates them."""
    video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith((".mp4", ".mov"))]
    logger.info(f"Found {len(video_files)} video files in folder: {video_folder}")
    if not video_files:
        logger.error("No valid video files found in the folder.")
        return False

    random.shuffle(video_files)
    logger.info("Shuffled video files for random selection.")

    total_video_duration = 0
    filter_complex = []
    temp_files = []

    try:
        for i, video in enumerate(video_files):
            if total_video_duration >= audio_duration:
                logger.info("Reached the required total video duration.")
                break

            logger.info(f"Processing video: {video}")
            preprocessed_video = preprocess_video(video)
            temp_files.append(preprocessed_video)
            logger.info(f"Preprocessed video saved as temporary file: {preprocessed_video}")

            video_duration = get_audio_duration(preprocessed_video)
            logger.info(f"Duration of {preprocessed_video}: {video_duration} seconds")

            filter_complex.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
            total_video_duration += video_duration
            logger.info(f"Total accumulated video duration: {total_video_duration} seconds")

        if not filter_complex:
            logger.error("No videos were successfully processed.")
            return False

        # Construct concatenation filter
        concat_filter = f"{';'.join(filter_complex)};{''.join(f'[v{i}]' for i in range(len(temp_files)))}concat=n={len(temp_files)}:v=1[vconcat]"

        # Add final processing (format and fade) to the concatenated video
        final_filter = f"{concat_filter};[vconcat]fade=t=out:st={audio_duration - 2}:d=2,format=yuv420p[vout]"

        # Build the FFmpeg command
        command = (
            ["ffmpeg", "-y", "-loglevel", "info", "-hide_banner"] +
            [item for temp_file in temp_files for item in ["-i", temp_file]] +
            [
                "-filter_complex", final_filter,
                "-t", str(audio_duration),
                "-map", "[vout]",
                output_video
            ]
        )

        logger.info(f"Running FFmpeg command: {' '.join(command)}")
        subprocess.run(command, check=True)
        logger.info(f"Video prepared successfully and saved to {output_video}")
        return True

    except Exception as e:
        logger.error(f"Error preparing video clips: {e}")
        return False

    finally:
        for temp_file in temp_files:
            os.remove(temp_file)
            logger.info(f"Temporary file {temp_file} deleted.")


def combine_audio_video_subtitles(audio_path: str, srt_path: str, video_path: str, output_file: str) -> bool:
    """Combines audio, video, and subtitles into a single video file."""
    logger.info("Combining audio, video, and subtitles...")
    if not os.path.exists(video_path):
        logger.error(f"Video file {video_path} does not exist!")
        return False

    command = [
    "ffmpeg", "-y", "-loglevel", "warning", "-hide_banner",
    "-i", video_path, "-i", audio_path,
    "-vf", f"subtitles={srt_path}", "-c:v", "libx264", "-c:a", "aac",
    "-strict", "experimental", output_file
]


    try:
        logger.info(f"Running FFmpeg command to combine media...")
        subprocess.run(command, check=True)
        logger.info(f"Final video created successfully and saved as {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error combining media: {e}")
        return False


def main():
    """Main function to orchestrate the entire process."""
    check_dependencies()

    logger.info("Welcome to the Text-to-Subtitle-Video Tool!")
    list_voices()

    file_path = input("Enter the path to the .txt file: ").strip()
    video_folder = input("Enter the path to the folder with videos: ").strip()
    voice_choice = int(input("Choose a voice index: "))

    if not os.path.exists(file_path):
        logger.error(f"The file {file_path} does not exist.")
        return

    if not os.path.exists(video_folder):
        logger.error(f"The folder {video_folder} does not exist.")
        return

    audio_file = "output.mp3"
    srt_file = "output.srt"
    prepared_video = "prepared_video.mp4"
    final_video = "final_output.mp4"

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        text_to_speech_with_voice(text, voice_choice, audio_file)
    except Exception as e:
        logger.error(f"Error during text-to-speech conversion: {e}")
        return

    try:
        transcribe_to_srt(audio_file, srt_file)
    except Exception as e:
        logger.error(f"Error during subtitle generation: {e}")
        return

    try:
        if not prepare_video_clips(video_folder, get_audio_duration(audio_file), prepared_video):
            logger.error("Video preparation failed. Exiting.")
            return
    except Exception as e:
        logger.error(f"Error during video preparation: {e}")
        return

    if not os.path.exists(prepared_video):
        logger.error("Prepared video not found. Exiting.")
        return

    try:
        combine_audio_video_subtitles(audio_file, srt_file, prepared_video, final_video)
        logger.info(f"Process complete! Your final video is saved as {final_video}")
    except Exception as e:
        logger.error(f"Error during final video creation: {e}")


if __name__ == "__main__":
    main()
