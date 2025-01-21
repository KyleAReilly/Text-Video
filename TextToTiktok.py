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
import configparser # used for our config.ini values

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "default_text_file": "text.txt",
    "video_folder": "videos",
    "voice_index": "0",
    "rate": "160",
    "output_audio_file": "output.mp3",
    "output_video_file": "final_output.mp4",
    "use_ass_subtitles": "no",
    "subtitle_format": "output.srt"
}

def load_config(config_file: str) -> dict:
    """
    Load and validate the configuration file. Applies defaults for invalid or missing values.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing the validated configuration.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(config_file):
        logger.error(f"Configuration file '{config_file}' not found. Exiting.")
        raise FileNotFoundError(f"Configuration file '{config_file}' is missing.")

    config.read(config_file)

    # Check if the 'General' section exists
    if "General" not in config:
        logger.error("Missing 'General' section in the configuration file.")
        raise KeyError("Configuration file must contain a [General] section.")

    general_config = config["General"]
    validated_config = DEFAULT_CONFIG.copy()

    # Validate and apply configuration options
    try:
        validated_config["default_text_file"] = general_config.get("default_text_file", DEFAULT_CONFIG["default_text_file"])
        if not os.path.exists(validated_config["default_text_file"]):
            logger.warning(f"Text file '{validated_config['default_text_file']}' not found. Using default.")

        validated_config["video_folder"] = general_config.get("video_folder", DEFAULT_CONFIG["video_folder"])
        if not os.path.isdir(validated_config["video_folder"]):
            logger.warning(f"Video folder '{validated_config['video_folder']}' not found. Using default.")

        validated_config["voice_index"] = general_config.get("voice_index", DEFAULT_CONFIG["voice_index"])
        if not validated_config["voice_index"].isdigit():
            logger.warning(f"Invalid voice index '{validated_config['voice_index']}'. Using default.")
            validated_config["voice_index"] = DEFAULT_CONFIG["voice_index"]

        validated_config["rate"] = general_config.get("rate", DEFAULT_CONFIG["rate"])
        if not validated_config["rate"].isdigit():
            logger.warning(f"Invalid TTS rate '{validated_config['rate']}'. Using default.")
            validated_config["rate"] = DEFAULT_CONFIG["rate"]

        validated_config["use_ass_subtitles"] = general_config.get("use_ass_subtitles", DEFAULT_CONFIG["use_ass_subtitles"]).strip().lower()
        if validated_config["use_ass_subtitles"] not in ["yes", "no"]:
            logger.warning(f"Invalid value for 'use_ass_subtitles': '{validated_config['use_ass_subtitles']}'. Using default.")
            validated_config["use_ass_subtitles"] = DEFAULT_CONFIG["use_ass_subtitles"]

        validated_config["subtitle_format"] = general_config.get("subtitle_format", DEFAULT_CONFIG["subtitle_format"])

        # Validate subtitle_font_size
        subtitle_font_size_str = general_config.get("subtitle_font_size", None)
        if subtitle_font_size_str:
            try:
                validated_config["subtitle_font_size"] = int(subtitle_font_size_str.strip())
            except ValueError:
                logger.warning(f"Invalid 'subtitle_font_size': '{subtitle_font_size_str}'. Using default: {DEFAULT_CONFIG['subtitle_font_size']}.")
                validated_config["subtitle_font_size"] = int(DEFAULT_CONFIG["subtitle_font_size"])
        else:
            logger.warning(f"'subtitle_font_size' not found in configuration. Using default: {DEFAULT_CONFIG['subtitle_font_size']}.")
            validated_config["subtitle_font_size"] = int(DEFAULT_CONFIG["subtitle_font_size"])

    except Exception as e:
        logger.error(f"Unexpected error while parsing configuration: {e}")
        raise

    return validated_config


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


def text_to_speech_with_voice(
    text: str, tts_rate: int, voice_index: int = 0, output_file: str = "output.mp3"
) -> None:
    """
    Converts text to speech using a specific voice and saves to a file.
    Allows customization of the speech rate.
    """
    logger.info("Converting text to speech...")
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    if 0 <= voice_index < len(voices):
        logger.info(f"Using voice: {voices[voice_index].name}")
        engine.setProperty('voice', voices[voice_index].id)
    else:
        logger.warning("Invalid voice index. Using default voice.")

    # Set the speech rate
    logger.info(f"Setting TTS rate to {tts_rate} words per minute.")
    engine.setProperty('rate', tts_rate)

    try:
        engine.save_to_file(text, output_file)
        engine.runAndWait()
        logger.info(f"Audio saved to {output_file}")
    except Exception as e:
        logger.error(f"Error during text-to-speech conversion: {e}")
        raise



def format_timestamp(seconds: float, for_ass: bool = False) -> str:
    """
    Convert seconds into timestamp format.
    - For SRT: HH:MM:SS,ms
    - For ASS: H:MM:SS.ms
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    milliseconds = int((td.total_seconds() - total_seconds) * 1000)

    if for_ass:
        # ASS format: H:MM:SS.ms
        return f"{hours}:{minutes:02}:{seconds:02}.{milliseconds:02}"
    else:
        # SRT format: HH:MM:SS,ms
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def transcribe_to_srt(audio_path: str, srt_path: str, model_name: str = "base", offset: float = -0.00) -> None:
    """Convert audio to subtitles with word-level or segment-level timestamps."""
    logger.info("Transcribing audio to subtitles...")
    try:
        logger.info("Loading Whisper model...")
        model = whisper.load_model(model_name)

        logger.info("Transcribing audio with word-level timestamps...")
        result = model.transcribe(audio_path, word_timestamps=True)

        if not result.get("segments"):
            logger.error("No segments found in transcription result.")
            return

        logger.info(f"Saving subtitles to {srt_path}...")
        with open(srt_path, "w", encoding="utf-8") as srt_file:
            index = 1
            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]

                # Use segment-level text if no word-level timestamps
                if not segment.get("words"):
                    logger.warning(f"Skipping word-level timestamps for segment: {segment.get('text')}")
                    srt_file.write(
                        f"{index}\n"
                        f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n"
                        f"{segment['text'].strip()}\n\n"
                    )
                    index += 1
                    continue

                # Process word-level timestamps
                for word_data in segment["words"]:
                    if all(key in word_data for key in ["word", "start", "end"]):
                        word_start = max(0, word_data["start"] + offset)
                        word_end = max(0, word_data["end"] + offset)
                        word_text = word_data["word"].strip()

                        srt_file.write(
                            f"{index}\n"
                            f"{format_timestamp(word_start)} --> {format_timestamp(word_end)}\n"
                            f"{word_text}\n\n"
                        )
                        index += 1
                    else:
                        logger.warning(f"Skipping invalid word data: {word_data}")

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


def preprocess_video(video_path: str, output_path: str) -> str:
    """
    Preprocess video to TikTok format: maintain original width, scale height to 1920,
    and crop to fill 1080x1920 frame (cropping sides).
    """
    try:
        logger.info(f"Preprocessing video: {video_path}")

        # FFmpeg command to scale height to 1920 and crop excess sides
        command = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", (
                "scale=-1:1920,"                      # Scale height to 1920, width adjusts proportionally
                "crop=1080:1920"                      # Crop to center to fit the 1080x1920 frame
            ),
            "-c:v", "libx264", "-crf", "23", "-preset", "medium",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]

        subprocess.run(command, check=True)
        logger.info(f"Preprocessed video saved to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
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
            preprocessed_video = preprocess_video(video, tempfile.mktemp(suffix=".mp4"))
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

        # Add fade and format filters
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

def generate_ass_from_srt(srt_path: str, ass_path: str, format_ass_path: str) -> None:
    """
    Converts an SRT file to an ASS file with enhanced styling and positioning.
    Ensures words appear one at a time without overlap.
    """
    try:
        logger.info("Converting SRT to ASS...")

        # Load the format.ass template
        with open(format_ass_path, "r", encoding="utf-8") as format_file:
            format_template = format_file.read()

        with open(ass_path, "w", encoding="utf-8") as ass_file:
            # Write the format.ass template as the header
            ass_file.write(format_template)

            # Start the [Events] section
            ass_file.write("\n[Events]\n")
            ass_file.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            with open(srt_path, "r", encoding="utf-8") as srt_file:
                start_time = 0.0
                end_time = 0.0
                for line in srt_file:
                    if "-->" in line:
                        # Parse SRT time
                        times = line.strip().split(" --> ")
                        start_time = float(times[0].split(":")[-1].replace(",", "."))
                        end_time = float(times[1].split(":")[-1].replace(",", "."))
                    elif line.strip().isdigit() or not line.strip():
                        continue
                    else:
                        # Process text and ensure each word is sequential
                        words = line.strip().split()
                        word_count = len(words)
                        duration = (end_time - start_time) / word_count
                        for i, word in enumerate(words):
                            word_start = start_time + (i * duration)
                            word_end = word_start + duration
                            # Write each word as a separate line in ASS format
                            ass_file.write(
                                f"Dialogue: 0,{format_timestamp(word_start, for_ass=True)},"
                                f"{format_timestamp(word_end, for_ass=True)},Default,,0,0,0,,{word}\n"
                            )
        logger.info(f"ASS file generated at {ass_path}")
    except Exception as e:
        logger.error(f"Error generating ASS file: {e}")
        raise


        logger.info(f"ASS file generated successfully at {ass_output}")
    except Exception as e:
        logger.error(f"Error generating ASS file: {e}")
        raise

def generate_fixed_ass(srt_path: str, format_ass_path: str, ass_output: str) -> None:
    """
    Converts an SRT file into a properly formatted ASS file using a template,
    ensuring no overlapping subtitles and one word at a time.
    """
    try:
        logger.info("Converting SRT to fixed ASS format...")

        # Load the format.ass template
        with open(format_ass_path, "r", encoding="utf-8") as format_file:
            format_template = format_file.read()

        with open(ass_output, "w", encoding="utf-8") as ass_file:
            # Write the format template as the header
            ass_file.write(format_template)

            # Ensure only one [Events] section
            if "[Events]" not in format_template:
                ass_file.write("\n[Events]\n")
                ass_file.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

            # Process the SRT file and add each word as a separate dialogue
            with open(srt_path, "r", encoding="utf-8") as srt_file:
                for line in srt_file:
                    if "-->" in line:
                        # Parse time intervals
                        times = line.strip().split(" --> ")
                        start_time = float(times[0].replace(",", ".").split(":")[-1]) + \
                                     int(times[0].split(":")[-2]) * 60 + \
                                     int(times[0].split(":")[-3]) * 3600
                        end_time = float(times[1].replace(",", ".").split(":")[-1]) + \
                                   int(times[1].split(":")[-2]) * 60 + \
                                   int(times[1].split(":")[-3]) * 3600
                    elif line.strip().isdigit() or not line.strip():
                        continue
                    else:
                        # Split line into individual words
                        words = line.strip().split()
                        duration_per_word = (end_time - start_time) / len(words)

                        # Write each word as a separate line in ASS format
                        for i, word in enumerate(words):
                            word_start = start_time + i * duration_per_word
                            word_end = word_start + duration_per_word
                            ass_file.write(
                                f"Dialogue: 0,{format_timestamp(word_start, for_ass=True)},"
                                f"{format_timestamp(word_end, for_ass=True)},Default,,0,0,0,,{word}\n"
                            )

        logger.info(f"Fixed ASS file generated successfully at {ass_output}")
    except Exception as e:
        logger.error(f"Error generating fixed ASS file: {e}")
        raise


def combine_audio_video_subtitles(
        audio_path: str, subtitle_path: str, video_path: str, output_file: str, use_ass: bool, subtitle_font_size: int
) -> bool:
    """
    Combines audio, video, and subtitles into a single video file.
    """
    logger.info("Combining audio, video, and subtitles...")

    # Check subtitle file
    if not os.path.exists(subtitle_path) or os.stat(subtitle_path).st_size == 0:
        logger.error(f"Subtitle file {subtitle_path} is missing or empty.")
        return False

    # Determine subtitle format and options
    subtitle_format = "ass" if use_ass else "srt"
    subtitle_option = (
        f"subtitles={subtitle_path}:force_style='FontSize={subtitle_font_size},Alignment=6,MarginV=120,MarginL=50,MarginR=50'"
        if subtitle_format == "srt" else f"ass={subtitle_path}"
    )
    logger.debug(f"Subtitle option: {subtitle_option}")

    # Construct FFmpeg command
    command = [
        "ffmpeg", "-y", "-loglevel", "warning", "-hide_banner",
        "-i", video_path, "-i", audio_path,
        "-vf", subtitle_option,
        "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental",
        output_file,
    ]

    # Execute FFmpeg command
    try:
        logger.info(f"Running FFmpeg command: {' '.join(command)}")
        subprocess.run(command, check=True)
        logger.info(f"Final video created successfully and saved as {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error combining media: {e}")
        return False



def main():
    """Main function to orchestrate the entire process."""
    # Check dependencies
    check_dependencies()

    # Load and validate configuration
    CONFIG_FILE = "config.ini"
    config = load_config(CONFIG_FILE)

    logger.info("Welcome to the Text-to-Subtitle-Video Tool!")
    logger.info("Loaded configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Extract validated configuration values
    file_path = config["default_text_file"]
    video_folder = config["video_folder"]
    voice_index = int(config["voice_index"])
    tts_rate = int(config["rate"])
    audio_file = config["output_audio_file"]
    final_video = config["output_video_file"]
    use_ass = config["use_ass_subtitles"] == "yes"
    subtitle_path = config["subtitle_format"]
    subtitle_font_size = config["subtitle_font_size"]  # Already validated and parsed as int in load_config
    srt_file = "output.srt"
    ass_file = "output.ass"
    prepared_video = "prepared_video.mp4"
    format_ass_path = config.get("ass_template_path", "format.ass")
    model_name = config.get("model_name", "base")
    whisper_offset = float(config.get("whisper_offset", 0.0))

    # Validate paths
    if not os.path.exists(file_path):
        logger.error(f"The file {file_path} does not exist.")
        return

    if not os.path.isdir(video_folder):
        logger.error(f"The folder {video_folder} does not exist.")
        return

    # Convert text to speech
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        text_to_speech_with_voice(text, tts_rate, voice_index, audio_file)
    except Exception as e:
        logger.error(f"Error during text-to-speech conversion: {e}")
        return

    # Generate subtitles
    try:
        transcribe_to_srt(audio_file, srt_file, model_name, whisper_offset)
    except Exception as e:
        logger.error(f"Error during subtitle generation: {e}")
        return

    # Prepare video clips
    try:
        if not prepare_video_clips(video_folder, get_audio_duration(audio_file), prepared_video):
            logger.error("Video preparation failed. Exiting.")
            return
    except Exception as e:
        logger.error(f"Error during video preparation: {e}")
        return

    # Generate ASS file if required
    if use_ass:
        try:
            generate_fixed_ass(srt_file, format_ass_path, ass_file)
            subtitle_path = ass_file  # Update to use the ASS file
        except Exception as e:
            logger.error(f"Error generating ASS subtitles: {e}")
            return

    # Combine audio, video, and subtitles
    try:
        combine_audio_video_subtitles(
            audio_file, subtitle_path, prepared_video, final_video, use_ass, subtitle_font_size
        )
        logger.info(f"Process complete! Your final video is saved as {final_video}")
    except Exception as e:
        logger.error(f"Error during final video creation: {e}")


if __name__ == "__main__":
    main()
