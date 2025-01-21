import whisper # https://github.com/openai/whisper
from datetime import timedelta
import os

def format_timestamp(seconds):
    # hh:mm:ss,ms format
    td = timedelta(seconds=seconds)
    total_Seconds = int(td.total_seconds())
    hours = total_Seconds // 3600
    minutes = (total_Seconds % 3600) // 60
    seconds = total_Seconds % 60
    milliseconds = int((td.microseconds / 1000))
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def transcribe_to_srt(audio_path, srt_path, model_name="base"):
    # load the model
    print("loading whisper model to convert your audio")
    model = whisper.load_model(model_name)

    # transcribe audio
    result = model.transcribe(audio_path)

    # Write SRT file
    print(f"Writing SRT file to {srt_path}...")
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result["segments"], start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

            print("Audio Transcribed !")

# input audio file
audio_file = r"C:\Users\Cropt\PycharmProjects\ssrt\output.mp3" # Replace with your audio path
srt_file = os.path.splitext(audio_file)[0] + ".srt" # Output SRT file

# Run program transcription
transcribe_to_srt(audio_file, srt_file)

