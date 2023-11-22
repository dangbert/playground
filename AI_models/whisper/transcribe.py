#!/usr/bin/env python3
################################################################################
# references:
# https://github.com/ElliotGestrin/NAOChat/blob/3f0f4dfce49eed2803f80564f40c6ed3c96cb5b5/src/Listener.py
# https://brunoscheufler.com/blog/2023-03-12-generating-subtitles-in-real-time-with-openai-whisper-and-pyaudio
################################################################################

import os
import argparse
import time
import wave

import whisper
import speech_recognition as sr
import pyaudio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WAVE_BUFFER_FNAME = "buffer.wav"


def main():
    parser = argparse.ArgumentParser(
        description="transcribe a file with whisper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fname",
        type=str,
        help="filename to transcribe",
    )
    parser.add_argument(
        "--model", type=str, help="Model version to use", default="tiny.en"
    )

    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List available microphones and exit",
    )

    parser.add_argument(
        "--mic",
        type=int,
        default=None,
        help="Index of microphone to use (see --list-mics)",
    )

    args = parser.parse_args()

    if args.list_mics:
        print(f"microphone list:")
        for i, m in enumerate(sr.Microphone.list_microphone_names()):
            print(f"   {i}: {m}")
        exit(0)

    # transcribe single file if provided
    print(f"using model {args.model}")
    model = whisper.load_model(args.model)
    if args.fname:
        print(f"transcribing '{args.fname}'")
        dur = time.time()
        res = model.transcribe(args.fname)
        dur = time.time() - dur

        print(f"\ntranscription (took {dur:.2f} seconds):")
        print(res["text"])
        exit(0)

    # live transcribe from mic
    # mic = sr.Microphone(device_index=args.mic)
    # rec = sr.Recognizer()
    # rec.adjust_for_ambient_noise(mic)

    # LISTEN_DUR = 4
    print("recording...", flush=True)
    fname = record_audio(args.mic)
    print("done!", flush=True)

    res = model.transcribe(fname)
    print(f"\nlive transcription:")
    print(res["text"])


def record_audio(device_index: int, dur_sec: int = 3):
    """
    From https://brunoscheufler.com/blog/2023-03-12-generating-subtitles-in-real-time-with-openai-whisper-and-pyaudio
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=device_index,
    )

    frames = []

    for _ in range(0, int(RATE / CHUNK * dur_sec)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_BUFFER_FNAME, "wb")
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b"".join(frames))
    waveFile.close()
    return WAVE_BUFFER_FNAME


if __name__ == "__main__":
    main()
