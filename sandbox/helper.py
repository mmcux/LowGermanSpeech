import os
import re

from pydub import AudioSegment


def clean_text(text):
    # Ersetze Whitespaces durch Unterstriche
    text = re.sub(r'\s', '_', text)
    text = text.replace("ß", "ss")

    # Behalte nur a-z, A-Z, Umlaute und Unterstriche
    text = re.sub(r'[^a-zA-ZäöüÄÖÜ_]', '', text)

    # Konvertiere alles zu Kleinbuchstaben
    text = text.lower()

    return text


def convert_stereo_to_mono(input_dir, output_dir):
    # Erstellen Sie das Ausgabeverzeichnis, falls es nicht existiert
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Durchlaufen Sie alle WAV-Dateien im Eingabeverzeichnis
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{filename}")

            # Laden Sie die Audiodatei
            audio = AudioSegment.from_wav(input_path)

            # Konvertieren Sie zu Mono
            mono_audio = audio.set_channels(1)

            # Exportieren Sie die Mono-Datei
            mono_audio.export(output_path, format="wav")

            # print(f"Konvertiert: {filename}")
