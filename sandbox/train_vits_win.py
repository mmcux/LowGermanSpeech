"""
    original from
    https://github.com/thorstenMueller/Thorsten-Voice
"""
import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from sandbox.config import vits_config
import dotenv

dotenv.load_dotenv()


def main():
    data_wav_path = os.getenv("DATA_WAV_PATH", None)
    output_path = os.getenv("DATA_WAV_PATH_OUT", None)
    if not os.path.exists(data_wav_path):
        raise Exception("Error: data folder not existent")
    if not os.path.exists(output_path):
        raise Exception("Error: out folder not existent")
    meta_file_train = "metadata_train.csv"
    meta_file_train = "metadata_dev.csv"
    meta_file_train = "metadata_test.csv"

    # output_path = os.path.dirname(os.path.abspath(__file__))
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train=meta_file_train,
        path=f"{data_wav_path}"
    )
    audio_config = VitsAudioConfig(
        sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    test_sentences = [
        "Es hat mich viel Zeit gekostet ein Stimme zu entwickeln, jetzt wo ich sie habe werde ich nicht mehr schweigen.",
        "Sei eine Stimme, kein Echo.",
        "Es tut mir Leid David. Das kann ich leider nicht machen.",
        "Dieser Kuchen ist großartig. Er ist so lecker und feucht.",
        "Vor dem 22. November 1963.",
    ]

    config = VitsConfig(**dict(
        **vits_config.copy(),
        audio=audio_config,
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        output_path=output_path,
        datasets=[dataset_config],
        test_sentences=test_sentences,
    ))
    config.run_name = "vits_some"
    config.epochs = 100

    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    print("Fertig!")


from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()  # needed for Windows
    main()
