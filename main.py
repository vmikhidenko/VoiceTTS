# main.py

import argparse
import logging
import sys
from datetime import datetime
import os
import torch
from src.tts_generator import TTSGenerator
from src.model import VITSModel  # Import the VITS model
from src.my_tokenizer import TTSTokenizer  # Import your tokenizer
from src.utils import get_output_path, load_json  # Ensure load_json is defined in utils.py

# Import additional libraries for audio processing
import librosa
import soundfile as sf

def setup_logging(verbosity: int) -> None:
    """
    Configures the logging settings based on the verbosity level. Args:
        verbosity (int): Logging verbosity level. 0 for WARNING, 1 for INFO, 2 for DEBUG.
    """
    level = logging.WARNING  # Default level
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments provided by the user.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate speech from text using a custom-trained voice model."
    )

    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help="Path to the model-specific TTS configuration file."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="Path to the trained voice model file."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="output",
        help="Directory to save the generated audio files."
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Increase output verbosity: 0=WARNING, 1=INFO, 2=DEBUG."
    )
    parser.add_argument(
        '--output_filename',
        type=str,
        default=None,
        help="Filename for the generated audio. If not provided, a timestamped filename will be used."
    )

    return parser.parse_args()

def generate_unique_filename(output_dir: str, extension: str = ".wav") -> str:
    """
    Generates a unique filename based on the current timestamp.
    Args:
        output_dir (str): Directory where the file will be saved.
        extension (str): File extension.
    Returns:
        str: Unique filename with the specified extension.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output_speech_{timestamp}{extension}"

def main():
    """
    Main function to execute the speech generation process based on user-specified arguments.
    """
    args = parse_arguments()
    setup_logging(args.verbosity)
    logger = logging.getLogger("main.py")

    logger.info("Starting Text-to-Speech generation process.")

    # Load model-specific configuration
    try:
        model_config = load_json(args.config_path)
        logger.debug(f"Model-specific configuration loaded: {model_config}")
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        sys.exit(1)

    # Load global configuration
    try:
        global_config = load_json("configs/default_config.json")
        logger.debug(f"Global configuration loaded: {global_config}")
    except Exception as e:
        logger.error(f"Failed to load global configuration file: {e}")
        sys.exit(1)

    # Merge configurations
    merged_config = {**global_config, **model_config}
    logger.debug(f"Merged configuration: {merged_config}")

    # Initialize tokenizer with <UNK> token
    try:
        tokenizer = TTSTokenizer(
            characters=merged_config.get('characters', "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789,!.\"'-"),
            pad=merged_config.get('pad_token', "<PAD>"),
            eos=merged_config.get('eos_token', "<EOS>"),
            bos=merged_config.get('bos_token', "<BOS>"),
            unk=merged_config.get('unk_token', "<UNK>")
        )
        logger.info(f"Tokenizer initialized with vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"Failed to initialize tokenizer: {e}")
        sys.exit(1)

    # Initialize model
    try:
        model = VITSModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=merged_config.get('model', {}).get('embedding_dim', 80),
            hidden_size=merged_config.get('model', {}).get('hidden_size', 256),
            num_layers=merged_config.get('model', {}).get('num_layers', 2)
        )
        logger.info("Model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)

    # Ensure vocab size matches between tokenizer and model
    if tokenizer.vocab_size != model.embedding.num_embeddings:
        logger.error("Vocab size mismatch between tokenizer and model!")
        sys.exit(1)
    logger.info("Tokenizer and model vocab sizes are consistent.")

    # Initialize generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = TTSGenerator(model, tokenizer, device)

    # Load model weights
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        sys.exit(1)

    # Ensure output directory exists
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.debug(f"Output directory verified/created at: {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory '{args.output_dir}': {e}")
        sys.exit(1)

    while True:
        try:
            text = input("Enter the text to convert to speech (or 'q' to quit): ").strip()
            if text.lower() == 'q':
                logger.info("Exiting the program as per user request.")
                break

            if not text:
                logger.warning("No text entered. Please provide valid text.")
                continue

            # Determine output filename
            if args.output_filename:
                # Ensure the output filename has a .wav extension
                if not args.output_filename.lower().endswith('.wav'):
                    args.output_filename += '.wav'
                output_file = get_output_path(args.output_dir, args.output_filename)
            else:
                unique_filename = generate_unique_filename(args.output_dir, extension=".wav")
                output_file = get_output_path(args.output_dir, unique_filename)

            # Generate speech
            try:
                mel_spectrogram = generator.synthesize(text)  # Use the synthesize method

                # Convert mel_spectrogram to audio waveform
                mel_spectrogram = mel_spectrogram.cpu().numpy()  # Convert to NumPy array
                audio_waveform = librosa.feature.inverse.mel_to_audio(
                    mel_spectrogram,
                    sr=merged_config['audio']['sample_rate'],
                    n_fft=merged_config['audio']['filter_length'],
                    hop_length=merged_config['audio']['hop_length'],
                    win_length=merged_config['audio']['win_length'],
                    fmin=merged_config['audio']['mel_fmin'],
                    fmax=merged_config['audio']['mel_fmax']
                )

                # Save the audio waveform to a .wav file
                sf.write(output_file, audio_waveform, merged_config['audio']['sample_rate'])
                logger.info(f"Speech successfully generated and saved to '{output_file}'.")

            except Exception as e:
                logger.error(f"Failed to generate speech: {e}")

        except KeyboardInterrupt:
            logger.info("\nProgram interrupted by user. Exiting gracefully.")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()
