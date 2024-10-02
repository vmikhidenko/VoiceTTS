# train.py

import argparse
import json
import logging
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.tts_dataset import CustomDataset
from src.collate_fn import custom_collate_fn
from src.model import VITSModel
from src.my_tokenizer import TTSTokenizer
from src.utils import load_json


def setup_logging(verbosity: int) -> None:
    """
    Configures the logging settings based on the verbosity level.
    Args:
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
    parser = argparse.ArgumentParser(description="Train a VITS TTS model with custom data.")

    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help="Path to the model-specific TTS configuration file."
    )

    return parser.parse_args()


def main():
    """
    Main function to execute the training process based on user-specified arguments.
    """
    args = parse_arguments()
    
    # Load configuration first to get verbosity level
    try:
        config = load_json(args.config_path)
    except Exception as e:
        print(f"Failed to load configuration file: {e}")
        sys.exit(1)
    
    # Retrieve verbosity level from config
    verbosity = config.get('logging', {}).get('verbosity', 1)
    setup_logging(verbosity)
    logger = logging.getLogger("Train")

    logger.info("Starting training process.")

    # Initialize tokenizer
    try:
        tokenizer_config = config.get('tokenizer', {})
        tokenizer = TTSTokenizer(
            characters=tokenizer_config.get('characters', "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789,!.\"'-"),
            pad=tokenizer_config.get('pad_token', "<PAD>"),
            eos=tokenizer_config.get('eos_token', "<EOS>"),
            bos=tokenizer_config.get('bos_token', "<BOS>"),
            unk=tokenizer_config.get('unk_token', "<UNK>")
        )
        logger.info(f"Tokenizer initialized with vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"Failed to initialize tokenizer: {e}")
        sys.exit(1)

    # Initialize model with tokenizer's vocab_size and audio parameters
    try:
        model_config = config.get('model', {})
        audio_config = config.get('audio', {})
        model = VITSModel(
            embedding_dim=model_config.get('embedding_dim', 80),
            hidden_size=model_config.get('hidden_size', 256),
            num_layers=model_config.get('num_layers', 2),
            vocab_size=tokenizer.vocab_size,
            n_mel_channels=model_config.get('n_mel_channels', 80),
            time_frames=model_config.get('time_frames', 5027),
            upsampling_factor=model_config.get('upsampling_factor', 8)
        )
        logger.info("Model initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)

    # Ensure vocab size matches between tokenizer and model
    if tokenizer.vocab_size != model.vocab_size:
        logger.error("Vocab size mismatch between tokenizer and model!")
        sys.exit(1)
    logger.info("Tokenizer and model vocab sizes are consistent.")

    # Initialize dataset and DataLoader
    try:
        dataset_config = config.get('audio', {})
        dataset = CustomDataset(
            metadata_path=config.get('paths', {}).get('data_path', "data/metadata.csv"),
            tokenizer=tokenizer,
            config=dataset_config,
            augment=True  # Enable data augmentation
        )
        # Read training configuration
        training_config = config.get('training', {})
        batch_size = training_config.get('batch_size', 2)
        drop_last = training_config.get('drop_last', False)
        num_workers = training_config.get('num_workers', 2)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=custom_collate_fn
        )
        logger.info(f"Dataset and DataLoader initialized. Total samples: {len(dataset)}")
        logger.info(f"Batch size: {batch_size}, Drop last: {drop_last}, Num workers: {num_workers}")
    except Exception as e:
        logger.error(f"Failed to initialize dataset or DataLoader: {e}")
        sys.exit(1)

    # Define loss function and optimizer
    try:
        loss_function_name = training_config.get('loss_function', 'MSELoss')
        if loss_function_name == 'MSELoss':
            criterion = nn.MSELoss()
        elif loss_function_name == 'HuberLoss':
            criterion = nn.SmoothL1Loss()
        elif loss_function_name == 'MAELoss':
            criterion = nn.L1Loss()
        else:
            logger.error(f"Unsupported loss function: {loss_function_name}")
            sys.exit(1)

        optimizer_name = training_config.get('optimizer', 'Adam')
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=training_config.get('learning_rate', 0.0001), weight_decay=1e-5)
        else:
            logger.error(f"Unsupported optimizer: {optimizer_name}")
            sys.exit(1)

        logger.info(f"Optimizer ({optimizer_name}) and loss function ({loss_function_name}) set. Learning rate: {training_config.get('learning_rate', 0.0001)}")
    except Exception as e:
        logger.error(f"Failed to set up optimizer or loss function: {e}")
        sys.exit(1)

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Create output directory if it doesn't exist
    output_path = config.get('paths', {}).get('output_path', "models/vits_custom_voice")
    os.makedirs(output_path, exist_ok=True)

    # Initialize TensorBoard writer
    log_dir = os.path.join(output_path, 'logs')
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logging initialized at {log_dir}")

    # Initialize validation dataset and DataLoader if validation is enabled
    try:
        validation_config = config.get('training', {}).get('early_stopping', {})
        validation_enabled = validation_config.get('enabled', False)
        if validation_enabled:
            validation_metadata_path = config.get('paths', {}).get('validation_data_path', "data/validation_metadata.csv")
            validation_dataset = CustomDataset(
                metadata_path=validation_metadata_path,
                tokenizer=tokenizer,
                config=dataset_config,
                augment=False  # Typically, do not augment validation data
            )
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                collate_fn=custom_collate_fn
            )
            logger.info(f"Validation Dataset and DataLoader initialized. Total validation samples: {len(validation_dataset)}")
    except Exception as e:
        logger.error(f"Failed to initialize validation dataset or DataLoader: {e}")
        sys.exit(1)

    # Initialize Early Stopping variables
    best_validation_loss = float('inf')
    trigger_times = 0
    patience = validation_config.get('patience', 10)

    # Initialize Learning Rate Scheduler
    scheduler_type = training_config.get('scheduler', None)
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        logger.info("Learning Rate Scheduler 'ReduceLROnPlateau' initialized.")
    elif scheduler_type == 'StepLR':
        step_size = training_config.get('step_size', 100)
        gamma = training_config.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"Learning Rate Scheduler 'StepLR' initialized with step_size={step_size}, gamma={gamma}.")
    elif scheduler_type == 'ExponentialLR':
        gamma = training_config.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        logger.info(f"Learning Rate Scheduler 'ExponentialLR' initialized with gamma={gamma}.")
    else:
        scheduler = None
        logger.info("No Learning Rate Scheduler selected.")

    avg_validation_loss = float('inf')

    # Start training loop
    try:
        epochs = training_config.get('epochs', 1000)
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0

            # Check if dataloader has batches
            if len(dataloader) == 0:
                logger.warning(f"No batches to process for epoch {epoch}. Skipping epoch.")
                continue

            for batch_idx, batch in enumerate(dataloader, 1):
                tokens = batch['tokens'].to(device)        # Shape: (batch_size, sequence_length)
                targets = batch['targets'].to(device)      # Shape: (batch_size, n_mel_channels, time_frames)

                optimizer.zero_grad()
                outputs = model(tokens)                    # Expected shape: (batch_size, n_mel_channels, time_frames)

                # Check if outputs have the expected shape
                if outputs.shape != targets.shape:
                    logger.error(f"Shape mismatch: outputs {outputs.shape} vs targets {targets.shape}")
                    logger.error("Skipping this batch due to shape mismatch.")
                    continue

                loss = criterion(outputs, targets)
                loss.backward()                             # Compute gradients
                optimizer.step()                            # Update the model's parameters

                epoch_loss += loss.item()

                # Verbose logging for debugging
                if config.get('logging', {}).get('verbosity', 1) >= 2:
                    logger.debug(f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

            # Calculate average loss
            if len(dataloader) > 0:
                avg_loss = epoch_loss / len(dataloader)
            else:
                avg_loss = 0.0  # Avoid division by zero
                logger.warning(f"No batches were processed in epoch {epoch}. Average loss set to 0.0.")

            logger.info(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}")

            # Log the average loss to TensorBoard
            writer.add_scalar('Loss/Train', avg_loss, epoch)

            # Learning Rate Scheduler Step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if validation_enabled:
                        scheduler.step(avg_validation_loss)
                    else:
                        scheduler.step(avg_loss)
                else:
                    scheduler.step()
                    logger.debug(f"Learning rate updated to {optimizer.param_groups[0]['lr']}")

            # Validation and Early Stopping
            if validation_enabled:
                model.eval()
                validation_loss = 0.0
                with torch.no_grad():
                    for val_batch in validation_dataloader:
                        val_tokens = val_batch['tokens'].to(device)
                        val_targets = val_batch['targets'].to(device)
                        val_outputs = model(val_tokens)

                        if val_outputs.shape != val_targets.shape:
                            logger.error(f"Validation Shape mismatch: outputs {val_outputs.shape} vs targets {val_targets.shape}")
                            continue

                        val_loss = criterion(val_outputs, val_targets)
                        validation_loss += val_loss.item()

                avg_validation_loss = validation_loss / len(validation_dataloader)
                logger.info(f"Epoch [{epoch}/{epochs}] Validation Loss: {avg_validation_loss:.4f}")
                writer.add_scalar('Loss/Validation', avg_validation_loss, epoch)

                # Check for improvement
                if avg_validation_loss < best_validation_loss:
                    best_validation_loss = avg_validation_loss
                    trigger_times = 0
                    # Save the best model
                    best_checkpoint_path = os.path.join(output_path, "best_model.pth")
                    torch.save(model.state_dict(), best_checkpoint_path)
                    logger.info(f"Saved best model checkpoint at {best_checkpoint_path}")
                else:
                    trigger_times += 1
                    logger.info(f"No improvement in validation loss for {trigger_times} epochs.")
                    if trigger_times >= patience:
                        logger.info("Early stopping triggered.")
                        break
                # Switch back to training mode
                model.train()

            # Save model checkpoint every 5 epochs or on the last epoch
            if epoch % 5 == 0 or epoch == epochs:
                checkpoint_path = os.path.join(output_path, f"model_epoch_{epoch}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved model checkpoint at {checkpoint_path}")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        writer.close()  # Ensure the writer is closed even if an error occurs
        sys.exit(1)

    logger.info("Training completed successfully.")

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
