"""
validate_config.py

This script validates the config.json file against the config_schema.json schema.
"""

import json
import sys
import logging
import os
from jsonschema import validate, ValidationError

def setup_logging():
    """
    Configures the logging settings for the validation script.
    Logs are output to the console with timestamps and severity levels.
    """
    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_json(file_path):
    """
    Loads and parses a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The parsed JSON content.

    Raises:
        SystemExit: Exits the program if the file cannot be read or parsed.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            logging.debug(f"Loading JSON file from {file_path}")
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in file '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred while loading '{file_path}': {e}")
        sys.exit(1)

def validate_config(config, schema):
    """
    Validates a configuration dictionary against a JSON schema.

    Args:
        config (dict): The configuration data to validate.
        schema (dict): The JSON schema to validate against.

    Raises:
        SystemExit: Exits the program if validation fails.
    """
    try:
        validate(instance=config, schema=schema)
        logging.info("Configuration is valid.")
    except ValidationError as ve:
        logging.error(f"Configuration validation error: {ve.message}")
        logging.debug(f"Validation error details: {ve}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during validation: {e}")
        sys.exit(1)

def main():
    """
    The main function orchestrates the validation process.
    It parses command-line arguments, loads JSON files, and performs validation.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Validate config.json against config_schema.json.")
    parser.add_argument('--config', type=str, required=True, help="Path to config.json")
    parser.add_argument('--schema', type=str, required=True, help="Path to config_schema.json")
    args = parser.parse_args()

    setup_logging()

    logging.info(f"Starting validation: Config='{args.config}', Schema='{args.schema}'")

    config = load_json(args.config)
    schema = load_json(args.schema)

    validate_config(config, schema)

    logging.info("Validation completed successfully.")

if __name__ == "__main__":
    main()
