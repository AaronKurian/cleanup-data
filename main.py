#!/usr/bin/env python3
"""
Ollama Client Script
A simple Python script to connect and interact with your local Ollama instance.
"""

import json
import os
import requests
from typing import List, Dict, Optional

# Default read timeout for /api/generate (seconds). CPU inference can exceed 30s.
# Override with env: OLLAMA_GENERATE_TIMEOUT=1200
_DEFAULT_READ_TIMEOUT = float(os.environ.get("OLLAMA_GENERATE_TIMEOUT", "600"))

# Preferred model for normalize_ingredients.py (quantized Qwen 7B — good for ~16GB RAM / CPU)
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.

        Args:
            base_url (str): The base URL for the Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url.rstrip("/")

    def check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Connection failed: {e}")
            return False

    def list_models(self) -> List[Dict]:
        """
        Get a list of available models.

        Returns:
            List[Dict]: List of available models
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: {e}")
            return []

    def generate_response(
        self, model: str, prompt: str, stream: bool = False
    ) -> Optional[str]:
        """
        Generate a response from the specified model.

        Args:
            model (str): The name of the model to use
            prompt (str): The input prompt
            stream (bool): Whether to stream the response

        Returns:
            Optional[str]: The generated response or None if error
        """
        try:
            payload = {"model": model, "prompt": prompt, "stream": stream}

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(30, _DEFAULT_READ_TIMEOUT),
            )
            response.raise_for_status()

            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            full_response += data["response"]
                            print(data["response"], end="", flush=True)
                        if data.get("done"):
                            break
                print()  # New line after streaming
                return full_response
            else:
                # Handle non-streaming response
                return response.json().get("response", "")

        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            return None

    def chat(self, model: str, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Have a chat conversation with the model.

        Args:
            model (str): The name of the model to use
            messages (List[Dict]): List of messages in chat format

        Returns:
            Optional[str]: The model's response or None if error
        """
        try:
            payload = {"model": model, "messages": messages, "stream": False}

            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=30
            )
            response.raise_for_status()

            return response.json().get("message", {}).get("content", "")

        except requests.exceptions.RequestException as e:
            print(f"Error in chat: {e}")
            return None


def main():
    """Main function to demonstrate the Ollama client."""
    print("🤖 Ollama Client Script")
    print("=" * 50)

    # Initialize client
    client = OllamaClient()

    # Check connection
    print("Checking connection to Ollama...")
    if not client.check_connection():
        print(
            "❌ Failed to connect to Ollama. Make sure Ollama is running on http://localhost:11434"
        )
        return

    print("✅ Successfully connected to Ollama!")

    # List available models
    print("\n📋 Available Models:")
    models = client.list_models()

    if not models:
        print(
            f"No models found. Install the default recipe model with:\n"
            f"  ollama pull {DEFAULT_OLLAMA_MODEL}\n"
            f"(or any other model, then use it from normalize_ingredients.py --model …)"
        )
        return

    for i, model in enumerate(models, 1):
        name = model.get("name", "Unknown")
        size = model.get("size", 0)
        size_gb = size / (1024**3) if size else 0
        modified = model.get("modified_at", "Unknown")
        print(f"{i}. {name} (Size: {size_gb:.2f} GB, Modified: {modified[:19]})")

    # Interactive mode
    print("\n🔄 Interactive Mode")
    print("Type 'quit' to exit, 'models' to list models again")

    # Use first available model as default
    default_model = models[0]["name"]
    print(f"Using default model: {default_model}")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye! 👋")
            break
        elif user_input.lower() == "models":
            for i, model in enumerate(models, 1):
                print(f"{i}. {model.get('name', 'Unknown')}")
            continue
        elif not user_input:
            continue

        print("🤖 Ollama:", end=" ")
        response = client.generate_response(default_model, user_input, stream=True)

        if response is None:
            print("❌ Failed to get response from model.")


if __name__ == "__main__":
    main()