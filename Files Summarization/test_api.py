import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY", "").strip()

print(f"Current directory: {os.getcwd()}")
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"API Key length: {len(api_key)} characters")
print(f"API Key first 10 chars: {api_key[:10]}...")

if not api_key:
    print("\n❌ No API key found!")
    print("Please create a .env file with: GEMINI_API_KEY=your_key_here")
else:
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        print("\n✅ Successfully connected to Gemini API!")
        print("\nAvailable models:")
        for model in models:
            if "gemini" in model.name:
                print(f"  - {model.name}")
    except Exception as e:
        print(f"\n❌ Failed to connect: {e}")