import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ No API key found in .env file")
    exit(1)

print(f"🔑 API Key found: {api_key[:10]}...")
print("-" * 60)

try:
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # List all available models
    print("📋 Listing all available models:\n")
    models = genai.list_models()
    
    embedding_models = []
    generation_models = []
    
    for model in models:
        print(f"Model: {model.name}")
        print(f"  Display name: {model.display_name}")
        print(f"  Supported methods: {model.supported_generation_methods}")
        
        # Check if it supports embeddings
        if 'embedContent' in model.supported_generation_methods:
            embedding_models.append(model.name)
            print(f"  ✅ SUPPORTS EMBEDDINGS")
        
        # Check if it supports content generation
        if 'generateContent' in model.supported_generation_methods:
            generation_models.append(model.name)
        
        print()
    
    print("=" * 60)
    print("\n✅ EMBEDDING MODELS FOUND:")
    for i, model in enumerate(embedding_models, 1):
        print(f"  {i}. {model}")
    
    print("\n✅ GENERATION MODELS FOUND:")
    for i, model in enumerate(generation_models[:10], 1):  # Show first 10
        print(f"  {i}. {model}")
    
    # Test each embedding model
    print("\n" + "=" * 60)
    print("🧪 TESTING EACH EMBEDDING MODEL:\n")
    
    test_text = "This is a test sentence for embedding."
    
    for emb_model in embedding_models:
        try:
            print(f"Testing {emb_model}...", end=" ")
            result = genai.embed_content(
                model=emb_model,
                content=test_text,
                task_type="retrieval_document",
            )
            if result and 'embedding' in result:
                embedding = result['embedding']
                print(f"✅ SUCCESS - Embedding dimension: {len(embedding)}")
            else:
                print("❌ Failed - No embedding in response")
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)[:100]}")
    
    # Also check for specific models by name
    print("\n" + "=" * 60)
    print("🔍 CHECKING SPECIFIC MODEL NAMES:\n")
    
    specific_models = [
        "models/embedding-001",
        "models/text-embedding-004",
        "models/embedding-gecko-001",
        "models/embedding-gecko-002",
        "models/embedding-gecko-003",
        "models/embedding-002",
        "models/embedding-003",
        "models/embedding-multilingual-001",
        "models/embedding-multilingual-002"
    ]
    
    for model_name in specific_models:
        try:
            # First check if model exists by trying to get it
            model_info = genai.get_model(model_name)
            print(f"✅ {model_name} - EXISTS")
            
            # Try to use it for embedding
            try:
                result = genai.embed_content(
                    model=model_name,
                    content=test_text,
                    task_type="retrieval_document",
                )
                if result and 'embedding' in result:
                    print(f"   └─ ✓ Working - Dim: {len(result['embedding'])}")
                else:
                    print(f"   └─ ✗ Failed - No embedding")
            except Exception as e:
                print(f"   └─ ✗ Usage failed: {str(e)[:100]}")
                
        except Exception as e:
            print(f"❌ {model_name} - NOT AVAILABLE")
    
    print("\n" + "=" * 60)
    print("\n✅ RECOMMENDATION:")
    if embedding_models:
        print(f"Use: {embedding_models[0]} (first working embedding model)")
    else:
        print("No embedding models found. You may need to:")
        print("1. Enable the Embeddings API in Google Cloud Console")
        print("2. Use a different API key with embeddings enabled")
        print("3. The app will use fallback mode (mock embeddings)")

except Exception as e:
    print(f"❌ Error: {e}")