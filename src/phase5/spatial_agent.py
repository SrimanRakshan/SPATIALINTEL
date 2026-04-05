import os
import argparse
import json
from pathlib import Path
import requests

def load_scene_graph_3d(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def format_scene_context(scene_graph: dict) -> str:
    objects = scene_graph.get("objects", [])
    if not objects:
        return "The scene is empty."
        
    context = "The following objects were detected in the 3D scene, along with their 3D coordinates (X, Y, Z):\n"
    for obj in objects:
        name = obj["name"]
        pos = obj["position"]
        obs = obj["observations"]
        rx, ry, rz = round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)
        context += f"- A {name} located at position [X: {rx}, Y: {ry}, Z: {rz}] (Aggregated from {obs} views)\n"
    return context

def query_huggingface(context: str, user_query: str):
    """ Completely Free API via Hugging Face Models - No API Key Required! """
    print("\nQuerying HuggingFace Free Inference API (Mistral-7B)...")
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    prompt = f"<s>[INST] You are an intelligent 3D spatial reasoning agent analyzing a 3D scene reconstructed from a monocular video. You are provided with a semantic scene graph containing coordinates.\n\nContext about the room:\n{context}\n\nQuestion: {user_query} [/INST]"
    
    try:
        response = requests.post(API_URL, json={"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.2}})
        if response.status_code == 200:
            result = response.json()[0]['generated_text']
            return result.split("[/INST]")[-1].strip()
        else:
            return f"HuggingFace Rate Limited / Error: {response.text}"
    except Exception as e:
        return f"Request failed: {e}"

def query_gemini(context: str, user_query: str, api_key: str):
    """ Google Gemini 1.5 - Has a 100% Free Tier (requires API Key from Google AI Studio) """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")
    
    try:
        import google.generativeai as genai
    except ImportError:
        return "Error: google-generativeai package not installed. Run `pip install google-generativeai`"
        
    if not api_key:
        return "Error: Gemini requires a free API key. Get one at: https://aistudio.google.com/"
        
    print("\nQuerying Google Gemini (Free Tier)...")
    genai.configure(api_key=api_key)
    
    try:
        # Dynamically discover the latest available flash model that supports generation
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not models:
            return "Error: No generative models available for this API key."
            
        # Target a fast flash model implicitly
        target_model = next((m for m in models if "flash" in m and "exp" not in m), models[0])
        model = genai.GenerativeModel(target_model)
        
        prompt = f"You are an intelligent 3D spatial reasoning agent analyzing a 3D scene reconstructed from a monocular video. You are provided with a semantic scene graph.\n\nContext about the room:\n{context}\n\nQuestion: {user_query}"
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

def query_openai(context: str, user_query: str, api_key: str):
    """ OpenAI - Requires paid credits """
    try:
        from openai import OpenAI
    except ImportError:
        return "Error: openai package not installed."
        
    if not api_key:
        return "Error: OpenAI requires a paid API key."
        
    print("\nQuerying OpenAI GPT-4o...")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a 3D spatial reasoning agent analyzing a 3D scene reconstructed from a monocular video. You are provided with a semantic scene graph."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_query}"}
    ]
    try:
        response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.2)
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI Error: {e}"
        
def interactive_agent(scene_graph_path: str, provider: str, api_key: str):
    scene_graph_path = Path(scene_graph_path)
    if not scene_graph_path.exists():
        print(f"Error: Missing 3D Scene Graph at {scene_graph_path}")
        exit(1)
        
    scene_graph = load_scene_graph_3d(scene_graph_path)
    context = format_scene_context(scene_graph)
    
    print("=========================================")
    print(f"🧠 Spatial Intelligence Agent ({provider.upper()}) 🧠")
    print("=========================================")
    print(f"Loaded {len(scene_graph.get('objects', []))} semantic objects into memory.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            query = input("User Query > ")
            if query.lower() in ["exit", "quit", "q"]:
                break
                
            if provider == "huggingface":
                res = query_huggingface(context, query)
            elif provider == "gemini":
                res = query_gemini(context, query, api_key)
            else:
                res = query_openai(context, query, api_key)
                
            print("\n[LLM Response]")
            print(res)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Based Spatial Reasoning over a 3D Scene Graph.")
    parser.add_argument("--scene_graph_3d", required=True, help="Path to 3D Scene Graph JSON")
    parser.add_argument("--provider", type=str, choices=["huggingface", "gemini", "openai"], default="huggingface", help="LLM Provider to use (default: huggingface)")
    parser.add_argument("--api_key", type=str, default=os.getenv("API_KEY"), help="API Key (Not needed for HuggingFace)")
    parser.add_argument("--query", type=str, help="Run a single query and exit")
    
    args = parser.parse_args()
    
    if args.query:
        scene_graph = load_scene_graph_3d(args.scene_graph_3d)
        ctx = format_scene_context(scene_graph)
        if args.provider == "huggingface":
            print(f"\n[LLM Response]\n{query_huggingface(ctx, args.query)}")
        elif args.provider == "gemini":
            print(f"\n[LLM Response]\n{query_gemini(ctx, args.query, args.api_key)}")
        else:
            print(f"\n[LLM Response]\n{query_openai(ctx, args.query, args.api_key)}")
    else:
        interactive_agent(args.scene_graph_3d, args.provider, args.api_key)
