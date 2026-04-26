"""Test HF API endpoints and token validity."""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()

import httpx

token = os.environ.get("HF_TOKEN")
print(f"Token: {token}" if token else "NO TOKEN FOUND!")

# Test 1: Token validity
print("\n--- Token Check ---")
resp = httpx.get("https://huggingface.co/api/whoami-v2", 
                 headers={"Authorization": f"Bearer {token}"}, timeout=15)
print(f"Token valid: {resp.status_code == 200}")
if resp.status_code == 200:
    data = resp.json()
    print(f"User: {data.get('name', 'unknown')}")
else:
    print(f"Error: {resp.status_code} - {resp.text[:200]}")

# Test 2: Try both endpoints with multiple models
models = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-2-1",
]

endpoints = {
    "Router": "https://router.huggingface.co/hf-inference/models",
    "Legacy": "https://api-inference.huggingface.co/models",
}

headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
payload = {"inputs": "a red t-shirt on white background"}

for ep_name, base_url in endpoints.items():
    print(f"\n--- {ep_name} Endpoint: {base_url} ---")
    for m in models:
        url = f"{base_url}/{m}"
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=60)
            ct = resp.headers.get("content-type", "")
            if resp.status_code == 200 and ("image" in ct or len(resp.content) > 1000):
                print(f"  OK: {m} -> image ({len(resp.content)} bytes)")
            elif resp.status_code == 503:
                est = "unknown"
                try:
                    est = resp.json().get("estimated_time", "?")
                except:
                    pass
                print(f"  LOADING: {m} -> 503 (est: {est}s)")
            elif resp.status_code == 401:
                print(f"  AUTH FAIL: {m} -> 401 (bad token?)")
            else:
                text = resp.text[:150].replace('\n', ' ')
                print(f"  FAIL: {m} -> {resp.status_code}: {text}")
        except Exception as e:
            print(f"  ERROR: {m} -> {e}")
