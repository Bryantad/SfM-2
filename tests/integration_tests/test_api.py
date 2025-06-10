"""
SFM-2 Open Source Release
This file contains the public architecture and methodology.
For production deployment, additional private components are required.
"""

#!/usr/bin/env python3
"""
Test script for the SonaCore AI API with OpenAI fallback
"""
import requests
import json

def test_api():
    base_url = "http://localhost:8000"
    
    print("🚀 Testing SonaCore AI API...")
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        health_data = response.json()
        print(f"Health data: {json.dumps(health_data, indent=2)}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Simple inference without OpenAI key (should show error message)
    print("\n2. Testing inference without OpenAI key...")
    test_request = {
        "prompt": "Create a simple Sona function to add two numbers",
        "prompt_type": "sona",
        "complexity": "simple"
    }
    
    try:
        response = requests.post(f"{base_url}/inference", json=test_request)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
    
    # Test 3: Natural language prompt
    print("\n3. Testing natural language prompt...")
    test_request = {
        "prompt": "Write a Python function to calculate factorial",
        "prompt_type": "natural",
        "complexity": "simple"
    }
    
    try:
        response = requests.post(f"{base_url}/inference", json=test_request)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"❌ Natural language test failed: {e}")
    
    print("\n✅ API tests completed!")
    print("\n💡 To enable OpenAI fallback:")
    print("   Set environment variable: OPENAI_API_KEY=your_key_here")
    print("   Then restart the API server")

if __name__ == "__main__":
    test_api()
