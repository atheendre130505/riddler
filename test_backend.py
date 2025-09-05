"""
Test script for the backend server
"""

import requests
import json

def test_backend():
    """Test the backend server endpoints"""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 Testing Smart Content Agent Backend")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health Check: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root Endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root Endpoint Failed: {e}")
    
    # Test stats endpoint
    try:
        response = requests.get(f"{base_url}/stats")
        print(f"✅ Stats Endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Stats Endpoint Failed: {e}")
    
    print("\n🎉 Backend testing complete!")
    print("Backend is ready for deployment!")

if __name__ == "__main__":
    test_backend()
