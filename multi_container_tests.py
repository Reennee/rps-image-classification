#!/usr/bin/env python3
"""
Simple Multi-Container Load Testing Script
"""

import subprocess
import time
import os

def run_locust_test(port, test_name):
    """Run Locust test on specific container port"""
    print(f"🚀 Testing container on port {port}...")
    
    cmd = [
        "locust", "-f", "locustfile.py",
        "--host", f"http://localhost:{port}",
        "--headless", "-u", "20", "-r", "2", "-t", "30s",
        "--csv", f"locust_results_{port}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ Test completed for port {port}")
            return True
        else:
            print(f"❌ Test failed for port {port}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Test timed out for port {port}")
        return False
    except Exception as e:
        print(f"❌ Error testing port {port}: {e}")
        return False

def main():
    print("🏃 Multi-Container Load Testing")
    print("=" * 40)
    
    # Container ports to test
    ports = [8001, 8002, 8003]
    
    print("Make sure your containers are running on these ports!")
    print("If not, run: docker-compose up -d")
    print()
    
    # Test each container
    for port in ports:
        print(f"🔍 Testing container on port {port}...")
        success = run_locust_test(port, f"container_{port}")
        
        if success:
            print(f"📊 Results saved to locust_results_{port}_*.csv")
        else:
            print(f"⚠️  Failed to test port {port}")
        
        print("-" * 30)
        time.sleep(2)  # Small delay between tests
    
    print("🎯 All tests completed!")
    print("📁 Check the CSV files for detailed results")
    print("📖 Update your README with the actual performance numbers")

if __name__ == "__main__":
    main()
