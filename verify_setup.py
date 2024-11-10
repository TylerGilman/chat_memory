import subprocess
import os
import sys

def check_command(command, description):
    print(f"Checking {description}...", end=" ")
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True)
        print("✅")
        return True
    except subprocess.CalledProcessError:
        print("❌")
        return False

def verify_setup():
    print("\n=== System Requirements ===")
    
    checks = [
        ("docker --version", "Docker installation"),
        ("nvidia-smi", "NVIDIA drivers"),
        ("docker info | grep -i nvidia", "NVIDIA Docker runtime")
    ]
    
    failed = False
    for command, description in checks:
        if not check_command(command, description):
            failed = True
    
    print("\n=== Directory Structure ===")
    required_files = [
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "run.sh",
        "training/train.py",
        "training/preprocessing/transformed_conversations/training_data.jsonl"
    ]
    
    for file_path in required_files:
        print(f"Checking {file_path}...", end=" ")
        if os.path.exists(file_path):
            print("✅")
        else:
            print("❌")
            failed = True
    
    print("\n=== Verification Result ===")
    if failed:
        print("❌ Some checks failed. Please install missing requirements or fix file structure.")
        return False
    else:
        print("✅ All checks passed! You can now run: ./run.sh")
        return True

if __name__ == "__main__":
    if verify_setup():
        sys.exit(0)
    else:
        sys.exit(1)
