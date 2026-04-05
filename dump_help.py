import subprocess
import sys

def get_help():
    try:
        # Dump help with utf-8 encoding to bypass Windows charmap errors
        with open("ns_help.txt", "w", encoding="utf-8") as f:
            subprocess.run(
                ["ns-render", "spiral", "-h"],
                stdout=f, stderr=subprocess.STDOUT, text=True, errors="ignore"
            )
        print("Help dumped to ns_help.txt")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    get_help()
