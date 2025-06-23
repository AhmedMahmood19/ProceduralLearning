import os
import subprocess
import requests

annotation_dir = "/workspace/Egoprocel/annotations/EPIC-Tents"
output_dir = "/workspace/Egoprocel/videos"
os.makedirs(output_dir, exist_ok=True)

for csvfile in os.listdir(annotation_dir):
    if not csvfile.endswith(".csv"):
        continue  # Skip non-csv files

    base_url = "https://data.bris.ac.uk/datasets/2ite3tu1u53n42hjfh3886sa86/data/"
    name_part = csvfile.split('.egoprocel', 1)[0]
    subfolder = csvfile.split('.', 1)[0]
    link = f"{base_url}{subfolder}/{name_part}.MP4"

    # HEAD request to verify it's a valid MP4
    try:
        head = requests.head(link, allow_redirects=True, timeout=10)
        content_type = head.headers.get('Content-Type', '')
        if 'video/mp4' in content_type:
            print(f"{link} leads to a valid MP4. Downloading...")
            subprocess.run(['wget', '-c', link, '-P', output_dir])
        else:
            print(f"Skipped (Not an MP4): {link} — Content-Type: {content_type}")
    except requests.RequestException as e:
        print(f"Failed to check URL: {link} — Error: {e}")
