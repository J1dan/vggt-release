#!/usr/bin/env python3
"""
combine_convert_jsons.py

Combines multiple JSON files containing labeled video data and generates a combined_dataset.json file.
Extracts "instruction" and "filename" from input JSON files and converts them to caption/media_path format.

Usage:
    python combine_convert_jsons.py --files file1.json file2.json file3.json
    python combine_convert_jsons.py --folder /path/to/json/folder
"""

import argparse
import json
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Combine and convert JSON files to dataset format")
    parser.add_argument("--files", nargs="+", help="Path(s) to JSON file(s) to combine")
    parser.add_argument("--folder", type=str, help="Folder containing JSON files to combine")
    parser.add_argument("--output", type=str, default="combined_dataset.json", help="Output JSON file for combined dataset")
    
    args = parser.parse_args()
    
    # Collect JSON files
    json_files = []
    
    if args.files:
        json_files = [Path(f) for f in args.files if Path(f).exists() and f.endswith('.json')]
        if len(json_files) < len(args.files):
            print(f"Warning: Some specified files do not exist or are not JSON files")
    
    elif args.folder:
        folder = Path(args.folder)
        if folder.exists():
            json_files = list(folder.glob("*.json"))
        else:
            print(f"Error: Folder {args.folder} does not exist")
            return
    else:
        print("Error: Must provide either --files or --folder")
        return
    
    if not json_files:
        print("No JSON files found to process")
        return
    
    print(f"Found {len(json_files)} JSON files to combine")
    
    # Combine and convert data
    dataset = []
    
    for json_file in json_files:
        print(f"Loading {json_file.name}...")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Handle list of entries
                if isinstance(data, list):
                    for item in data:
                        if 'instruction' in item and 'filename' in item:
                            caption = item['instruction']
                            filename = item['filename']
                            # Set media_path as videos/filename
                            media_path = f"{filename}"
                            dataset.append({
                                "caption": caption,
                                "media_path": media_path
                            })
                        elif 'caption' in item and 'media_path' in item:
                            # Already in desired format, just append
                            dataset.append({
                                "caption": item['caption'],
                                "media_path": item['media_path']
                            })
                        else:
                            print(f"Warning: Entry in {json_file.name} missing 'instruction' or 'filename' or 'caption' or 'media_path', skipping")
                else:
                    print(f"Warning: {json_file.name} does not contain a list, skipping")
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
    
    print(f"Combined {len(dataset)} entries from {len(json_files)} files")
    
    # Write combined dataset JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Combined dataset saved to {output_path} with {len(dataset)} entries")

if __name__ == "__main__":
    main()
