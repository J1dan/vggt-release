#!/usr/bin/env python3
"""
analyze_dataset_statistics.py

Analyzes dataset JSON files and reports statistics on motion primitives distribution.

Usage:
    python analyze_dataset_statistics.py --input dataset.json
    python analyze_dataset_statistics.py --input dataset.json --detailed
"""

import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

def extract_motion_primitives(filename):
    """
    Extract motion primitives from filename.
    Example: IMG_3685_resized_480x256_1248_1369_MOVE_FORWARD_TURN_LEFT.avi -> ['MOVE_FORWARD', 'TURN_LEFT']

    Rules:
    - Preserve order as tokens appear in the filename.
    - Prefer longest/more specific tokens (e.g., 'TURN_LEFT' over 'TURN').
    - Recognize both old tokens (TURN_*) and new tokens (PAN/DOLLY variants).
    """
    # Remove extension and split by underscores (keep order)
    name_without_ext = filename.rsplit('.', 1)[0]
    parts = name_without_ext.split('_')

    # Only consider parts that are uppercase and don't contain digits (skip IMG and numeric parts)
    caps_parts = [p for p in parts if p.isupper() and p != 'IMG' and not any(c.isdigit() for c in p)]

    # Reconstruct a single primitive string (preserves order and underscores between tokens)
    # e.g. ['MOVE', 'FORWARD', 'TURN', 'LEFT'] -> "MOVE_FORWARD_TURN_LEFT"
    full_primitive_str = '_'.join(caps_parts)

    # Known tokens (longer first to prefer multi-word tokens)
    tokens = [
        'TURN_LEFT', 'TURN_RIGHT', 'MOVE_FORWARD', 'MOVE_BACKWARD',
        'PAN_LEFT', 'PAN_RIGHT', 'DOLLY_FORWARD', 'DOLLY_BACKWARD',
        'PAN', 'DOLLY', 'TURN', 'MOVE'
    ]

    # Parse left-to-right, greedily matching the longest token at each position
    found = []
    i = 0
    s = full_primitive_str
    s_len = len(s)
    while i < s_len:
        # Skip underscores
        if s[i] == '_':
            i += 1
            continue
        matched = False
        for t in sorted(tokens, key=lambda x: -len(x)):
            if s.startswith(t, i):
                found.append(t)
                i += len(t)
                matched = True
                break
        if not matched:
            # No known token matches; advance until next underscore or end
            while i < s_len and s[i] != '_':
                i += 1

    return found

def get_color_for_primitive(primitive_name):
    """
    Assign colors based on primitive type to group similar motions.
    """
    name = primitive_name.upper()
    
    # Left turns (Greenish)
    if 'LEFT' in name:
        if name == 'TURN_LEFT':
            return '#2ca02c'  # Strong Green
        elif 'MOVE_FORWARD' in name:
            if name.startswith('MOVE_FORWARD'):
                return '#98df8a'  # Light Green
            else:
                return '#74c476'  # Medium Green
        return '#a1d99b'  # Fallback Green
        
    # Right turns (Orange-ish)
    elif 'RIGHT' in name:
        if name == 'TURN_RIGHT':
            return '#ff7f0e'  # Strong Orange
        elif 'MOVE_FORWARD' in name:
            if name.startswith('MOVE_FORWARD'):
                return '#ffbb78'  # Light Orange
            else:
                return '#fd8d3c'  # Medium Orange
        return '#fdae6b'  # Fallback Orange
        
    # Forward (Blueish)
    elif 'FORWARD' in name:
        return '#1f77b4'  # Strong Blue
        
    return '#c7c7c7'  # Gray for others

def get_sort_key(item):
    """
    Helper to sort primitives by semantic group.
    Order: Forward -> Left -> Right (with Right reversed)
    """
    name = item[0].upper()
    count = item[1]
    
    # Primary sort key: Group
    if 'LEFT' in name:
        group = 1
        sort_count = -count  # Descending for left
    elif 'RIGHT' in name:
        group = 2
        sort_count = count   # Ascending for right (reverse order)
    else:
        group = 0 # Forward first
        sort_count = -count  # Descending for forward
        
    return (group, sort_count)

def plot_statistics(combination_counter, output_file, dataset_name):
    """
    Plot a pie chart of the primitive combination counts.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting. Please install it with 'pip install matplotlib'")
        return

    # Set font properties
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16  # Default font size for labels and percentages

    labels = []
    sizes = []
    colors = []
    
    # Sort by semantic group instead of just count
    # Convert counter to list of items and sort
    sorted_items = sorted(combination_counter.items(), key=get_sort_key)
    
    for combination, count in sorted_items:
        # Replace underscores with spaces/newlines for better readability in chart
        readable_label = combination.replace('_', '\n')
        labels.append(f"{readable_label}\n({count})")
        sizes.append(count)
        colors.append(get_color_for_primitive(combination))
        
    plt.figure(figsize=(16, 12))
    
    # Calculate startangle to center move_forward at the top
    # move_forward has 104 entries out of 640 total
    move_forward_count = combination_counter.get('MOVE_FORWARD', 0)
    total_count = sum(combination_counter.values())
    move_forward_angle = (move_forward_count / total_count) * 360
    startangle = 90 - (move_forward_angle / 2)  # Center at top (90 degrees)
    
    # Create pie chart
    wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                      startangle=startangle, colors=colors, pctdistance=0.5)
    
    # Make percentage text bold and adjust size if needed
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_weight('bold')
        autotext.set_size(16)

    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    # Title in upper left with smaller font
    plt.title(f"Motion Primitive Distribution of {dataset_name} dataset", 
              loc='left', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file)
    print("Done.")

def analyze_dataset(input_json, detailed=False, save_plot=None):
    """
    Analyze dataset and report motion primitive statistics.
    """
    print(f"Loading dataset from {input_json}...")
    with open(input_json, 'r') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} entries\n")
    
    # Track statistics
    primitive_counter = Counter()
    combination_counter = Counter()
    filename_to_primitives = defaultdict(list)
    
    # Process each entry
    for entry in dataset:
        media_path = entry['media_path']
        filename = media_path.split('/')[-1]
        
        # Extract primitives
        primitives = extract_motion_primitives(filename)
        
        if primitives:
            # Count individual primitives
            for primitive in primitives:
                primitive_counter[primitive] += 1
            
            # Count combinations
            combination = '_'.join(primitives)
            combination_counter[combination] += 1
            
            # Track for detailed view
            if detailed:
                filename_to_primitives[filename].append(entry['caption'])
    
    # Compute unknown entries (no primitives)
    unknown_entries = [entry for entry in dataset if not extract_motion_primitives(entry['media_path'].split('/')[-1])]
    unknown_filenames = sorted({e['media_path'].split('/')[-1] for e in unknown_entries})

    # Print statistics
    print("=" * 70)
    print("MOTION PRIMITIVE STATISTICS")
    print("=" * 70)

    # Report unknowns
    if unknown_filenames:
        print(f"\nNOTE: {len(unknown_filenames)} unique video(s) have NO detected motion primitives (e.g., REJECT_NO_MOTION or non-standard filenames).")
        if detailed:
            print("These videos are:")
            for fn in unknown_filenames[:50]:
                print(f"  - {fn}")
            if len(unknown_filenames) > 50:
                print(f"  ... and {len(unknown_filenames)-50} more")
        print("")
    
    print("\n1. Individual Primitive Counts:")
    print("-" * 70)
    total_primitives = sum(primitive_counter.values())
    for primitive, count in primitive_counter.most_common():
        percentage = (count / total_primitives) * 100
        print(f"  {primitive:<20} {count:>6} ({percentage:>5.1f}%)")
    
    print(f"\n  {'TOTAL':<20} {total_primitives:>6} (100.0%)")
    
    print("\n2. Primitive Combination Counts:")
    print("-" * 70)
    total_combinations = sum(combination_counter.values())
    for combination, count in combination_counter.most_common():
        percentage = (count / total_combinations) * 100
        print(f"  {combination:<40} {count:>6} ({percentage:>5.1f}%)")
    
    print(f"\n  {'TOTAL':<40} {total_combinations:>6} (100.0%)")
    
    # Detailed breakdown
    if detailed:
        print("\n3. Detailed Breakdown by Unique Video:")
        print("-" * 70)
        unique_videos = {}
        for entry in dataset:
            media_path = entry['media_path']
            filename = media_path.split('/')[-1]
            if filename not in unique_videos:
                unique_videos[filename] = {
                    'primitives': extract_motion_primitives(filename),
                    'captions': []
                }
            unique_videos[filename]['captions'].append(entry['caption'])
        
        # Group by primitive combination
        combo_groups = defaultdict(list)
        for filename, info in unique_videos.items():
            combo = '_'.join(info['primitives']) if info['primitives'] else 'UNKNOWN'
            combo_groups[combo].append(filename)
        
        for combo, filenames in sorted(combo_groups.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n  {combo} ({len(filenames)} unique videos):")
            for filename in sorted(filenames)[:5]:  # Show first 5
                print(f"    - {filename}")
            if len(filenames) > 5:
                print(f"    ... and {len(filenames) - 5} more")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total entries:           {len(dataset)}")
    print(f"Unique videos:           {len(set(entry['media_path'] for entry in dataset))}")
    print(f"Primitive types:         {len(primitive_counter)}")
    print(f"Combination types:       {len(combination_counter)}")
    print("=" * 70 + "\n")

    if save_plot:
        # Extract dataset name from path (parent folder name)
        dataset_name = Path(input_json).parent.name
        plot_statistics(combination_counter, save_plot, dataset_name)

def main():
    parser = argparse.ArgumentParser(description="Analyze motion primitive statistics in dataset")
    parser.add_argument("--input", type=str, required=True, help="Input dataset JSON file")
    parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown")
    parser.add_argument("--save_plot", type=str, help="Path to save the pie chart visualization (e.g., plot.png)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File {args.input} does not exist")
        return
    
    analyze_dataset(args.input, detailed=args.detailed, save_plot=args.save_plot)

if __name__ == "__main__":
    main()
