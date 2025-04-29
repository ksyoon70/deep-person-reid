import xml.etree.ElementTree as ET
import json
import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_vehicle_type(type_id):
    vehicle_map = {
        1: "car",
        2: "car",
        3: "car",
        4: "car",
        5: "car",
        6: "truck",
        7: "bus",
        8: "truck",
        9: "car"
    }
    return vehicle_map.get(int(type_id), "unknown")

def get_color(color_id):
    color_map = {
        1: "yellow",
        2: "redorange",
        3: "green",
        4: "gray",
        5: "redorange",
        6: "blue",
        7: "white",
        8: "gray",
        9: "redorange",
        10: "black"
    }
    return color_map.get(int(color_id), "unknown")

def process_veri_xml():
    # Paths
    xml_path = "VeRi/veri/train_label.xml"
    image_dir = "VeRi/veri/image_train"
    
    # Create output directory if it doesn't exist
    output_dir = "VeRi/veri/json_labels"
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse XML file with explicit encoding
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    root = ET.fromstring(xml_content)
    
    # Get all items
    items = root.findall('.//Item')
    
    # Process each item with progress bar
    for item in tqdm(items, desc="Processing images", unit="image"):
        image_name = item.get('imageName')
        type_id = item.get('typeID')
        color_id = item.get('colorID')
        
        # Check if image exists
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"\nWarning: Image not found: {image_name}")
            continue
        
        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Create JSON data
        json_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [
                {
                    "label": get_vehicle_type(type_id),
                    "points": [
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height -1],
                        [0, height - 1]
                    ],
                    "group_id": None,
                    "shape_type": "polygon",
                    "color": get_color(color_id),
                    "flags": {}
                }
            ],
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
            "imagePath": image_name
        }
        
        # Create JSON file
        json_filename = os.path.splitext(image_name)[0] + '.json'
        json_path = os.path.join(output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)
        
        # Copy image file to output directory
        output_image_path = os.path.join(output_dir, image_name)
        shutil.copy2(image_path, output_image_path)
        
    
    print("\nProcessing completed successfully!")

if __name__ == "__main__":
    process_veri_xml()
