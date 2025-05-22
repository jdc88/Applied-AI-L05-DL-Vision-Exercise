import pytesseract
from PIL import Image
import os
import csv
import re

# Folder containing receipt images
receipts_folder = "L05_DL_Vision_Receipts"
output_csv = "shopping_summary.csv"

# Extract store name from the top line of the text
def extract_store_name(text):
    lines = text.splitlines()
    return lines[0].strip() if lines else "Unknown"

# Extract item and amount pairs using regex
def extract_items(text):
    item_pattern = re.compile(r"(.+?)\s+\$?([0-9]+\.[0-9]{2})")
    items = []
    for line in text.splitlines():
        match = item_pattern.search(line)
        if match:
            item = match.group(1).strip()
            price = float(match.group(2))
            items.append((item, price))
    return items

# Process each image file
rows = [["Store Name", "Item Name", "Amount"]]

for filename in sorted(os.listdir(receipts_folder)):
    if filename.lower().endswith('.jpg'):
        path = os.path.join(receipts_folder, filename)
        image = Image.open(path)

        # Optional: Improve OCR accuracy
        image = image.convert('L')  # Grayscale
        text = pytesseract.image_to_string(image)

        store = extract_store_name(text)
        items = extract_items(text)
        total = sum(amount for _, amount in items)

        for item, amount in items:
            rows.append([store, item, f"${amount:>6.2f}"])
        rows.append([store, "Total", f"${total:>6.2f}"])
        rows.append(["", "", ""])  # Blank row between receipts

# Write to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"✅ Extracted data saved to {output_csv}")
