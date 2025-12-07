# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pillow>=10.0.0",
# ]
# ///
"""
Validate that visualization images were generated correctly.

Checks:
    - Files exist
    - Images are valid PNG files
    - Images meet minimum size requirements
    - Images are not corrupted
"""

import sys
from pathlib import Path
from PIL import Image

def validate_image(png_path):
    """Validate a single image file."""
    if not png_path.exists():
        return False, f"File not found: {png_path}"
    
    try:
        img = Image.open(png_path)
        img.verify()  # Verify it's not corrupted
        
        # Reopen for size check (verify() closes the file)
        img = Image.open(png_path)
        width, height = img.size
        
        if width < 800 or height < 600:
            return False, f"Image too small: {width}x{height} (minimum 800x600)"
        
        if width > 5000 or height > 5000:
            return False, f"Image suspiciously large: {width}x{height}"
        
        return True, f"OK ({width}x{height})"
    
    except Exception as e:
        return False, f"Invalid or corrupted image: {e}"

def main():
    """Validate all visualization images."""
    script_dir = Path(__file__).parent
    
    # Expected visualization files
    expected_files = [
        'rrf_statistical_analysis.png',
        'rrf_method_comparison.png',
        'rrf_k_statistical.png',
        'rrf_hypothesis_testing.png',
        'rrf_effect_size.png',
    ]
    
    all_valid = True
    
    for filename in expected_files:
        filepath = script_dir / filename
        is_valid, message = validate_image(filepath)
        
        if is_valid:
            print(f"✅ {filename}: {message}")
        else:
            print(f"❌ {filename}: {message}")
            all_valid = False
    
    if all_valid:
        print("\n✅ All visualizations validated successfully!")
        return 0
    else:
        print("\n❌ Some visualizations failed validation")
        return 1

if __name__ == '__main__':
    sys.exit(main())

