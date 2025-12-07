# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "anthropic>=0.18.0",
#     "openai>=1.12.0",
#     "pillow>=10.0.0",
#     "requests>=2.31.0",
# ]
# ///
"""
Visually inspect rendered README pages using VLM (Vision Language Model).

Opportunistically uses API keys from environment variables:
    - ANTHROPIC_API_KEY (Claude Vision)
    - OPENAI_API_KEY (GPT-4 Vision)

Renders markdown using mdpreview, takes screenshot, then uses VLM to critique.

Usage:
    python vlm_inspect_readme.py <readme_path> [output_dir]
    
Environment Variables:
    ANTHROPIC_API_KEY: Claude API key (preferred)
    OPENAI_API_KEY: OpenAI API key (fallback)
"""

import sys
import os
import subprocess
import time
import base64
from pathlib import Path
from PIL import Image

# Try importing VLM clients
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

def get_api_keys():
    """Get available API keys from environment."""
    keys = {}
    if HAS_ANTHROPIC:
        keys['anthropic'] = os.getenv('ANTHROPIC_API_KEY')
    if HAS_OPENAI:
        keys['openai'] = os.getenv('OPENAI_API_KEY')
    return keys

def image_to_base64(image_path):
    """Convert image to base64 for API."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def inspect_with_claude(image_path, readme_path):
    """Inspect README screenshot using Claude Vision."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    
    if not HAS_ANTHROPIC:
        return None
    
    try:
        client = Anthropic(api_key=api_key)
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(image_data).decode('utf-8')
                            }
                        },
                        {
                            "type": "text",
                            "text": f"""Critique this rendered README page for a technical library.

Focus on:
1. Visual clarity and readability
2. Mathematical formulas rendering correctly
3. Code blocks formatting
4. Image/visualization placement and quality
5. Overall professional appearance
6. Any rendering issues or visual problems

Be specific and constructive. Note any issues with:
- Formula rendering (LaTeX/MathJax)
- Code syntax highlighting
- Image quality or placement
- Text readability
- Layout issues
- Missing or broken elements

Provide a score 1-10 for visual quality and specific recommendations."""
                        }
                    ]
                }
            ]
        )
        
        return message.content[0].text
    
    except Exception as e:
        print(f"⚠️  Error with Claude API: {e}")
        return None

def inspect_with_openai(image_path, readme_path):
    """Inspect README screenshot using GPT-4 Vision."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return None
    
    if not HAS_OPENAI:
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Critique this rendered README page for a technical library.

Focus on:
1. Visual clarity and readability
2. Mathematical formulas rendering correctly
3. Code blocks formatting
4. Image/visualization placement and quality
5. Overall professional appearance
6. Any rendering issues or visual problems

Be specific and constructive. Note any issues with:
- Formula rendering (LaTeX/MathJax)
- Code syntax highlighting
- Image quality or placement
- Text readability
- Layout issues
- Missing or broken elements

Provide a score 1-10 for visual quality and specific recommendations."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"⚠️  Error with OpenAI API: {e}")
        return None

def take_screenshot(readme_path, output_dir):
    """Take screenshot of rendered README using mdpreview + Playwright."""
    readme_path = Path(readme_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check mdpreview is available
    try:
        subprocess.run(['mdpreview', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: mdpreview not found. Install with: go install github.com/henrywallace/mdpreview@latest")
        return None
    
    # Start mdpreview server
    PORT = 8080
    screenshot_file = output_dir / f"{readme_path.stem}_screenshot.png"
    
    # Start mdpreview in background
    process = subprocess.Popen(
        ['mdpreview', '-addr', f':{PORT}', str(readme_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    # Take screenshot with Playwright
    screenshot_script = Path(__file__).parent / 'screenshot_readme.js'
    if not screenshot_script.exists():
        print("❌ Error: screenshot_readme.js not found")
        process.terminate()
        return None
    
    try:
        result = subprocess.run(
            ['node', str(screenshot_script), f'http://localhost:{PORT}', str(screenshot_file)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"❌ Error taking screenshot: {result.stderr}")
            process.terminate()
            return None
        
        # Wait a bit for file to be written
        time.sleep(1)
        
        if screenshot_file.exists():
            return screenshot_file
        else:
            print("❌ Error: Screenshot file not created")
            return None
    
    except subprocess.TimeoutExpired:
        print("❌ Error: Screenshot timeout")
        process.terminate()
        return None
    finally:
        # Kill mdpreview
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python vlm_inspect_readme.py <readme_path> [output_dir]")
        sys.exit(1)
    
    readme_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('readme_screenshots')
    
    if not readme_path.exists():
        print(f"❌ Error: README not found: {readme_path}")
        sys.exit(1)
    
    # Check for API keys
    api_keys = get_api_keys()
    available_apis = [k for k, v in api_keys.items() if v]
    
    if not available_apis:
        print("⚠️  No API keys found in environment variables.")
        print("   Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable VLM inspection.")
        print("   Falling back to screenshot generation only.")
        available_apis = []
    
    print(f"📸 Taking screenshot of {readme_path}...")
    screenshot_path = take_screenshot(readme_path, output_dir)
    
    if not screenshot_path:
        print("❌ Failed to generate screenshot")
        sys.exit(1)
    
    print(f"✅ Screenshot saved: {screenshot_path}")
    
    # Validate image
    try:
        img = Image.open(screenshot_path)
        print(f"   Image size: {img.size[0]}x{img.size[1]}")
    except Exception as e:
        print(f"❌ Error validating image: {e}")
        sys.exit(1)
    
    # VLM inspection (opportunistic)
    if available_apis:
        print(f"\n🔍 Inspecting with VLM ({', '.join(available_apis)})...")
        
        critique = None
        vlm_used = None
        
        # Try Claude first (preferred)
        if 'anthropic' in available_apis:
            print("   Trying Claude Vision...")
            critique = inspect_with_claude(screenshot_path, readme_path)
            if critique:
                vlm_used = 'Claude'
        
        # Fallback to OpenAI
        if not critique and 'openai' in available_apis:
            print("   Trying GPT-4 Vision...")
            critique = inspect_with_openai(screenshot_path, readme_path)
            if critique:
                vlm_used = 'GPT-4'
        
        if critique:
            print(f"\n✅ VLM Critique ({vlm_used}):")
            print("=" * 70)
            print(critique)
            print("=" * 70)
            
            # Save critique to file
            critique_file = output_dir / f"{readme_path.stem}_vlm_critique.txt"
            with open(critique_file, 'w') as f:
                f.write(f"VLM Inspection ({vlm_used})\n")
                f.write(f"README: {readme_path}\n")
                f.write(f"Screenshot: {screenshot_path}\n")
                f.write("=" * 70 + "\n\n")
                f.write(critique)
            
            print(f"\n💾 Critique saved: {critique_file}")
        else:
            print("⚠️  VLM inspection failed (check API keys and network)")
    else:
        print("\n⚠️  Skipping VLM inspection (no API keys available)")
        print("   Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable")
    
    print(f"\n✅ Complete! Screenshot: {screenshot_path}")

if __name__ == '__main__':
    main()

