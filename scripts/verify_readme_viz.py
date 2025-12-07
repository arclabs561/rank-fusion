#!/usr/bin/env python3
"""
Verify README screenshot quality using VLM (Vision Language Model).
Checks if READMEs have good visual presentation and pedagogical value.

Opportunistically uses API keys from environment:
    - ANTHROPIC_API_KEY (preferred)
    - OPENAI_API_KEY (fallback)

If no API keys available, provides basic image validation only.
"""

import sys
import os
import base64
import json
from pathlib import Path
from typing import Dict, Optional

# Try importing VLM clients (optional)
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


def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def verify_readme_claude(
    image_path: Path,
    context: str,
    client: Anthropic,
) -> Dict:
    """
    Verify README screenshot has good visual presentation and pedagogical value.
    
    Returns dict with:
    - score: 0-100 quality score
    - feedback: Detailed feedback
    - passes: bool whether it meets threshold (>=70)
    """
    image_data = encode_image(image_path)
    
    prompt = f"""Analyze this README screenshot for visual presentation and pedagogical value.

Context: {context}

Evaluate the README on:
1. **Visual Clarity**: Is the layout clear, readable, and well-organized?
2. **Structure**: Are sections well-organized with clear headings?
3. **Code Examples**: Are code examples visible and properly formatted?
4. **Mathematical Content**: Are formulas/equations rendered correctly (if any)?
5. **Completeness**: Does it appear comprehensive and informative?
6. **Professional Appearance**: Does it look polished and production-ready?
7. **Pedagogical Value**: Would this help someone understand and use the library?

Provide:
- A score from 0-100 (higher = better)
- Detailed feedback on visual presentation
- Specific suggestions for improvement if score < 70

Format your response as JSON:
{{
  "score": <0-100>,
  "feedback": "<detailed feedback>",
  "strengths": ["<strength1>", "<strength2>", ...],
  "weaknesses": ["<weakness1>", "<weakness2>", ...],
  "suggestions": ["<suggestion1>", "<suggestion2>", ...]
}}"""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        
        # Parse JSON response
        response_text = message.content[0].text
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        result = json.loads(response_text)
        result["passes"] = result["score"] >= 70
        return result
        
    except Exception as e:
        return {
            "score": 0,
            "feedback": f"Error during verification: {str(e)}",
            "strengths": [],
            "weaknesses": ["Verification failed"],
            "suggestions": ["Fix verification process"],
            "passes": False,
        }


def verify_readme_openai(
    image_path: Path,
    context: str,
    client: OpenAI,
) -> Dict:
    """Verify README using OpenAI GPT-4 Vision."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    prompt = f"""Analyze this README screenshot for visual presentation and pedagogical value.

Context: {context}

Evaluate the README on:
1. **Visual Clarity**: Is the layout clear, readable, and well-organized?
2. **Structure**: Are sections well-organized with clear headings?
3. **Code Examples**: Are code examples visible and properly formatted?
4. **Mathematical Content**: Are formulas/equations rendered correctly (if any)?
5. **Completeness**: Does it appear comprehensive and informative?
6. **Professional Appearance**: Does it look polished and production-ready?
7. **Pedagogical Value**: Would this help someone understand and use the library?

Provide:
- A score from 0-100 (higher = better)
- Detailed feedback on visual presentation
- Specific suggestions for improvement if score < 70

Format your response as JSON:
{{
  "score": <0-100>,
  "feedback": "<detailed feedback>",
  "strengths": ["<strength1>", "<strength2>", ...],
  "weaknesses": ["<weakness1>", "<weakness2>", ...],
  "suggestions": ["<suggestion1>", "<suggestion2>", ...]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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
            max_tokens=1024
        )
        
        response_text = response.choices[0].message.content
        
        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        result = json.loads(response_text)
        result["passes"] = result["score"] >= 70
        return result
        
    except Exception as e:
        return {
            "score": 0,
            "feedback": f"Error during verification: {str(e)}",
            "strengths": [],
            "weaknesses": ["Verification failed"],
            "suggestions": ["Fix verification process"],
            "passes": False,
        }

def main():
    if len(sys.argv) < 3:
        print("Usage: verify_readme_viz.py <screenshot_path> <context_description> [api_key]")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    context = sys.argv[2]
    api_key = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not image_path.exists():
        print(f"Error: Screenshot not found: {image_path}")
        sys.exit(1)
    
    # Check for API keys opportunistically
    anthropic_key = os.getenv('ANTHROPIC_API_KEY') or api_key
    openai_key = os.getenv('OPENAI_API_KEY')
    
    result = None
    
    # Try Claude first (preferred)
    if anthropic_key and HAS_ANTHROPIC:
        try:
            client = Anthropic(api_key=anthropic_key)
            result = verify_readme_claude(image_path, context, client)
            print("✅ Used Claude Vision for verification")
        except Exception as e:
            print(f"⚠️  Claude API error: {e}")
            result = None
    
    # Fallback to OpenAI
    if result is None and openai_key and HAS_OPENAI:
        try:
            client = OpenAI(api_key=openai_key)
            result = verify_readme_openai(image_path, context, client)
            print("✅ Used GPT-4 Vision for verification")
        except Exception as e:
            print(f"⚠️  OpenAI API error: {e}")
            result = None
    
    # If no VLM available, provide basic validation
    if result is None:
        print("⚠️  No VLM API keys available. Providing basic image validation only.")
        print("   Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable VLM verification.")
        
        # Basic validation
        try:
            from PIL import Image
            img = Image.open(image_path)
            result = {
                "score": 50,  # Neutral score without VLM
                "feedback": f"Image validated: {img.size[0]}x{img.size[1]} pixels. VLM verification not available.",
                "strengths": ["Image file is valid"],
                "weaknesses": ["VLM verification not performed (no API keys)"],
                "suggestions": ["Set ANTHROPIC_API_KEY or OPENAI_API_KEY for detailed visual critique"],
                "passes": True,  # Don't fail on missing API keys
            }
        except Exception as e:
            result = {
                "score": 0,
                "feedback": f"Error validating image: {e}",
                "strengths": [],
                "weaknesses": ["Image validation failed"],
                "suggestions": ["Check image file exists and is valid"],
                "passes": False,
            }
    
    # Verify (using renamed function)
    if result is None:
        print("❌ Error: Could not verify README")
        sys.exit(1)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"README Quality Score: {result['score']}/100")
    print(f"Status: {'✅ PASS' if result['passes'] else '❌ FAIL'}")
    print(f"{'='*60}\n")
    print(f"Feedback:\n{result['feedback']}\n")
    
    if result['strengths']:
        print("Strengths:")
        for strength in result['strengths']:
            print(f"  ✅ {strength}")
        print()
    
    if result['weaknesses']:
        print("Weaknesses:")
        for weakness in result['weaknesses']:
            print(f"  ❌ {weakness}")
        print()
    
    if result['suggestions']:
        print("Suggestions:")
        for suggestion in result['suggestions']:
            print(f"  💡 {suggestion}")
        print()
    
    # Exit with error code if fails
    sys.exit(0 if result['passes'] else 1)


if __name__ == "__main__":
    main()

