from smolagents import CodeAgent, LiteLLMModel, \
    DuckDuckGoSearchTool, VisitWebpageTool, tool
import os
import requests
import textwrap
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import logging


models_to_choose = ["google/gemini-2.0-flash-001", 
                    "google/gemini-flash-1.5-8b", 
                    "google/gemma-3-27b-it:free",
                    "google/gemma-3-1b-it:free",
                    "openai/gpt-4.1-nano",
                    "meta-llama/llama-4-scout",
                    "meta-llama/llama-4-maverick:free",
                    "mistralai/mistral-small-3.1-24b-instruct",
                    "mistralai/mistral-small-3.1-24b-instruct:free",
                    "qwen/qwen2.5-vl-72b-instruct:free",
                    "microsoft/phi-4-multimodal-instruct"]


PROVIDER = "openrouter"
MODEL_ID_MANAGER = f"{PROVIDER}/{models_to_choose[6]}"
MODEL_ID_SCRAPPER = f"{PROVIDER}/{models_to_choose[6]}"
OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN", "<type your token here>")


@tool
def add_caption_to_image(image_url: str, caption: str, 
                               output_path: str = "output_image.png", 
                               font_path: str = "arial.ttf", 
                               font_size_ratio: float = 0.05) -> Image:
    """
    Add a centered caption with a semi-transparent background to an image from a URL.
    
    Args:
        image_url (str): URL of the image.
        caption (str): Text to add to the image.
        output_path (str): Path to save the output image (default: output_image.png).
        font_path (str): Path to the TTF font file (default: arial.ttf).
        font_size_ratio (float): Ratio of image height to determine font size (default: 0.05).
    
    Returns:
        PIL.Image: The modified image object.
    
    Raises:
        ValueError: If the image cannot be fetched or processed.
    """
    try:
        # Fetch the image asynchronously
        response = requests.get(image_url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch image: HTTP {response.status_code}")
        image_data = response.content
        
        # Load and convert image
        image = Image.open(BytesIO(image_data)).convert("RGBA")
        width, height = image.size
        
        
        # Calculate font size based on image height
        font_size = max(12, int(height * font_size_ratio))  # Ensure minimum readable size
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            logging.warning(f"Font loading failed: {e}. Using default font.")
            font = ImageFont.load_default()
        
        # Create draw object
        draw = ImageDraw.Draw(image)
        
        # Calculate text wrapping to fit 80% of image width
        max_text_width = int(width * 0.8)
        wrapped_text = []
        for line in caption.split("\n"):
            # Dynamically wrap text based on actual pixel width
            while line:
                for i in range(len(line), -1, -1):
                    test_line = line[:i]
                    text_bbox = draw.textbbox((0, 0), test_line, font=font)
                    if text_bbox[2] - text_bbox[0] <= max_text_width or i == 0:
                        wrapped_text.append(test_line)
                        line = line[i:].strip()
                        break
        
        # Calculate text dimensions
        line_spacing = int(font_size * 0.2)  # 20% of font size for spacing
        text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in wrapped_text)
        text_height += (len(wrapped_text) - 1) * line_spacing
        text_width = max(draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in wrapped_text)
        
        # Create semi-transparent background for text
        padding = 10
        background = Image.new("RGBA", (text_width + 2 * padding, text_height + 2 * padding), (0, 0, 0, 128))
        image.paste(background, ((width - text_width - 2 * padding) // 2, padding), background)
        
        # Draw text
        y = padding
        for line in wrapped_text:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            line_width = text_bbox[2] - text_bbox[0]
            x = (width - line_width) // 2  # Center text
            draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
            y += (text_bbox[3] - text_bbox[1]) + line_spacing
        
        # Save the image
        image.save(output_path, "PNG")
        return image
    
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise ValueError(f"Failed to process image: {str(e)}")


model_manager = LiteLLMModel(
        model_id=MODEL_ID_MANAGER,
        api_key=OPENROUTER_TOKEN,
        temperature=1.0
)

model_scrapper = LiteLLMModel(
    model_id=MODEL_ID_SCRAPPER,
    api_key=OPENROUTER_TOKEN,
    temperature=1.0
)

web_scrapper_agent = CodeAgent(tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
                               model=model_scrapper, 
                               max_steps=3,
                               name="agent_of_web_scrapping", 
                               description="You are the agent that creates dataset for research purposes. You doing it by web data parsing and scrapping")


agent = CodeAgent(tools=[add_caption_to_image], 
                model=model_manager, managed_agents=[web_scrapper_agent],
                max_steps=5
                )

if __name__ == "__main__":
    response = agent.run("Find any url image of the rabbit in the internet")
    print(response)