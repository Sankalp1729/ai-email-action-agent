from PIL import Image, ImageDraw, ImageFont

# Create a transparent image
width, height = 400, 200
img = Image.new("RGBA", (width, height), (255, 255, 255, 0))  # Transparent background
draw = ImageDraw.Draw(img)

# Add text in the center
text = "AI Email Agent"
font = ImageFont.load_default()

# Calculate text position
bbox = draw.textbbox((0, 0), text, font=font)
text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
x = (width - text_width) // 2
y = (height - text_height) // 2

# Draw text on image
draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))

# Save image to assets folder
img.save("assets/ai_email_logo.png", "PNG")
print("✅ Transparent logo created at assets/ai_email_logo.png")
