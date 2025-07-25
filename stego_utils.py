from PIL import Image
import base64
from cryptography.fernet import Fernet
import hashlib

# Function to derive key from password
def generate_key(password: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

# Hide message in image
def hide_message_in_image(input_image_path, output_image_path, message, password):
    # Encrypt message
    key = generate_key(password)
    fernet = Fernet(key)
    encrypted_message = fernet.encrypt(message.encode())

    # Convert encrypted message to binary
    binary_data = ''.join(format(byte, '08b') for byte in encrypted_message)

    image = Image.open(input_image_path).convert("RGB")
    pixels = list(image.getdata())

    if len(binary_data) > len(pixels) * 3:
        raise ValueError("Message is too large to hide in this image.")

    new_pixels = []
    data_index = 0

    for pixel in pixels:
        r, g, b = pixel
        if data_index < len(binary_data):
            r = (r & ~1) | int(binary_data[data_index])
            data_index += 1
        if data_index < len(binary_data):
            g = (g & ~1) | int(binary_data[data_index])
            data_index += 1
        if data_index < len(binary_data):
            b = (b & ~1) | int(binary_data[data_index])
            data_index += 1
        new_pixels.append((r, g, b))

    image.putdata(new_pixels)
    image.save(output_image_path)
    print(f"✅ Message hidden in {output_image_path}")

# Reveal message from image
def reveal_message_from_image(stego_image_path, password):
    key = generate_key(password)
    fernet = Fernet(key)

    image = Image.open(stego_image_path)
    pixels = list(image.getdata())

    binary_data = ""
    for pixel in pixels:
        for color in pixel[:3]:
            binary_data += str(color & 1)

    all_bytes = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
    encrypted_message_bytes = bytearray()
    for byte in all_bytes:
        encrypted_message_bytes.append(int(byte, 2))

    # Remove trailing zero bytes
    encrypted_message_bytes = encrypted_message_bytes.rstrip(b'\x00')

    # Decrypt the message
    try:
        hidden_message = fernet.decrypt(bytes(encrypted_message_bytes))
        return hidden_message.decode('utf-8')  # ✅ Removed the wrong .decode() issue
    except Exception:
        raise ValueError("Wrong password or corrupted image.")
