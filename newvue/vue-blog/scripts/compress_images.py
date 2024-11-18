import os
from PIL import Image
import sys
from pathlib import Path

def compress_image(image_path, max_width=1920, quality=85):
    """
    Compress image and adjust resolution
    
    Parameters:
    image_path: path to image
    max_width: maximum width (maintains aspect ratio)
    quality: compression quality (1-100)
    """
    try:
        img = Image.open(image_path)
        
        # Convert image format to RGB (handle RGBA images)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Adjust resolution
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # Compress and save
        img.save(image_path, 'JPEG', quality=quality, optimize=True)
        
        # Get file size
        size_kb = os.path.getsize(image_path) / 1024
        return True, size_kb
    except Exception as e:
        return False, str(e)

def process_directory(directory):
    """Process all images in specified directory"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.webp'}
    total_saved = 0
    processed_count = 0
    errors = []
    
    print(f"\nProcessing directory: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_formats:
                file_path = os.path.join(root, file)
                original_size = os.path.getsize(file_path) / 1024
                
                print(f"\nProcessing: {file}")
                print(f"Original size: {original_size:.2f}KB")
                
                success, result = compress_image(file_path)
                
                if success:
                    saved = original_size - result
                    total_saved += saved
                    processed_count += 1
                    print(f"Compressed size: {result:.2f}KB")
                    print(f"Saved: {saved:.2f}KB ({(saved/original_size*100):.1f}%)")
                else:
                    errors.append(f"{file}: {result}")
    
    return processed_count, total_saved, errors

def main():
    # Ensure dist directory exists
    if not os.path.exists('dist'):
        print("Error: dist directory not found. Please run npm run build first")
        sys.exit(1)
    
    directories = [
        'dist/img',
        'dist/posts/images'
    ]
    
    total_processed = 0
    total_saved_space = 0
    all_errors = []
    
    for directory in directories:
        if os.path.exists(directory):
            processed, saved, errors = process_directory(directory)
            total_processed += processed
            total_saved_space += saved
            all_errors.extend(errors)
    
    # Print summary
    print("\n" + "="*50)
    print(f"Compression completed!")
    print(f"Total images processed: {total_processed}")
    print(f"Total space saved: {total_saved_space:.2f}KB ({total_saved_space/1024:.2f}MB)")
    
    if all_errors:
        print("\nErrors encountered during processing:")
        for error in all_errors:
            print(f"- {error}")

if __name__ == "__main__":
    main()