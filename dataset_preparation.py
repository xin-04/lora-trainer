import gradio as gr
import cv2
import os
import shutil
import tkinter as tk
from tkinter import filedialog, ttk
from transformers import BlipProcessor, BlipForConditionalGeneration

UPLOAD_DIR = r"C:\lora-trainer\dataset\preedited_img"

# Ensure the directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def load_dataset_dir():
    root = tk.Tk()
    root.withdraw()
    default_path = os.path.abspath(r"C:\lora-trainer\dataset\img")

    dataset_path = filedialog.askdirectory(
        initialdir = default_path,
        title = "Select Dataset Image Directory",
    )

    return dataset_path or ""

def save_uploaded_image(image):
    if image is None:
        return "No image uploaded."

    # Get the original filename
    filename = os.path.basename(image.name)

    # Construct full path to save
    save_path = os.path.join(UPLOAD_DIR, filename)

    # Copy the uploaded image to the target folder
    shutil.copy(image.name, save_path)

    output_text = "Image saved to: " + save_path
    return output_text

def load_preedit_image():
    default_dir = os.path.abspath(r"C:\lora-trainer\dataset\preedited_img")
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        initialdir = default_dir,
        title="Select an image file",
        filetypes=[
            ("All image files", "*.png *.jpg *.jpeg *.webp"),
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg"),
            ("JPEG files", "*.jpeg"),
            ("WEBP files", "*.webp")           
        ]
    )

    return file_path

def free_crop_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
        return

    clone = img.copy()
    roi = cv2.selectROI("Press Enter to confirm and press enter again to save the cropped image.", img, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi

    if w == 0 or h == 0:
        print("No crop selected.")
        cv2.destroyAllWindows()
        return

    cropped_img = clone[y:y+h, x:x+w]
    cv2.imshow("Cropped Image", cropped_img)
    output_text = "Press Enter to confirm and press enter again to save the cropped image. Press any other key to cancel."
    print("Press Enter to confirm and press enter again to save the cropped image. Press any other key to cancel.")

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == 13:  # Enter key
        # Prompt user to choose save path
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPG files", "*.jpg"),
                ("JPEG files", "*.jpeg"),
                ("WEBP files", "*.webp")
            ],
            title="Save Cropped Image"
        )

        if file_path:
            success = cv2.imwrite(file_path, cropped_img)
            if success:
                output_text = "Cropped image saved to " + file_path
                print(f"Cropped image saved to {file_path}")
            else:
                output_text = "Failed to save the image. Make sure the file extension is supported."
                print("Failed to save the image. Make sure the file extension is supported.")
        else:
            print("Save canceled.")
    else:
        print("Crop canceled.")

    return output_text

def crop_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image.")
        return

    def get_crop_size():
        selected_size = tk.StringVar()
        crop_result = {"size": None}

        def submit():
            crop_result["size"] = selected_size.get()
            popup.quit()

        popup = tk.Tk()
        popup.title("Select Crop Size")
        popup.geometry("250x100")
        popup.resizable(False, False)

        tk.Label(popup, text="Choose crop size:").pack(pady=5)
        options = ["512x512", "512x768", "768x512", "768x768"]
        selected_size.set(options[0])

        combo = ttk.Combobox(popup, values=options, textvariable=selected_size, state="readonly")
        combo.pack(pady=5)

        tk.Button(popup, text="OK", command=submit).pack(pady=5)
        popup.mainloop()
        popup.destroy()

        return crop_result["size"]

    crop_size = get_crop_size()
    crop_w, crop_h = map(int, crop_size.split("x"))

    clone = img.copy()
    window_name = "Click to center crop box (Press Enter to save)"
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, clone)

    crop_coords = []

    def select_crop(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x1 = max(0, x - crop_w // 2)
            y1 = max(0, y - crop_h // 2)
            x2 = min(img.shape[1], x1 + crop_w)
            y2 = min(img.shape[0], y1 + crop_h)

            # Adjust if crop area exceeds image
            x1 = max(0, x2 - crop_w)
            y1 = max(0, y2 - crop_h)

            crop_coords.clear()
            crop_coords.extend([x1, y1, x2, y2])

            preview = clone.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow(window_name, preview)

    cv2.setMouseCallback(window_name, select_crop)
    print("Click to select center of crop box. Press Enter to save.")

    while True:
        key = cv2.waitKey(0)
        if key == 13:  # Enter
            break

    cv2.destroyAllWindows()

    if not crop_coords:
        print("No crop selected.")
        return

    x1, y1, x2, y2 = crop_coords
    cropped_img = clone[y1:y2, x1:x2]

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg"),
            ("JPEG files", "*.jpeg"),
            ("WEBP files", "*.webp")
        ],
        title="Save Cropped Image"
    )

    if file_path:
        success = cv2.imwrite(file_path, cropped_img)
        if success:
            output_text = "Cropped image saved to " + file_path
            print(f"Cropped image saved to {file_path}")
        else:
            output_text = "Failed to save image. Check the file extension."
            print("Failed to save image. Check the file extension.")
    else:
        output_text = "Save cancelled."
        print("Save cancelled.")

    return output_text

def resize_image(image_path, width, height, save_path=None):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not load image."

    # Resize the image
    resized_img = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)

    if save_path is None:
        root = tk.Tk()
        root.withdraw()

        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPG files", "*.jpg"),
                ("JPEG files", "*.jpeg"),
                ("WEBP files", "*.webp")
            ],
            title="Save Resized Image"
        )

        if not save_path:
            return "Save cancelled."

    # Save the resized image
    success = cv2.imwrite(save_path, resized_img)
    if success:
        output_text = "Image successfully resized and saved to: " + save_path
    else:
        output_text = "Error: Failed to save resized image."
    
    return output_text

def resize_image_by_ratio(image_path, scale_ratio):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        output_text = "Error: Could not load image."
        return output_text

    # Get original dimensions
    original_height, original_width = img.shape[:2]

    # Calculate new dimensions
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Ask user for save location
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg"),
            ("JPEG files", "*.jpeg"),
            ("WEBP files", "*.webp")
        ],
        title="Save Resized Image"
    )

    if not file_path:
        return "Save cancelled."

    # Save the resized image
    success = cv2.imwrite(file_path, resized_img)
    if success:
        output_text = "Image resized to " + str(new_width) + "x" + str(new_height) + " and saved to: " + file_path
    else:
        output_text = "Error: Failed to save resized image."

    return output_text

# Automatic captioning function for a single image
def generate_caption(image_path):
    # Open and preprocess image
    raw_image = cv2.imread(image_path)
    if raw_image is None:
        output_text = "Error: Could not load image."
        return output_text

    # Convert image to RGB for the BLIP model
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    # Process the image and generate caption
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

def generate_captions_for_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if it's an image file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            caption = generate_caption(file_path)
            
            # Create a .txt file with the same name as the image
            base_name, _ = os.path.splitext(filename)
            caption_file_path = os.path.join(folder_path, base_name + ".txt")

            with open(caption_file_path, 'w', encoding='utf-8') as f:
                f.write(caption)

            print(f"Caption saved: {caption_file_path}")
    
    output_text = "Captions generated and saved for all images."
    return output_text

def dataset_preparation_tab():
    # Tab layout for "Dataset Preparation"
    with gr.Tab("Dataset Preparation"):
        
        gr.Markdown("### ⬆️ Upload Pre-edited Image")
        with gr.Group():
            with gr.Row():
                file_upload = gr.File(label="Upload Image", file_types=[".png", ".jpg", ".jpeg", ".webp"])
                upload_status = gr.Textbox(label="Upload Status")
        
        gr.Markdown("### 👇 Select Image To Crop / Resize")
        with gr.Group():
            with gr.Row():
                preedit_img_path = gr.Textbox(label="Select Image Path", placeholder="Path to pre-edited image", scale=4)
                browse_preedit_img_btn = gr.Button("📁", variant="secondary", scale=1)
            
        gr.Markdown("### ✂️ Image Cropping Tool")
        with gr.Row():
            free_crop_btn = gr.Button("Crop Image Freely")            
            crop_btn = gr.Button("Crop Image with Aspect")

        gr.Markdown("### 🪟 Image Resizing Tool")  
        with gr.Group():
            with gr.Row():
                width = gr.Number(label="Enter Width:")
                height = gr.Number(label="Enter Height:")
                resize_image_btn = gr.Button("Resize", variant="secondary")
            with gr.Row():
                ratio = gr.Number(label="Enter Ratio")
                resize_image_ratio_btn = gr.Button("Resize while keeping aspect", variant="secondary")

        with gr.Group():
            gr.Markdown("### 📄 Auto-captioning Tool")
            with gr.Row():
                dataset_dir = gr.Textbox(label="Select Dataset Directory", placeholder="Path to dataset", scale=4)
                dataset_dir_btn = gr.Button("📁", variant="secondary", scale=1)
            caption_btn = gr.Button("Auto-captioning")

        gr.Markdown("### ⌛ Your Current Status") 
        output_text = gr.Textbox(label="Output Status")

        dataset_dir_btn.click(fn=load_dataset_dir, inputs=[], outputs=dataset_dir)
        browse_preedit_img_btn.click(fn=load_preedit_image, inputs=[], outputs=preedit_img_path)
        free_crop_btn.click(fn=free_crop_image, inputs=[preedit_img_path], outputs=output_text)
        crop_btn.click(fn=crop_image, inputs=[preedit_img_path], outputs=output_text)
        file_upload.change(fn=save_uploaded_image, inputs=[file_upload], outputs=[upload_status])
        resize_image_btn.click(fn=resize_image, inputs=[preedit_img_path, width, height], outputs=output_text)
        resize_image_ratio_btn.click(fn=resize_image_by_ratio, inputs=[preedit_img_path, ratio], outputs=output_text)
        caption_btn.click(fn=generate_captions_for_folder, inputs=dataset_dir, outputs=[])