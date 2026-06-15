import torch
import os
import random  # रैंडम नंबर जनरेट करने के लिए
from diffusers import StableDiffusionXLPipeline 
from IPython.display import display
from google.colab import drive

# 1. गूगल ड्राइव को कनेक्ट करें
print("गूगल驱动 (Google Drive) को जोड़ा जा रहा है...")
drive.mount('/content/drive')

save_dir = "/content/drive/MyDrive/AI_Images"
os.makedirs(save_dir, exist_ok=True)

# 2. GPU चेक करें
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# 3. SDXL मॉडल लोड करें (यह तुरंत लोड होगा क्योंकि पहले से डाउनलोडेड है)
print("SDXL मॉडल लोड हो रहा है, कृपया रुकें...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16"
)
pipe = pipe.to(device)
pipe.enable_model_cpu_offload()  # रैम बचाने के लिए

print("\n🎉 मॉडल तैयार है! अब आप असीमित फोटो बना सकते हैं।")
print("(बाहर निकलने के लिए 'exit' टाइप करें)\n")

# 🔄 यहाँ से लूप शुरू होता है जो बार-बार प्रॉम्प्ट मांगेगा
while True:
    # यूजर से इनपुट मांगना
    user_prompt = input("अपना प्रॉम्प्ट यहाँ लिखें (या बंद करने के लिए 'exit' लिखें): ")
    
    # अगर यूजर exit लिखता है तो लूप बंद हो जाएगा
    if user_prompt.lower() == 'exit':
        print("एआई इमेज जनरेटर बंद हो गया है। बाय-बाय भाई!")
        break
        
    # अगर इनपुट खाली है तो दोबारा मांगो
    if not user_prompt.strip():
        print("कृपया कुछ तो लिखिए भाई!\n")
        continue
        
    print(f"\nAI फोटो बना रहा है: '{user_prompt}'...")
    
    try:
        # फोटो जनरेट करना
        bad_stuff = "blurred, deformed, low quality"
        output_images = pipe(user_prompt, negative_prompt=bad_stuff, num_inference_steps=30).images
        final_photo = output_images[0]
        
        # 🎲 रैंडम नंबर जनरेट करना (1000 से 9999 के बीच)
        random_num = random.randint(1000, 999999)
        file_name = f"ai_image_gen_{random_num}.png"
        full_save_path = os.path.join(save_dir, file_name)
        
        # फोटो सेव करना
        final_photo.save(full_save_path)
        print(f"✅ फोटो सेव हो गई: AI_Images/{file_name}")
        
        # फोटो स्क्रीन पर दिखाना
        display(final_photo)
        print("-" * 50 + "\n") # एक लाइन खींचना ताकि अगला प्रॉम्प्ट साफ दिखे
        
    except Exception as e:
        print(f"अरे भाई, एरर बाबा फिर आ गए: {e}\n")
