# TransArt: A Multimodal Application for Vernacular Language Translation and Image Synthesis
To develop a web-based application that first translates text from Tamil to English and then uses the translated text to generate relevant images. This application aims to demonstrate the
seamless integration of language translation and creative AI to produce visual content from
textual descriptions.

# Skills take away From This Project
 
   * Deep Learning
   * Transformers
   * Hugging face models
   * LLM
   * Gradio/AWS
# Domain

     AIOPS
    
# Approach

1. Model Selection:

     *  Select a robust Tamil to English translation model from Hugging Face, such as
for example Helsinki-NLP/opus-mt-ta-en.

     *  Choose a reliable text-to-image model, example like
CompVis/stable-diffusion-v1-4, to generate images from the translated text.

     *  Integrate a text generation model like gpt-3 or gpt-neo or google
gemini api for producing creative English text based on the translated input.

2. Application Development:
    
     *  Build the app using gradio or stremlit to handle translation and image
generation requests.

3. Integration and Testing:
 
     *  Integrate the Hugging Face models using their APIs.

     *  Conduct thorough testing to ensure accurate translations and image relevance.

7. Deployment:
   
     *  Deployed on Hugging Face Spaces for easy access.

   
  TransArt - Hugging Face Spaces App

Explore the **TransArt** app, deployed on Hugging Face Spaces. This app allows you to [describe the app functionality briefly, e.g., "transform artworks with AI"].

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Hugging%20Face-orange)](https://huggingface.co/spaces/Nanthu22/TransArt)

Click the badge above or [here](https://huggingface.co/spaces/Nanthu22/TransArt) to try it out!


<iframe
    src="https://huggingface.co/spaces/Nanthu22/TransArt"
    width="100%"
    height="500"
    frameborder="0">
</iframe>
   
# Technology Used
   1. Deep Learning
      
   3. Python
      
   3. LLM
   
   4. Transformer

# PACKAGES AND LIBRARIES
  * import torch
  * import gradio as gr
  * import tempfile
  * import os
  * from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer
  * import requests
  * from PIL import Image
  * import io
# TransArt: A Multimodal Application for Vernacular Language Translation and Image Synthesis

**TransArt** is a cutting-edge multimodal application designed to bridge language barriers and enable creative visual representation. The app performs vernacular language translation and generates stunning images from the translated text. Powered by advanced models and built using Gradio, TransArt is deployed on Hugging Face Spaces for seamless access.

---

## Features

1. **Language Translation**
   - Utilizes models like `Helsinki-NLP/opus-mt-ta-en` for accurate vernacular-to-English text translation.

2. **Text-to-Image Generation**
   - Leverages reliable text-to-image models such as `CompVis/stable-diffusion-v1-4` to create visually engaging images from the translated text.

3. **Creative Text Generation**
   - Integrates text generation models like `GPT-3`, `GPT-Neo`, or Google Gemini API to produce imaginative English text based on the input.

4. **User-Friendly Interface**
   - Built with Gradio, offering a simple yet powerful interface for translation and image generation tasks.

---

## Application Development Workflow

### 1. Model Selection:
- **Translation:** `Helsinki-NLP/opus-mt-ta-en` for reliable vernacular-to-English translations.
- **Text-to-Image:** `CompVis/stable-diffusion-v1-4` for image synthesis.
- **Text Generation:** GPT models like `GPT-3`, `GPT-Neo`, or Google Gemini API for generating creative English text.

### 2. Application Development:
- Developed with **Gradio** to handle translation and image generation requests seamlessly.

### 3. Integration and Testing:
- Integrated Hugging Face models via their APIs.
- Conducted extensive testing to ensure translation accuracy and image relevance.

### 4. Deployment:
- Deployed on [Hugging Face Spaces](https://huggingface.co/spaces/Nanthu22/TransArt) for easy access.

---

## Try the TransArt App

Explore the **TransArt** app, deployed on Hugging Face Spaces. This app allows you to translate text and generate creative images.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Hugging%20Face-orange)](https://huggingface.co/spaces/Nanthu22/TransArt)

Click the badge above or [here](https://huggingface.co/spaces/Nanthu22/TransArt) to try it out!

```html
<iframe
    src="https://huggingface.co/spaces/Nanthu22/TransArt"
    width="100%"
    height="500"
    frameborder="0">
</iframe>
```

---

## Technology Used
1. **Deep Learning**
2. **Python**
3. **Large Language Models (LLMs)**
4. **Transformers**

---

## Packages and Libraries
```python
import torch
import gradio as gr
import tempfile
import os
from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer
import requests
from PIL import Image
import io
```

---

### Future Enhancements
- Expand support for additional vernacular languages.
- Improve the quality and diversity of generated images.
- Add voice input/output for enhanced accessibility.


