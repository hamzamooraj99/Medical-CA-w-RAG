from unsloth import FastVisionModel, FastLanguageModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from sentence_transformers import SentenceTransformer
import json
import faiss
from gtts import gTTS
import tempfile

# User Input containers for UI
user_text, user_image, user_audio = None
rag_switch = False

#region LOAD MODELS

# Text Model
text_model, text_processor = FastLanguageModel.from_pretrained(
    "esrgesbrt/trained_health_model_llama3.1_8B_bnb_4bits",
    load_in_4bit=True
)

# Vision Model
vision_model, vision_processor = FastVisionModel.from_pretrained(
    "hamzamooraj99/MedQA-Qwen-2B-LoRA16",
    load_in_4bit=True
)
vision_processor.image_processor.max_pixels = 512*512
vision_processor.image_processor.min_pixels = 224*224

# Whisper Model
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# RAG Model
rag_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#endregion



#region INPUT PROCESSING
def preprocess_text(text, vision_response, retrieved_info, rag_flag):
    if(rag_flag):
        alpaca_prompt = """ 
            Below is a query from a user regarding a medical condition or a description of symptoms. The user may also provide an image related to the query. Please provide an appropriate response to the user input with reference to the image response (if provided), making use of the retrieved information from the knowledge source.
            ### User Input:
            {}  

            ### Image Response:
            {}

            ### Retrieved Information
            {}
        
            ### Response:
            {}
        """
        prompt = alpaca_prompt.format(text, vision_response, retrieved_info, "")
    else:
        alpaca_prompt = """ 
            Below is a query from a user regarding a medical condition or a description of symptoms. The user may also provide an image related to the query. Please provide an appropriate response to the user input with reference to the image response (if provided).
            ### User Input:
            {}

            ### Image Response:
            {}
        
            ### Response:
            {}
        """
        prompt = alpaca_prompt.format(text, vision_response, "")
    
    inputs = text_processor([prompt], return_tensors="pt").to('cuda')
    return inputs

def preprocess_image(image, text):
    messages = [
        {'role': 'system',
         'content': [
             {'type': 'text', 'text': "You are a medical imaging analyst. Your job is to firstly provide a description of the image and then answer the question provided by the user with reference to the image"}
            ]
        },
        {'role': 'user',
         'content': [
             {'type': 'image'},
             {'type': 'text', 'text': f"Please describe what is shown in the image and answer the following query with reference to the image: '{text}'"}
            ]
        }
    ]

    input_text = vision_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = vision_processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to('cuda')

    return inputs

def transcribe_audio(audio_input):
    inputs = whisper_processor(audio_input, return_tensors='pt')
    with torch.no_grad():
        speech_out = whisper_model.generate(**inputs)
        transcription = whisper_processor.decode(speech_out[0], skip_special_tokens=True)
    return transcription
#endregion



#region RAG PIPELINE
class RAGPipeline:
    def __init__(self, user_text, vision_response, k=5):
        self.index = faiss.read_index(r'..\..\dataset\nhsInform\faiss_index.bin')
        with open(r'..\..\dataset\nhsInform\texts.json', "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        self.k = k
        self.query = self.embed_query(user_text, vision_response)
        self.results = self.search_faiss()
        self.context = self.format_rag_context()

    def embed_query(self, text: str, vision_response: str) -> str:
        text = text.strip()
        periods = ['.', '?', '!']
        if(text[-1] not in periods):
            text = text + '.'
        if(vision_response):
            return(text + " " + vision_response.strip())
    
        return(text)
    
    def search_faiss(self):
        query_embedding = rag_model.encode([self.query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_embedding, self.k)
        
        return [(self.texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
    
    def format_rag_context(self):
        context = "\n".join([f"Retrieved Info {i+1}: {res[0]}" for i, res in enumerate(self.results)])
        return context
#endregion



#region MODEL GENS
def gen_image_response():
    if user_image:
        vision_inputs = preprocess_image(user_image, user_text)
        with torch.no_grad():
            vision_outputs = vision_model.generate(**vision_inputs, max_new_tokens=128, use_cache=True)
            return vision_processor.decode(vision_outputs[0], skip_special_tokens=True)
    else:
        return None

def gen_final_response(inputs):
    gen_kwargs = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50
    }

    with torch.no_grad():
        outputs = text_model.generate(**inputs, **gen_kwargs)
        return text_processor.decode(outputs[0], skip_special_tokens=True)
#endregion



#region TTS Stuffs
def text_to_speech(text):
    if not text:
        return None
    tts = gTTS(text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.write_to_fp(fp)
        temp_filename = fp.name
    return temp_filename

def play_answer(final_response):
    audio_file = text_to_speech(final_response)
    return audio_file


#region MAIN
if __name__ == '__main__':
    if(user_audio):
        user_text = transcribe_audio(user_audio)
    
    vision_response = gen_image_response()

    if(rag_switch):
        retrieved_info = RAGPipeline(user_text, vision_response).context
        inputs = preprocess_text(user_text, vision_response, retrieved_info, rag_switch)
        final_response = gen_final_response(inputs)
    else:
        inputs = preprocess_text(user_text, vision_response, "", rag_switch)
        final_response = gen_final_response(inputs)

    audio_file = play_answer(final_response)
#endregion