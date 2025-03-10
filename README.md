# F20CA - Medical CA w/ RAG

> This is the coursework for F20CA - Conversational Agents and Spoken Language Processing

> The coursework requires us to perform an experimental study using conversational agents in the Healthcare Domain

> The CA is a Medical chatbot that answers questions about common diseases. Specifically those listed in the NHS Inform Scot A-Z Common Illnesses and conditions webpage:
> https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/


 ## Research Question
 How does RAG effect the accuracy of a pre-trained LLM in accordance to a specific knowledge source?
 
## Introduction
The core goals of this project are:

- **Create and implement a tri-model system** that uses:
	- A finetuned LLM
	- A finetuned/base Speech Model
	- A finetuned VLM
- **Conduct an experimental study on the effects of RAG** on such a system and determine its use and benefit in the future of CAs in the Healthcare Domain in the future.
- **Report the process and evaluation methods** used to conduct the study defined above
- **Distribute workload and collaborate effectively** as a team in order to maintain an impressive workflow.

  

## Datasets

### NHS Inform Scot
> This  dataset will be the knowledge source for RAG, and the ground truth used to evaluate the system.

#### Structure
- **A singular CSV/JSON file**
- **Features:**
	- ***Disease:** The name of the disease/illness/condition*
	- ***Symptoms:** The symptoms as described on the webpage*
	- ***Treatments:** The recommended treatments advised by the NHS*
---
### MedQA
> This dataset will be used to finetune the LLM and Speech model to create a medical-specific knowledge base within the models

#### Structure
- **QA:** Contains JSONL files for each split (train, test and validation)
	- ***Features include:***
		- ***Question:** A question for the CA*
		- ***Answer:** The answer expected by the CA*
		- ***Options:** Multi-choice options for the CA - A, B, C or D (not implementing)*
		- ***Answer IDX:** The letter answer from the options (not implementing)*
- **Textbook:** Contains .txt files for different medical related topics. The purpose is to train the model to learn medical-specialised terms
---
### PMC-VQA
> This dataset will be used to finetune the VLM to create a model that can correctly identify medical-related images, such as MRI scans, X-Ray scans, etc.

#### Structure
- **Images:** Saved in a separate folder called images2.zip
- **Features:**
	- ***Figure_path:** File name of the image*
	- ***Question:** The question that will act as the user input for finetuning*
	- ***Answer:** The answer expected by the VLM*

## Project Directory
- **root/dataset**
	- **./finetune_ds**
		- *./med_qa*
	- **./nhsInform**
		- *./NHS_data.csv*
		- *./NHS_data.json*
- **root/directory**
	- **./finetune**
		- *./llm_finetune.ipynb*
		- *./vlm_finetune.ipynb*
		- *./whisper_finetune.ipynb*
	- **./RAG**
		- *./NHS_dataset_RAG_construction.ipynb*
	- **./final**
		- *./tri_model_wo_RAG.ipynb*
		- *./tri_model_w_RAG.ipynb*
- **root/original_nb**
	- *This folder stores the original completed model for reference when refactoring*

## Methodology

> This project implements a Retrieval-Augmented Generation (RAG) system tailored for the medical domain, specifically focusing on the NHS Inform Scotland dataset. The methodology is structured to provide accurate and contextually relevant medical information through a multimodal interface.

**1. Data Preparation and Retrieval:**

* **NHS Dataset Processing:** The NHS Inform Scotland dataset, stored in `NHS_Data.json`, was processed to create a retrieval index. The "Disease," "Symptoms," and "Treatments" fields were combined into coherent text chunks, forming the basis for our retrieval documents.
* **Embedding Generation:** A pre-trained BERT model (`all-MiniLM-L6-v2`) was utilized to generate dense vector embeddings for each text chunk. These embeddings capture the semantic meaning of the medical information.
* **FAISS Indexing:** The generated embeddings were indexed using FAISS (Facebook AI Similarity Search) to enable efficient and rapid similarity searches. This FAISS index acts as our vector database for retrieving relevant medical information.
* **Focused Retrieval:** The RAG system exclusively uses the NHS FAISS index, ensuring that all retrieved information originates from the NHS Inform Scotland dataset, aligning with the project's focus. We removed the other FAISS indexes that were being created for other knowledge sources.

**2. Model Selection and Fine-tuning:**

* **Text Model:** We selected the LLaMa 3.2 model, loaded using Unsloth for efficient 4-bit quantization, as our primary language model. This model was chosen for its strong generative capabilities and adaptability. Alternatively, the Qwen model can be used.
* **Fine-tuning Strategy:** The LLaMa 3.2 (or Qwen) model was fine-tuned on a combination of medical question-answer data (MedQA) and medical textbook data (MedQA). This fine-tuning process adapts the model to the medical domain, enhancing its ability to generate accurate and relevant responses. The NHS dataset was intentionally excluded from the fine-tuning process to reserve it solely for RAG.
* **Multimodal Integration:**
    * **Visual Input:** The Qwen-VL model was integrated to process image inputs. It generates textual descriptions of images, which are then combined with the user's text query.
    * **Speech Input:** The Whisper model was used for speech-to-text transcription. This allows users to interact with the system using voice commands.
* **Embedding of Visual Input:** When a user uploads an image, the Qwen-VL model generates a caption. This caption is then embedded into the users text input.

**3. RAG Pipeline:**

* **User Input Processing:** The system processes user inputs, which can be text, speech, or images. Speech inputs are transcribed using Whisper, and image inputs are captioned using Qwen-VL.
* **Combined Input:** The caption generated by Qwen-VL is combined with the user's text or transcribed speech input.
* **Retrieval Augmentation:** The combined user input is used as a query to retrieve relevant medical information from the NHS FAISS index.
* **Prompt Construction:** A prompt is constructed, incorporating the retrieved information, the combined user input, and instructions for the language model.
* **Response Generation:** The fine-tuned LLaMa 3.2 (or Qwen) model generates a response based on the constructed prompt.
* **Output Display:** The generated response is displayed to the user through a Gradio interface.

**4. System Optimization:**

* **Performance Tuning:** The system's performance was optimized by fine-tuning prompts, adjusting retrieval parameters, and selecting appropriate model configurations.
* **Multimodal Integration:** The integration of visual and speech inputs was carefully designed to ensure seamless interaction and accurate information processing.

**5. Evaluation:**

* *Future work will include evaluation of the system's performance using relevant medical benchmarks and user feedback.*

This methodology ensures that the RAG system provides accurate, contextually relevant, and multimodal medical information, grounded in the NHS Inform Scotland dataset.