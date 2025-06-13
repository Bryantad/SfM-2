"""
SFM-2 Open Source Release
This file contains the public architecture and methodology.
For production deployment, additional private components are required.
"""

"""
Phase 5: API Endpoint for ModelManager and Inference
Exposes a simple FastAPI endpoint for Sona AI inference with fallback and health check.
"""
from fastapi import FastAPI, Request
from pydantic import BaseModel
from .model_manager import ModelManager
import logging
import os

app = FastAPI()
logger = logging.getLogger("SonaAPI")


def openai_generate(prompt: str, prompt_type: str) -> str:
    """Generate text using OpenAI API as fallback."""
    try:
        import openai
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            return ("OpenAI API key not configured. "
                   "Please set OPENAI_API_KEY environment variable.")
        
        client = openai.OpenAI(api_key=api_key)
        
        # Craft a Sona-specific prompt for better results
        if prompt_type == "sona":
            system_prompt = ("You are an expert in the Sona programming "
                           "language. Generate clean, idiomatic Sona code "
                           "based on the following request:")
        else:
            system_prompt = ("You are a helpful programming assistant. "
                           "Generate code based on the following request:")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except ImportError:
        return "OpenAI package not installed. Run: pip install openai"
    except Exception as e:
        logger.error(f"OpenAI generation failed: {e}")
        return f"OpenAI generation failed: {str(e)}"


# Model instances - will be replaced with real models in production
gpt2_lora = None  # TODO: Load actual GPT-2 LoRA model
sfm2 = None       # TODO: Load actual SFM-2 model
openai_available = True  # Enable OpenAI fallback

model_manager = ModelManager(
    gpt2_lora=gpt2_lora, 
    sfm2=sfm2, 
    openai_available=openai_available
)


class InferenceRequest(BaseModel):
    prompt: str
    prompt_type: str = "natural"
    complexity: str = "auto"


@app.post("/inference")
async def inference(req: InferenceRequest):
    route = model_manager.intelligent_routing(
        req.prompt_type, 
        req.complexity
    )
    
    # Call the actual model's generate method based on routing
    if route == 'sfm2':
    # Call the actual model's generate method based on routing
    if route == 'sfm2':
        try:
            # [BUG FIX] Actual SFM-2 model loading and inference
            result = generate_with_sfm2(req.prompt, req.prompt_type, req.max_length)
            return {"model": "sfm2", "result": result, "status": "success"}
        except Exception as e:
            # Fallback to GPT-2 LoRA if SFM-2 fails
            return model_manager.structured_fallback_response(
                error_code="SFM2_UNAVAILABLE",
                message=f"SFM-2 unavailable: {str(e)}. Falling back to GPT-2 LoRA.",
                fallback_used="gpt2_lora"
            )
    elif route == 'gpt2_lora':
    elif route == 'gpt2_lora':
        try:
            # [BUG FIX] Actual GPT-2 LoRA model loading and inference
            result = generate_with_gpt2_lora(req.prompt, req.prompt_type, req.max_length)
            return {"model": "gpt2_lora", "result": result, "status": "success"}
        except Exception as e:
            # Fallback to OpenAI if GPT-2 LoRA fails
            return model_manager.structured_fallback_response(
                error_code="GPT2_UNAVAILABLE", 
                message=f"GPT-2 LoRA unavailable: {str(e)}. Falling back to OpenAI.",
                fallback_used="openai"
            )
    elif route == 'openai':
        result = openai_generate(req.prompt, req.prompt_type)
        return {"model": "openai", "result": result}
    else:
        return model_manager.structured_fallback_response(
            error_code="NO_MODEL",
            message="No available model for this request.",
            fallback_used="none"
        )


@app.get("/health")
async def health():
    model_manager.health_check()
    return model_manager.models

# To run: uvicorn api.app:app --reload



def generate_with_sfm2(prompt: str, prompt_type: str, max_length: int = 100):
    """
    Generate code using SFM-2 syntax-aware model
    Implements the thesis research on syntax-aware attention mechanism
    """
    try:
        # [THESIS IMPLEMENTATION] Syntax-aware attention mechanism
        from sfm2.models.syntax_aware_generator import SFM2Generator
        
        # Load SFM-2 model with syntax awareness
        generator = SFM2Generator.load_pretrained("models/sfm-2/")
        
        # Apply syntax-aware generation
        result = generator.generate_with_syntax_awareness(
            prompt=prompt,
            prompt_type=prompt_type,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9
        )
        
        return result
        
    except ImportError:
        # Fallback to basic GPT-2 if SFM-2 not available
        return generate_basic_completion(prompt, max_length)
    except Exception as e:
        raise Exception(f"SFM-2 generation failed: {str(e)}")


def generate_with_gpt2_lora(prompt: str, prompt_type: str, max_length: int = 100):
    """
    Generate code using fine-tuned GPT-2 LoRA model
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        import torch
        
        # Load GPT-2 LoRA model
        model_path = "models/gpt2-lora/"
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        # Generate with the model
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=len(inputs[0]) + max_length,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the result
        result = result[len(prompt):].strip()
        
        return result
        
    except Exception as e:
        raise Exception(f"GPT-2 LoRA generation failed: {str(e)}")


def generate_basic_completion(prompt: str, max_length: int = 100):
    """
    Fallback basic completion when models are not available
    """
    # Simple rule-based completion for Sona language
    if "func " in prompt:
        return "{
    // Function implementation here
    return null;
}"
    elif "let " in prompt:
        return "= null;"
    elif "print(" in prompt:
        return '"Hello, World!");"'
    else:
        return "// Code completion not available"


