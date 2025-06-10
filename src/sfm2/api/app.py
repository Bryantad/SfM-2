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
from api.model_manager import ModelManager
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
        # TODO: result = sfm2.generate(req.prompt, req.prompt_type)
        return {"model": "sfm2", "result": "[SFM-2 not loaded yet]"}
    elif route == 'gpt2_lora':
        # TODO: result = gpt2_lora.generate(req.prompt, req.prompt_type)
        return {"model": "gpt2_lora", "result": "[GPT-2 LoRA not loaded yet]"}
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
