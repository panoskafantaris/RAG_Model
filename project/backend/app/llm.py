"""
Enhanced LLM handler optimized for Llama-Krikri-3B on 6GB GPU.
Simpler loading strategy since 3B model fits comfortably in VRAM.
"""
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import psutil

from .config import LLM_MODEL_NAME, LLM_CONFIG
from .exceptions import ModelLoadException
from .logger import setup_logger

logger = setup_logger(__name__)

# Global instances
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_pipe = None


def check_system_resources():
    """Check and log available system resources."""
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.2f}GB)")
    else:
        logger.info("No GPU detected, will use CPU")
    
    # Check RAM
    mem = psutil.virtual_memory()
    total_ram = mem.total / (1024**3)
    available_ram = mem.available / (1024**3)
    logger.info(f"System RAM: {total_ram:.1f}GB total, {available_ram:.1f}GB available")


def get_quantization_config():
    """
    Get 4-bit quantization config for memory efficiency.
    3B model with 4-bit quantization uses ~2-3GB VRAM.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def init_llm():
    """
    Initialize the LLM pipeline for 3B model.
    Optimized for 6GB GPU - model fits entirely in VRAM with quantization.
    
    Returns:
        Hugging Face pipeline
    
    Raises:
        ModelLoadException: If model fails to load
    """
    global _tokenizer, _model, _pipe
    
    if _pipe is not None:
        return _pipe
    
    try:
        logger.info(f"Loading LLM model: {LLM_MODEL_NAME}")
        check_system_resources()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        if _tokenizer is None:
            logger.info("Loading tokenizer...")
            _tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_NAME,
                trust_remote_code=True
            )
            logger.info("✓ Tokenizer loaded")
        
        # Load model
        if _model is None:
            if device == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                # Strategy based on GPU memory
                if gpu_memory >= 6:
                    # Use 4-bit quantization - fits 3B model in ~2-3GB
                    logger.info("Loading with 4-bit quantization for optimal memory usage")
                    
                    quantization_config = get_quantization_config()
                    
                    _model = AutoModelForCausalLM.from_pretrained(
                        LLM_MODEL_NAME,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                    )
                    logger.info("✓ Model loaded with 4-bit quantization")
                    
                elif gpu_memory >= 4:
                    # Use 8-bit quantization for smaller GPUs
                    logger.info("Loading with 8-bit quantization")
                    
                    _model = AutoModelForCausalLM.from_pretrained(
                        LLM_MODEL_NAME,
                        load_in_8bit=True,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    logger.info("✓ Model loaded with 8-bit quantization")
                    
                else:
                    # Use float16 for very small GPUs
                    logger.info("Loading with float16")
                    
                    _model = AutoModelForCausalLM.from_pretrained(
                        LLM_MODEL_NAME,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    )
                    logger.info("✓ Model loaded with float16")
                    
            else:
                # CPU loading
                logger.info("Loading on CPU (this will be slower)")
                
                _model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_NAME,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                logger.info("✓ Model loaded on CPU")
        
        # Create pipeline
        if _pipe is None:
            logger.info("Creating inference pipeline...")
            _pipe = pipeline(
                "text-generation",
                model=_model,
                tokenizer=_tokenizer,
                **LLM_CONFIG
            )
            logger.info("✓ Pipeline ready")
            
            # Log memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                logger.info(f"GPU Memory - Free: {(gpu_memory - allocated):.2f}GB")
        
        return _pipe
    
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check internet connection for model download")
        logger.error("2. Verify you have at least 4GB GPU VRAM available")
        logger.error("3. Try closing other GPU applications")
        logger.error("4. Check CUDA/PyTorch installation")
        raise ModelLoadException(LLM_MODEL_NAME, e)


def build_prompt(
    system_instruction: str,
    context: str,
    history: List[Dict[str, str]],
    user_query: str
) -> str:
    """
    Build a well-structured prompt for the LLM.
    
    Args:
        system_instruction: System-level instructions
        context: Retrieved documents context
        history: Conversation history
        user_query: Current user question
    
    Returns:
        Formatted prompt string
    """
    # Start with system instruction
    prompt_parts = [
        "<|system|>",
        system_instruction.strip(),
        "</|system|>\n"
    ]
    
    # Add conversation history (last 5 exchanges to avoid context overflow)
    if history:
        prompt_parts.append("<|history|>")
        # Take last 10 messages (5 exchanges)
        recent_history = history[-10:] if len(history) > 10 else history
        
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        
        prompt_parts.append("</|history|>\n")
    
    # Add retrieved context
    if context:
        prompt_parts.append("<|context|>")
        prompt_parts.append("Σχετικές πληροφορίες από την βάση γνώσης:")
        prompt_parts.append(context)
        prompt_parts.append("</|context|>\n")
    
    # Add current query
    prompt_parts.append("<|query|>")
    prompt_parts.append(f"User: {user_query}")
    prompt_parts.append("</|query|>\n")
    
    # Request response
    prompt_parts.append("Assistant:")
    
    return "\n".join(prompt_parts)


def generate_answer(prompt: str, max_retries: int = 2) -> str:
    """
    Generate an answer from the LLM with retry logic.
    Expected response time: 5-10 seconds with 3B model on GPU.
    
    Args:
        prompt: Formatted prompt
        max_retries: Number of retries on failure
    
    Returns:
        Generated text
    """
    for attempt in range(max_retries + 1):
        try:
            pipe = init_llm()
            
            logger.info(f"Generating response (attempt {attempt + 1}/{max_retries + 1})")
            
            # Clear CUDA cache before generation if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import time
            start_time = time.time()
            
            # Generate
            outputs = pipe(prompt)
            
            elapsed = time.time() - start_time
            logger.info(f"Generation completed in {elapsed:.1f} seconds")
            
            if not outputs or len(outputs) == 0:
                raise ValueError("Empty response from model")
            
            generated_text = outputs[0]["generated_text"]
            
            # Extract only the assistant's response
            if "Assistant:" in generated_text:
                answer = generated_text.split("Assistant:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            # Clean up the response
            answer = answer.strip()
            
            # Remove any trailing tags or artifacts
            if "</s>" in answer:
                answer = answer.split("</s>")[0].strip()
            
            logger.info(f"Response: {len(answer)} characters generated")
            
            return answer
        
        except Exception as e:
            logger.error(f"Generation attempt {attempt + 1} failed: {e}")
            
            # Clear cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if attempt == max_retries:
                logger.error("All generation attempts failed")
                return "Λυπάμαι, αντιμετώπισα πρόβλημα κατά τη δημιουργία της απάντησης. Παρακαλώ δοκιμάστε ξανά."
    
    return "Σφάλμα συστήματος."


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of tokens in text.
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    # Rough estimate: 1 token ≈ 4 characters for multilingual text
    return len(text) // 4


def clear_model_cache():
    """
    Clear model from memory. Useful for memory management.
    """
    global _tokenizer, _model, _pipe
    
    _tokenizer = None
    _model = None
    _pipe = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Model cache cleared")