"""
Enhanced LLM handler with proper error handling and prompt management.
"""
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from .config import LLM_MODEL_NAME, LLM_CONFIG
from .exceptions import ModelLoadException
from .logger import setup_logger

logger = setup_logger(__name__)

# Global instances
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_pipe = None


def init_llm():
    """
    Initialize the LLM pipeline with GPU support.
    
    Returns:
        Hugging Face pipeline
    
    Raises:
        ModelLoadException: If model fails to load
    """
    global _tokenizer, _model, _pipe
    
    if _pipe is not None:
        return _pipe
    
    try:
        from .config import DEVICE, GPU_MEMORY_FRACTION
        import torch
        
        logger.info(f"Loading LLM model: {LLM_MODEL_NAME}")
        logger.info(f"Target device: {DEVICE}")
        
        # Load tokenizer
        if _tokenizer is None:
            _tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL_NAME,
                trust_remote_code=True
            )
            logger.info("Tokenizer loaded")
        
        # Load model with GPU optimization
        if _model is None:
            # Determine dtype based on device
            if DEVICE == "cuda":
                dtype = torch.float16  # Use half precision on GPU for speed
                logger.info("Using float16 (half precision) for GPU inference")
            else:
                dtype = torch.float32
                logger.info("Using float32 for CPU inference")
            
            logger.info(f"Loading model on {DEVICE} with dtype {dtype}")
            
            # GPU-specific optimizations
            if DEVICE == "cuda":
                _model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_NAME,
                    device_map="auto",  # Automatically distribute model across GPUs
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: f"{GPU_MEMORY_FRACTION}GiB"}  # Limit GPU memory
                )
            else:
                _model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_NAME,
                    device_map="auto",
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            logger.info("Model loaded successfully")
            
            # Print memory usage if on GPU
            if DEVICE == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Create pipeline
        if _pipe is None:
            _pipe = pipeline(
                "text-generation",
                model=_model,
                tokenizer=_tokenizer,
                **LLM_CONFIG
            )
            logger.info("Pipeline created successfully")
        
        return _pipe
    
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
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
            
            # Generate
            outputs = pipe(prompt)
            
            if not outputs or len(outputs) == 0:
                raise ValueError("Empty response from model")
            
            generated_text = outputs[0]["generated_text"]
            
            # Extract only the assistant's response
            # Remove the prompt part
            if "Assistant:" in generated_text:
                answer = generated_text.split("Assistant:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            # Clean up the response
            answer = answer.strip()
            
            # Remove any trailing tags or artifacts
            if "</s>" in answer:
                answer = answer.split("</s>")[0].strip()
            
            logger.info(f"Response generated successfully ({len(answer)} chars)")
            return answer
        
        except Exception as e:
            logger.error(f"Generation attempt {attempt + 1} failed: {e}")
            
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