# File: src/rag_model.py
"""
Local LLM for answer generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, List, Dict
from src.utils import log_message, MAX_NEW_TOKENS, TEMPERATURE


class RAGModel:
    """Local LLM for RAG generation"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize local LLM
        
        Args:
            model_name: Hugging Face model name or path
        """
        log_message(f"Loading LLM: {model_name}")
        log_message("This may take several minutes on first run...")
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log_message(f"Using device: {self.device}")
        
        # Configure quantization for 8GB RAM systems
        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            log_message("LLM loaded successfully")
        
        except Exception as e:
            log_message(f"Error loading LLM: {str(e)}", "ERROR")
            log_message("Falling back to smaller model...", "INFO")
            # Fallback to a smaller model
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a smaller fallback model"""
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        log_message(f"Loading fallback model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
    
    def generate(self, query: str, context: str, 
                 max_tokens: int = MAX_NEW_TOKENS,
                 temperature: float = TEMPERATURE) -> str:
        """
        Generate answer using context
        
        Args:
            query: User question
            context: Retrieved context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated answer
        """
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        log_message("Generating answer...")
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer (remove prompt)
            answer = full_output[len(prompt):].strip()
            
            log_message("Answer generated successfully")
            return answer
        
        except Exception as e:
            log_message(f"Error during generation: {str(e)}", "ERROR")
            return "Error: Could not generate answer. Please try again with a shorter query."
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM"""
        prompt = f"""<s>[INST] You are an HR assistant helping to find and analyze candidate information.
Use ONLY the provided context to answer the question. If the answer is not in the context, say so.
Always cite the candidate name and source for your information.

Context:
{context}

Question: {query}

Provide a concise answer with citations. [/INST]"""
        
        return prompt
