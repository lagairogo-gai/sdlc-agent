import asyncio
from typing import Dict, Any, Optional, List
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import GooglePalm
from langchain.schema import HumanMessage, SystemMessage, AIMessage
import logging
import openai
import anthropic
import google.generativeai as genai
from openai import AzureOpenAI
import tiktoken

logger = logging.getLogger(__name__)

class LLMService:
    """Multi-provider LLM service supporting OpenAI, Anthropic, Google, and Azure"""
    
    def __init__(self):
        self.providers = {}
        self.current_provider = None
        self.current_model = None
        self.current_config = {}
        
    def configure(self, config: Dict[str, Any]):
        """Configure LLM provider and model"""
        self.current_config = config
        provider = config.get("provider", "openai")
        model = config.get("model", "gpt-3.5-turbo")
        
        self.current_provider = provider
        self.current_model = model
        
        try:
            if provider == "openai":
                self._configure_openai(config)
            elif provider == "anthropic":
                self._configure_anthropic(config)
            elif provider == "google":
                self._configure_google(config)
            elif provider == "azure":
                self._configure_azure(config)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
            logger.info(f"Configured LLM: {provider} - {model}")
            
        except Exception as e:
            logger.error(f"Error configuring LLM provider {provider}: {e}")
            raise e
    
    def _configure_openai(self, config: Dict[str, Any]):
        """Configure OpenAI provider"""
        api_key = config.get("api_key") or config.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        model = config.get("model", "gpt-3.5-turbo")
        
        if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
            self.providers["openai"] = ChatOpenAI(
                openai_api_key=api_key,
                model_name=model,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2000),
                request_timeout=config.get("timeout", 60)
            )
        else:
            self.providers["openai"] = OpenAI(
                openai_api_key=api_key,
                model_name=model,
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2000),
                request_timeout=config.get("timeout", 60)
            )
    
    def _configure_anthropic(self, config: Dict[str, Any]):
        """Configure Anthropic Claude provider"""
        api_key = config.get("api_key") or config.get("anthropic_api_key")
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        model = config.get("model", "claude-3-sonnet-20240229")
        
        self.providers["anthropic"] = ChatAnthropic(
            anthropic_api_key=api_key,
            model=model,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            timeout=config.get("timeout", 60)
        )
    
    def _configure_google(self, config: Dict[str, Any]):
        """Configure Google Gemini provider"""
        api_key = config.get("api_key") or config.get("google_api_key")
        if not api_key:
            raise ValueError("Google API key is required")
        
        genai.configure(api_key=api_key)
        model = config.get("model", "gemini-pro")
        
        # Custom Google implementation since LangChain might not have latest Gemini
        self.providers["google"] = {
            "model": genai.GenerativeModel(model),
            "config": {
                "temperature": config.get("temperature", 0.7),
                "max_output_tokens": config.get("max_tokens", 2000),
                "top_p": config.get("top_p", 0.8),
                "top_k": config.get("top_k", 40)
            }
        }
    
    def _configure_azure(self, config: Dict[str, Any]):
        """Configure Azure OpenAI provider"""
        api_key = config.get("api_key") or config.get("azure_openai_api_key")
        endpoint = config.get("endpoint") or config.get("azure_openai_endpoint")
        
        if not api_key or not endpoint:
            raise ValueError("Azure OpenAI API key and endpoint are required")
        
        deployment_name = config.get("deployment_name") or config.get("model", "gpt-35-turbo")
        
        self.providers["azure"] = ChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=endpoint,
            openai_api_type="azure",
            openai_api_version="2023-05-15",
            deployment_name=deployment_name,
            model_name=deployment_name,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 2000),
            request_timeout=config.get("timeout", 60)
        )
    
    async def generate(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text using the configured LLM"""
        if not self.current_provider or self.current_provider not in self.providers:
            raise ValueError("No LLM provider configured")
        
        try:
            provider = self.providers[self.current_provider]
            
            # Override config if provided
            if temperature is not None:
                provider.temperature = temperature
            if max_tokens is not None:
                provider.max_tokens = max_tokens
            
            if self.current_provider == "google":
                return await self._generate_google(prompt, system_message)
            elif hasattr(provider, 'agenerate'):
                # Chat models
                messages = []
                if system_message:
                    messages.append(SystemMessage(content=system_message))
                messages.append(HumanMessage(content=prompt))
                
                response = await provider.agenerate([messages])
                return response.generations[0][0].text
            else:
                # Completion models
                full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
                response = await provider.agenerate([full_prompt])
                return response.generations[0][0].text
                
        except Exception as e:
            logger.error(f"Error generating text with {self.current_provider}: {e}")
            raise e
    
    async def _generate_google(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate text using Google Gemini"""
        try:
            model_data = self.providers["google"]
            model = model_data["model"]
            config = model_data["config"]
            
            full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
            
            response = await asyncio.to_thread(
                model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(**config)
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error with Google Gemini: {e}")
            raise e
    
    async def generate_stream(
        self, 
        prompt: str, 
        system_message: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Generate text with streaming response"""
        if not self.current_provider or self.current_provider not in self.providers:
            raise ValueError("No LLM provider configured")
        
        provider = self.providers[self.current_provider]
        
        try:
            if self.current_provider == "google":
                # Google streaming implementation
                model_data = self.providers["google"]
                model = model_data["model"]
                config = model_data["config"]
                
                full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
                
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(**config),
                    stream=True
                )
                
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
            
            elif hasattr(provider, 'astream'):
                # Chat models with streaming
                messages = []
                if system_message:
                    messages.append(SystemMessage(content=system_message))
                messages.append(HumanMessage(content=prompt))
                
                async for chunk in provider.astream(messages):
                    if hasattr(chunk, 'content'):
                        yield chunk.content
                    else:
                        yield str(chunk)
            else:
                # Fallback to non-streaming
                result = await self.generate(prompt, system_message)
                yield result
                
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise e
    
    async def generate_structured(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured output using JSON schema"""
        structured_prompt = f"""
        {system_message or ''}
        
        Please respond with a JSON object that follows this exact schema:
        {schema}
        
        User Request: {prompt}
        
        Response (JSON only):
        """
        
        response = await self.generate(structured_prompt.strip())
        
        # Try to parse JSON response
        import json
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback: extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse structured response: {response}")
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text for the current model"""
        try:
            if self.current_provider == "openai" or self.current_provider == "azure":
                # Use tiktoken for OpenAI models
                encoding_name = "cl100k_base"  # GPT-4, GPT-3.5-turbo
                if "gpt-3.5" in self.current_model:
                    encoding_name = "cl100k_base"
                elif "gpt-4" in self.current_model:
                    encoding_name = "cl100k_base"
                
                encoding = tiktoken.get_encoding(encoding_name)
                return len(encoding.encode(text))
            
            elif self.current_provider == "anthropic":
                # Approximate token count for Claude (1 token ≈ 4 characters)
                return len(text) // 4
            
            elif self.current_provider == "google":
                # Approximate token count for Gemini
                return len(text.split())
            
            else:
                # Generic approximation
                return len(text.split())
                
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text.split())  # Fallback to word count
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.current_provider:
            return {"error": "No provider configured"}
        
        model_info = {
            "provider": self.current_provider,
            "model": self.current_model,
            "config": self.current_config
        }
        
        # Add provider-specific info
        if self.current_provider == "openai":
            model_info.update({
                "max_tokens": 4096 if "gpt-3.5" in self.current_model else 8192,
                "supports_functions": True,
                "supports_streaming": True
            })
        
        elif self.current_provider == "anthropic":
            model_info.update({
                "max_tokens": 100000,  # Claude-3 context length
                "supports_functions": False,
                "supports_streaming": True
            })
        
        elif self.current_provider == "google":
            model_info.update({
                "max_tokens": 30720,  # Gemini Pro context length
                "supports_functions": True,
                "supports_streaming": True
            })
        
        elif self.current_provider == "azure":
            model_info.update({
                "max_tokens": 4096,
                "supports_functions": True,
                "supports_streaming": True
            })
        
        return model_info
    
    async def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate LLM configuration"""
        provider = config.get("provider")
        if not provider:
            return {"valid": False, "error": "Provider is required"}
        
        if provider not in ["openai", "anthropic", "google", "azure"]:
            return {"valid": False, "error": f"Unsupported provider: {provider}"}
        
        # Check required fields
        required_fields = {
            "openai": ["api_key", "model"],
            "anthropic": ["api_key", "model"],
            "google": ["api_key", "model"],
            "azure": ["api_key", "endpoint", "deployment_name"]
        }
        
        missing_fields = []
        for field in required_fields[provider]:
            if not config.get(field) and not config.get(f"{provider}_{field}"):
                missing_fields.append(field)
        
        if missing_fields:
            return {
                "valid": False, 
                "error": f"Missing required fields for {provider}: {missing_fields}"
            }
        
        # Validate model names
        valid_models = {
            "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo"],
            "anthropic": ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
            "google": ["gemini-pro", "gemini-pro-vision"],
            "azure": []  # Azure uses deployment names
        }
        
        model = config.get("model")
        if provider != "azure" and model not in valid_models[provider]:
            return {
                "valid": False,
                "error": f"Invalid model for {provider}: {model}. Valid models: {valid_models[provider]}"
            }
        
        return {"valid": True}
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to the configured LLM provider"""
        if not self.current_provider:
            return {"success": False, "error": "No provider configured"}
        
        try:
            # Simple test prompt
            test_prompt = "Respond with 'Connection successful' if you can read this."
            response = await self.generate(test_prompt)
            
            return {
                "success": True,
                "provider": self.current_provider,
                "model": self.current_model,
                "response_length": len(response),
                "test_response": response[:100] + "..." if len(response) > 100 else response
            }
            
        except Exception as e:
            return {
                "success": False,
                "provider": self.current_provider,
                "model": self.current_model,
                "error": str(e)
            }
    
    async def get_available_models(self, provider: str) -> List[str]:
        """Get list of available models for a provider"""
        models = {
            "openai": [
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4-32k"
            ],
            "anthropic": [
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229",
                "claude-3-haiku-20240307"
            ],
            "google": [
                "gemini-pro",
                "gemini-pro-vision"
            ],
            "azure": [
                "gpt-35-turbo",
                "gpt-35-turbo-16k",
                "gpt-4",
                "gpt-4-32k"
            ]
        }
        
        return models.get(provider, [])
    
    def get_provider_capabilities(self, provider: str) -> Dict[str, Any]:
        """Get capabilities of a specific provider"""
        capabilities = {
            "openai": {
                "streaming": True,
                "function_calling": True,
                "vision": False,
                "max_context": 16385,
                "pricing_tier": "medium"
            },
            "anthropic": {
                "streaming": True,
                "function_calling": False,
                "vision": False,
                "max_context": 200000,
                "pricing_tier": "high"
            },
            "google": {
                "streaming": True,
                "function_calling": True,
                "vision": True,
                "max_context": 30720,
                "pricing_tier": "low"
            },
            "azure": {
                "streaming": True,
                "function_calling": True,
                "vision": False,
                "max_context": 16385,
                "pricing_tier": "medium"
            }
        }
        
        return capabilities.get(provider, {})
    
    async def estimate_cost(self, prompt: str, provider: str, model: str) -> Dict[str, Any]:
        """Estimate the cost of a request"""
        token_count = await self.count_tokens(prompt)
        
        # Approximate pricing (as of 2024 - should be updated regularly)
        pricing = {
            "openai": {
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03}
            },
            "anthropic": {
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
            },
            "google": {
                "gemini-pro": {"input": 0.0005, "output": 0.0015}
            },
            "azure": {
                "gpt-35-turbo": {"input": 0.0015, "output": 0.002},
                "gpt-4": {"input": 0.03, "output": 0.06}
            }
        }
        
        model_pricing = pricing.get(provider, {}).get(model)
        if not model_pricing:
            return {"error": f"Pricing not available for {provider}/{model}"}
        
        input_cost = (token_count / 1000) * model_pricing["input"]
        # Estimate output tokens (typically 10-20% of input for user stories)
        estimated_output_tokens = token_count * 0.15
        output_cost = (estimated_output_tokens / 1000) * model_pricing["output"]
        
        return {
            "input_tokens": token_count,
            "estimated_output_tokens": int(estimated_output_tokens),
            "input_cost_usd": round(input_cost, 6),
            "estimated_output_cost_usd": round(output_cost, 6),
            "total_estimated_cost_usd": round(input_cost + output_cost, 6)
        }