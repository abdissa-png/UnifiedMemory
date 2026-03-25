"""JSON validation and repair utilities."""

import json
import re
from typing import Any, Dict, List, Optional
from json_repair import repair_json


class JSONValidationError(Exception):
    """Custom exception for JSON validation errors."""
    pass


def clean_json_response(response_text: str) -> str:
    """Clean and extract JSON from LLM response text."""
    # Remove markdown code blocks
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*$', '', response_text)
    
    # Remove leading/trailing whitespace
    response_text = response_text.strip()
    
    # Try to extract JSON from text if it's embedded
    json_pattern = r'\{.*\}'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        response_text = match.group(0)
    
    return response_text


def validate_json_structure(data: Dict[str, Any], expected_keys: List[str]) -> bool:
    """Validate that JSON contains expected keys."""
    if not isinstance(data, dict):
        return False
    
    for key in expected_keys:
        if key not in data:
            return False
    
    return True


def validate_and_repair_json(
    response_text: str, 
    expected_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate and repair JSON from LLM response using json_repair library.
    
    Args:
        response_text: Raw response text from LLM
        expected_keys: List of required keys to validate
        
    Returns:
        Parsed and validated JSON dictionary
        
    Raises:
        JSONValidationError: If JSON cannot be parsed or validated
    """
    if not response_text or not response_text.strip():
        raise JSONValidationError("Empty response text provided")
    
    # Clean the response text
    cleaned_text = clean_json_response(response_text)
    
    try:
        # First try direct parsing
        data = json.loads(cleaned_text)
    except (json.JSONDecodeError, ValueError):
        try:
            # Use json-repair library for repair
            repaired_text = repair_json(cleaned_text)
            data = json.loads(repaired_text)
        except (json.JSONDecodeError, ValueError) as e:
            raise JSONValidationError(
                f"Failed to parse JSON. Error: {str(e)}. "
                f"Original text: {response_text[:200]}..."
            )
    
    # Validate structure if expected keys provided
    if expected_keys and not validate_json_structure(data, expected_keys):
        missing_keys = [key for key in expected_keys if key not in data]
        raise JSONValidationError(f"Missing required keys: {missing_keys}")
    
    return data
