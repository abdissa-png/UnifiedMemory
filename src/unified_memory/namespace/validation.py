"""
Namespace validation logic.

Ensures namespace identifiers follow the canonical format:
tenant:{id}/user:{id}[/agent:{id}][/session:{id}]
"""

import re
from typing import List

# Allow alphanumeric, underscore, hyphen, dot
ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

def validate_namespace_id(namespace_id: str) -> None:
    """
    Validate namespace identifier format.
    
    Raises ValueError if invalid.
    
    Valid format examples:
    - tenant:default/user:alice
    - tenant:acme/user:bob/agent:researcher
    - tenant:acme/user:bob/agent:researcher/session:123
    """
    if not namespace_id:
        raise ValueError("Namespace ID cannot be empty")
        
    parts = namespace_id.split("/")
    
    # Must have at least tenant and user
    if len(parts) < 2:
        raise ValueError("Namespace must minimally contain tenant and user components")
        
    # Check prefixes and order
    # 1. Tenant
    if not parts[0].startswith("tenant:"):
        raise ValueError("First component must be 'tenant:'")
    validate_component_id(parts[0][7:], "tenant_id")
    
    # 2. User
    if not parts[1].startswith("user:"):
        raise ValueError("Second component must be 'user:'")
    validate_component_id(parts[1][5:], "user_id")
    
    # 3. Optional components (strict order: agent -> session)
    if len(parts) > 2:
        current_idx = 2
        
        # Optional Agent
        if parts[current_idx].startswith("agent:"):
            validate_component_id(parts[current_idx][6:], "agent_id")
            current_idx += 1
            
        # Optional Session
        if current_idx < len(parts):
            if parts[current_idx].startswith("session:"):
                validate_component_id(parts[current_idx][8:], "session_id")
                current_idx += 1
            else:
                 # Found something that isn't session but expected specific order
                 pass
                 
        # If we still have parts, they are unknown or out of order
        if current_idx < len(parts):
             raise ValueError(f"Unexpected component or invalid order at: {parts[current_idx]}")

def validate_component_id(component_id: str, field_name: str) -> None:
    """
    Validate individual ID component.
    
    Allowed: Alphanumeric, underscore, hyphen, dot.
    """
    if not component_id:
        raise ValueError(f"{field_name} cannot be empty")
        
    if not ID_PATTERN.match(component_id):
        raise ValueError(f"{field_name} contains invalid characters. Allowed: a-z, A-Z, 0-9, _, -, .")
