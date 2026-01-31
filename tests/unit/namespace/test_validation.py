import pytest
from unified_memory.namespace.validation import validate_namespace_id

def test_valid_namespaces():
    # Canonical valid examples
    validate_namespace_id("tenant:default/user:alice")
    validate_namespace_id("tenant:acme/user:bob")
    validate_namespace_id("tenant:acme/user:bob/agent:researcher")
    validate_namespace_id("tenant:acme/user:bob/agent:researcher/session:123")
    validate_namespace_id("tenant:acme/user:bob/session:xyz-123") # order: agent is optional

def test_invalid_namespaces_empty_or_short():
    with pytest.raises(ValueError, match="Namespace ID cannot be empty"):
        validate_namespace_id("")
        
    with pytest.raises(ValueError, match="must minimally contain"):
        validate_namespace_id("tenant:only")

def test_invalid_prefixes():
    with pytest.raises(ValueError, match="First component must be 'tenant:'"):
        validate_namespace_id("user:alice/tenant:default")
        
    with pytest.raises(ValueError, match="Second component must be 'user:'"):
        validate_namespace_id("tenant:default/agent:bob")

def test_invalid_characters():
    with pytest.raises(ValueError, match="tenant_id contains invalid characters"):
        validate_namespace_id("tenant:def@ult/user:alice")
        
    with pytest.raises(ValueError, match="user_id contains invalid characters"):
        validate_namespace_id("tenant:default/user:ali ce")

def test_invalid_order_or_unknown():
    # Session before agent (if we enforce strict order for optional parts?)
    # My implementation was strict: Agent -> Session. 
    # If session comes first, it might fail if it doesn't match agent check or if logic expects agent first.
    # Logic: 
    #   check agent prefix. If matches, consume. 
    #   Then check session.
    #   If "session:..." is encountered where agent expected?
    #   Code: if parts[2] starts with "agent:" -> consume.
    #         if parts[2] starts with "session:" -> consume (as session).
    # So skipping agent is allowed.
    
    # Unknown component
    with pytest.raises(ValueError, match="Unexpected component"):
        validate_namespace_id("tenant:a/user:b/unknown:c")
