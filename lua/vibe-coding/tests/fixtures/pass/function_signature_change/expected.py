def update_allocation(mixture: str, name: str, allocation: str):
    """Update allocation for a mixture resource"""
    key = f"{REDIS_PREFIX}:shirt:{mixture}:{name}"
    data: str | None = redis_client.get(key)
    
    if data is None:
        raise ValueError(f"No allocation found for {mixture}/{name}")
    
    # Update the allocation
    redis_client.set(key, allocation.to_json())
    return True