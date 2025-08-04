@allocation_router.put(
    "/{mixture}/{name}",
    response_model=Allocation,
    dependencies=[Depends(check_operator_permission)],
)
def update_allocation(mixture: str, name: str, allocation: Allocation):
    key = f"{REDIS_PREFIX}:shirt:{mixture}:{name}"
    data: str | None = redis_client.get(key)  # type: ignore
    if not data:
        raise HTTPException(status_code=404, detail="Allocation not found")

    # Get existing shirt data and update the quota
    shirt_data = json.loads(data)
    shirt_data["quota_hard_threshold"] = allocation.allocation

    # Save updated shirt back to Redis
    redis_client.set(key, json.dumps(shirt_data))

    return allocation