"""Validate model predictions against exact runtime taxonomy paths."""


def validate_taxonomy_response(response: dict, taxonomy_paths: list[tuple[str, ...]], level_names: list[str]) -> dict:
    if response.get("status") == "uncertain":
        return {"valid": True, "path": (), "status": "uncertain"}
    values = []
    for i in range(4):
        value = response.get(f"l{i+1}")
        values.append(str(value).strip() if value not in (None, "") else "")
    depth = len(level_names)
    if any(values[i] for i in range(depth, 4)):
        return {"valid": False, "reason": "Prediction populated a taxonomy level that does not exist."}
    path = tuple(values[:depth])
    if not path or not path[0] or any(not value for value in path[:len(path) - 1]):
        return {"valid": False, "reason": "Prediction has a missing parent level."}
    valid_nodes = {tuple(candidate[:depth]) for candidate in taxonomy_paths
                   for depth in range(1, len(candidate) + 1)}
    if path not in valid_nodes:
        return {"valid": False, "reason": "Prediction is not an exact path in the runtime taxonomy."}
    return {"valid": True, "path": path, "status": "classified"}
