def validate_records(records, schema):
    """
    Validate records against a schema definition.
    """
    results = []

    for idx, record in enumerate(records):
        errors = []

        for field in schema:
            col = field["column"]
            expected = field["type"]
            nullable = field["nullable"]

            # Missing column
            if col not in record:
                errors.append(f"{col}: missing")
                continue

            value = record[col]

            # Null check
            if value is None:
                if not nullable:
                    errors.append(f"{col}: null")
                continue

            # Type check
            valid_type = False
            if expected == "int":
                valid_type = (type(value) is int)
            elif expected == "float":
                valid_type = (type(value) is int or type(value) is float)
            elif expected == "str":
                valid_type = (type(value) is str)

            if not valid_type:
                errors.append(f"{col}: expected {expected}, got {type(value).__name__}")
                continue

            # Range check (only for numeric values)
            if expected in ("int", "float"):
                if "min" in field and value < field["min"]:
                    errors.append(f"{col}: out of range")
                    continue
                if "max" in field and value > field["max"]:
                    errors.append(f"{col}: out of range")
                    continue

        results.append((idx, len(errors) == 0, errors))

    return results