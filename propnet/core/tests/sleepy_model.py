import time


def plug_in(symbol_values):
    time.sleep(2)
    return {
        'B': 0
    }


DESCRIPTION = """
Test model which takes a long time to evaluate.
"""
config = {
    "name": "sleepy",
    "connections": [
        {
            "inputs": [
                "A"
            ],
            "outputs": [
                "B"
            ]
        }
    ],
    "categories": [
        "test"
    ],
    "symbol_property_map": {
        "A": "A",
        "B": "B"
    },
    "description": DESCRIPTION,
    "references": [],
    "implemented_by": [
        "clegaspi"
    ],
    "plug_in": plug_in
}
