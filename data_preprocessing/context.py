import json


def load_context(token, timeframe):
    with open("context.json", "r") as f:
        context=json.loads(json.load(f))

    return context[token][timeframe]
