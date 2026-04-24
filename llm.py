import subprocess

def generate_answer(query, context):
    prompt = f"""
    Answer only from the context.

    Context:
    {context}

    Question:
    {query}
    """

    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt,
        text=True,
        capture_output=True
    )

    return result.stdout.strip()