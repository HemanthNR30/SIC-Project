from openai import OpenAI
client = OpenAI()

print(
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":"Hello"}]
    ).choices[0].message.content
)
