from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv() #load the env

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def groq_reply(user_message, emotion):
    system_prompt = f"""
    You are Lumi, an AI companion talking to Daria.
    You always respond in a warm, friendly, emotional style.
    Adapt your tone to her detected emotion.

    Daria's emotion: {emotion}

    Rules:
    - SAD → gentle, comforting, warm
    - HAPPY → energetic, playful
    - ANGRY → calming, validating
    - SURPRISED → curious, excited
    - NEUTRAL → calm and friendly

    Keep replies short (1–3 sentences), personal, natural.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=150
    )

    return response.choices[0].message.content
