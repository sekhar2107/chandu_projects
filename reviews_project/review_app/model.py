import warnings
warnings.filterwarnings("ignore")

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import time

SCRAPER_API_KEY = "405fda735bcddead083c3e548c4873f5"

client = OpenAI(

    base_url="https://router.huggingface.co/v1",
    api_key="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",  
)


def extract_reviews(url, max_attempts=3, max_review_blocks=50):
    for attempt in range(1, max_attempts + 1):
        try:
            scraper_url = (
                f"https://api.scraperapi.com?api_key={SCRAPER_API_KEY}"
                f"&url={url}&render=true"
            )
            resp = requests.get(scraper_url, timeout=90)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            reviews_list = []
            for tag in soup.find_all(["p", "span", "div"]):
                text = tag.get_text(strip=True)
                if len(text) > 40:
                    reviews_list.append(text)
                if len(reviews_list) >= max_review_blocks:
                    break

            if reviews_list:
                return "\n".join(reviews_list)

        except:
            pass

        time.sleep(2)

    return None


def summarize_reviews(reviews):
    MAX_LIMIT = 2000
    if len(reviews) > MAX_LIMIT:
        reviews = reviews[:MAX_LIMIT]

    prompt = f"""
Read all the reviews below and generate one clean paragraph.
Very important rules:
- Only one paragraph, no line breaks
- Do not use bullet points or headings like Overall sentiment or Main likes
- Do not list categories separately
- No special symbols like *, &, /, $, -, #
- Write naturally like a human summary
- Include: overall sentiment, what people liked, what they disliked, common issues, and final conclusion


Reviews:
{reviews}
"""

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content.strip()
