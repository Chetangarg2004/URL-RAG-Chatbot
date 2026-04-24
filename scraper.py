import requests
from bs4 import BeautifulSoup

def scrape_urls(urls):
    all_text = ""

    for url in urls:
        try:
            res = requests.get(url.strip())
            soup = BeautifulSoup(res.content, "html.parser")

            for tag in soup(["script", "style"]):
                tag.decompose()

            text = soup.get_text()
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            all_text += text

        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return all_text