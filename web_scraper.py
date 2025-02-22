import requests
from bs4 import BeautifulSoup


class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_page_title(self, url):
        """Fetch webpage title with error handling"""
        try:
            # Add scheme if not present
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url

            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            # Get title from different possible sources
            title = None

            # Try title tag first
            if soup.title:
                title = soup.title.string

            # If no title tag, try og:title
            if not title:
                og_title = soup.find('meta', property='og:title')
                if og_title:
                    title = og_title.get('content')

            # If still no title, try h1
            if not title:
                h1 = soup.find('h1')
                if h1:
                    title = h1.text

            return title.strip() if title else None

        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None