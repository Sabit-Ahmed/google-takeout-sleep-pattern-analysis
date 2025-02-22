import time

from bs4 import BeautifulSoup
import re
from datetime import datetime
import pandas as pd
from tensorflow import timestamp

from web_scraper import WebScraper

class FileParser:
    def __init__(self):
        self.entries = []

    def parse_html_search_history(self, file_path):
        num_lines = 100
        """Parse the MyActivity_search.html file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = "".join(f.readlines()[:num_lines])  # Read first N lines
            soup = BeautifulSoup(content, 'html.parser')

            # Find all entries
            cells = soup.find_all('div', class_='outer-cell')
            for cell in cells:
                entry = self._parse_cell(cell)
                if entry:
                    self.entries.append(entry)


    def _parse_cell(self, cell):
        """Parse individual cell from HTML"""
        content_cell = cell.find('div', class_='content-cell')
        if not content_cell:
            return None

        # Get timestamp
        timestamp_match = re.search(
            r'([A-Z][a-z]{2} \d{1,2}, (\d{4}), \d{1,2}:\d{2}:\d{2}\s+[AP]M EST)',
            content_cell.text
        )

        if not timestamp_match:
            return None  # Skip if no timestamp is found

        print(timestamp_match)
        timestamp = timestamp_match.group(1)

        year = timestamp_match.group(2)
        # **Filter out non-2025 entries**
        # if year != "2025":
        #     return None  # Skip this entry

        # Get link and title
        link = content_cell.find('a')
        title = link.text.strip() if link else ''
        url = link.get('href', '') if link else ''

        # Clean Google redirect URLs
        if url.startswith('https://www.google.com/url?q='):
            url_match = re.search(r'q=(.+?)&', url)
            if not url_match:
                return None
            url = url_match.group(1)

        # Get details
        details_cell = cell.find('div', class_='mdl-typography--caption')
        details = self._clean_details(details_cell.text) if details_cell else ''

        # print(f"timestamp: {timestamp}, url: {url}, title: {title}")

        # Add delay to be respectful to servers
        time.sleep(1)

        return {
            'timestamp': timestamp,
            'title': title,
            'url': url,
            'details': details,
        }

    def parse_text_browsing_history(self, file_path):
        """Parse the paste.txt browsing history"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            current_entry = {}

            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if line.startswith('Visited'):
                    # Save previous entry if exists
                    if current_entry.get('timestamp'):
                        self.entries.append(current_entry.copy())
                    current_entry = {}

                    # Parse URL
                    url_match = re.match(r'Visited\s+(.*)', line)
                    if url_match:
                        url = url_match.group(1).strip()
                        current_entry = {
                            'url': url,
                            'title': url,
                            'details': 'Browsing history entry'
                        }

                elif re.match(r'[A-Z][a-z]{2} \d{1,2}, \d{4}', line):
                    current_entry['timestamp'] = line.strip()

            # Add last entry
            if current_entry.get('timestamp'):
                self.entries.append(current_entry)

    def _clean_details(self, details):
        """Clean up details text"""
        details = re.sub(r'Products:\s+', '', details)
        details = re.sub(r'Why is this here\?.*', '', details)
        return details.strip()

def create_dataframe(entries):
    """Create pandas DataFrame from entries"""
    df = pd.DataFrame(entries)
    df = df.sort_values('timestamp', ascending=False)
    return df

def save_to_csv(df, output_file='search_history.csv'):
    """Save entries to CSV file"""

    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} entries to {output_file}")

# Now combine with TopicAnalyzer
def analyze_search_history(html_file_path, text_file_path, output_file):
    # First parse the files
    parser = FileParser()

    print("Parsing...")
    parser.parse_html_search_history(html_file_path)

    # print("Parsing browsing history file...")
    # parser.parse_text_browsing_history(text_file_path)

    print(f"Found {len(parser.entries)} total entries")

    # Print some sample entries to verify
    print("\nSample entries:")
    for entry in parser.entries[:5]:
        print(f"Timestamp: {entry['timestamp']}")
        print(f"Title: {entry['title']}")
        print(f"URL: {entry['url']}")
        print(f"Details: {entry['details']}")
        print("---")

    return parser.entries



if __name__ == "__main__":
    product_list = ['search', 'chrome', 'google_analytics', 'ads', 'video', 'image', 'youtube']
    # product_list = ['image']
    output_file = 'data/parsed_output_100.csv'

    merged_df = []

    try:
        # Usage example
        for product in product_list:
            print(f"Processing product {product}")
            results = analyze_search_history(f'data/MyActivity_{product}.html', 'paste.txt', output_file)
            # print("\nDiscovered Topics:", results['discovered_topics'])
            # print("\nNumber of entries processed:", len(results['entries']))
            print("\nNumber of entries processed:", len(results))
            print("Creating DataFrame and saving to CSV...")
            if len(results) > 0:
                df = create_dataframe(results)
                merged_df.append(df)
        if merged_df:
            final_df = pd.concat(merged_df, ignore_index=True)
            # # Convert to datetime with coercion
            # final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors='coerce')
            #
            # # Drop rows where timestamp is NaT (Not a Time)
            # final_df = final_df.dropna(subset=["timestamp"])
            #
            # # Ensure no None or NaT values exist in the column
            # final_df["timestamp"] = final_df["timestamp"].fillna(pd.Timestamp.min)
            #
            # # Sort the DataFrame by timestamp
            # final_df = final_df.sort_values(by="timestamp")

            # Save the cleaned and sorted DataFrame
            save_to_csv(final_df, output_file)

    except Exception as e:
        print(e)