import os

import pandas as pd
from io import StringIO
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

import pandas as pd
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import unquote
from anthropic import Anthropic
import time
from typing import Optional, Dict, List

from openai import OpenAI

# API_KEY = os.getenv('GPT_API_KEY', '')
# client = OpenAI(api_key=API_KEY)


class TopicAnalyzer:
    def __init__(self, n_topics=10):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Download required NLTK data
        nltk.download('punkt_tab')
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.n_topics = n_topics
        self.vectorizer = None
        self.lda_model = None
        self.topic_words = None
        self.topic_names = {}

    def get_webpage_content(self, url):
        """Get content from webpage."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get title and content
            title = soup.title.string if soup.title else ''
            paragraphs = soup.find_all('p')
            content = ' '.join([p.text.strip() for p in paragraphs[:5]])

            return f"{title} {content}"
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def extract_search_terms(self, url):
        """Extract search terms from URL."""
        query = re.findall(r'q=([^&]+)', url)
        if query:
            return requests.utils.unquote(query[0]).replace('+', ' ')
        return None

    def preprocess_text(self, text):
        """Preprocess text for analysis."""
        text = str(text).lower()
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token)
                  for token in tokens
                  if token.isalnum() and token not in self.stop_words]
        return ' '.join(tokens)

    def collect_documents(self, df):
        """Collect and preprocess all documents from DataFrame."""
        documents = []

        for _, row in df.iterrows():
            title = str(row['title'])
            url = str(row['url'])

            # Start with title if it's meaningful
            doc_content = []
            if title and title != '.' and not title.startswith('http'):
                doc_content.append(title)

            # Add search terms if it's a search URL
            if 'google.com/search' in url:
                search_terms = self.extract_search_terms(url)
                if search_terms and search_terms != '.':
                    doc_content.append(search_terms)

            # Get webpage content
            webpage_content = self.get_webpage_content(url)
            if webpage_content:
                doc_content.append(webpage_content)

            # Combine and preprocess all content
            if doc_content:
                combined_content = ' '.join(doc_content)
                processed_content = self.preprocess_text(combined_content)
                documents.append(processed_content)
            else:
                documents.append('')

        return documents

    def interpret_topic(self, top_words):
        """Interpret topic based on its most frequent words."""
        # Common patterns to look for in top words
        patterns = {
            'EDUCATION': {'university', 'college', 'student', 'course', 'study', 'class', 'academic', 'school',
                          'degree'},
            'TECHNOLOGY': {'software', 'computer', 'web', 'app', 'tech', 'digital', 'online', 'zoom'},
            'RESEARCH': {'research', 'paper', 'study', 'analysis', 'journal', 'data', 'science'},
            'SOCIAL_MEDIA': {'facebook', 'social', 'twitter', 'instagram', 'media', 'profile', 'share'},
            'NAVIGATION': {'map', 'location', 'direction', 'route', 'address', 'place'},
            'SHOPPING': {'shop', 'buy', 'price', 'store', 'product', 'sale', 'market'},
            'NEWS': {'news', 'article', 'report', 'update', 'latest', 'story'},
            'BUSINESS': {'business', 'company', 'service', 'professional', 'industry', 'management'}
        }

        # Count matches for each pattern
        matches = {category: len(set(top_words) & word_set)
                   for category, word_set in patterns.items()}

        # Get category with most matches
        if any(matches.values()):
            best_match = max(matches.items(), key=lambda x: x[1])
            if best_match[1] > 0:  # If there's at least one match
                return best_match[0]

        # If no good matches, create a custom topic name from top 3 words
        return f"{top_words[0].upper()}_{top_words[1].upper()}"

    def fit_lda(self, documents):
        """Fit LDA model and interpret topics."""
        # Create document-term matrix
        self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        doc_term_matrix = self.vectorizer.fit_transform(documents)

        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=10
        )
        self.lda_model.fit(doc_term_matrix)

        # Get and interpret topics
        feature_names = self.vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topic_name = self.interpret_topic(top_words)
            self.topic_names[topic_idx] = topic_name
            print(f"\nTopic {topic_name}:")
            print(f"Top words: {', '.join(top_words)}")

    def assign_topic(self, text):
        """Assign interpreted topic to a single document."""
        if not text.strip():
            return 'UNCATEGORIZED'

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Transform to document-term matrix
        doc_term_matrix = self.vectorizer.transform([processed_text])

        # Get topic distribution
        topic_dist = self.lda_model.transform(doc_term_matrix)[0]

        # Get dominant topic
        dominant_topic = topic_dist.argmax()
        confidence = topic_dist[dominant_topic]

        if confidence > 0.3:  # Confidence threshold
            topic_name = self.topic_names[dominant_topic]
            print(f"Text: {text[:50]}... -> {topic_name} (confidence: {confidence:.2f})")
            return topic_name
        return 'UNCATEGORIZED'

    def analyze_topics(self, df):
        """Main method to analyze topics in DataFrame."""
        print("Collecting documents...")
        documents = self.collect_documents(df)

        print("\nFitting LDA model...")
        self.fit_lda(documents)

        print("\nAssigning topics to documents...")
        df['topic'] = df.apply(
            lambda row: self.assign_topic(
                f"{row['title']} {self.extract_search_terms(row['url']) or ''}"
            ),
            axis=1
        )

        return df


class ModernTopicAnalyzer:
    def __init__(self):
        # Initialize zero-shot classifier
        self.classifier = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-mnli")

        self.candidate_labels = [
            "Education & Academia",
            "Technology & Software",
            "Search & Navigation",
            "Social Media & Communication",
            "Business & Professional",
            "Entertainment & Media",
            "Shopping & Commerce",
            "Travel & Transportation",
            "News & Current Events",
            "Personal & Lifestyle"
        ]

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def extract_search_query(self, url):
        """Extract search terms from Google URL."""
        if 'google.com/search' in url:
            query = re.findall(r'q=([^&]+)', url)
            if query:
                return unquote(query[0]).replace('+', ' ')
        return None

    def get_webpage_info(self, url):
        """Extract information from webpage."""
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get title and meta description
            title = soup.title.string if soup.title else ''
            meta_desc = soup.find('meta', {'name': 'description'})
            description = meta_desc['content'] if meta_desc else ''

            return f"{title} {description}"
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def classify_content(self, text):
        """Classify content using zero-shot classification."""
        if not text or text.strip() == '.' or len(text.strip()) < 3:
            return "Uncategorized"

        try:
            result = self.classifier(text, self.candidate_labels, multi_label=False)
            return result['labels'][0]
        except Exception as e:
            print(f"Classification error: {str(e)}")
            return "Uncategorized"

    def process_row(self, row):
        """Process a single row of data."""
        content = []

        # Add title if meaningful
        if row['title'] and row['title'] != '.' and not row['title'].startswith('http'):
            content.append(row['title'])

        # Handle different URL types
        url = row['url']

        # Case 1: Google Search
        if 'google.com/search' in url:
            query = self.extract_search_query(url)
            if query:
                content.append(query)

        # Case 2: Maps
        elif 'maps.google' in url or '/maps/' in url:
            return "Travel & Navigation"

        # Case 3: Video platforms
        elif any(x in url for x in ['youtube.com', 'vimeo.com', 'dailymotion']):
            webpage_info = self.get_webpage_info(url)
            if webpage_info:
                content.append(webpage_info)

        # Case 4: Regular websites
        else:
            webpage_info = self.get_webpage_info(url)
            if webpage_info:
                content.append(webpage_info)

        # Classify combined content
        text_to_classify = ' '.join(content)
        return self.classify_content(text_to_classify)

    def analyze_topics(self, df):
        """Analyze topics in the DataFrame."""
        print("Processing rows...")
        df['topic'] = df.apply(self.process_row, axis=1)
        return df


class ClaudeTopicAnalyzer:
    def __init__(self, api_key: str):
        """
        Initialize the topic analyzer with Claude API.

        Args:
            api_key: Anthropic API key
        """
        self.client = Anthropic(api_key=api_key)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def extract_search_query(self, url: str) -> Optional[str]:
        """Extract search terms from Google URL."""
        if 'google.com/search' in url:
            query = re.findall(r'q=([^&]+)', url)
            if query:
                return unquote(query[0]).replace('+', ' ')
        return None

    def get_webpage_info(self, url: str) -> Optional[str]:
        """Extract information from webpage."""
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get title and meta description
            title = soup.title.string if soup.title else ''
            meta_desc = soup.find('meta', {'name': 'description'})
            description = meta_desc['content'] if meta_desc else ''

            # Get main content (first few paragraphs)
            paragraphs = soup.find_all('p')[:3]
            content = ' '.join(p.text.strip() for p in paragraphs)

            return f"Title: {title}\nDescription: {description}\nContent: {content}"
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def classify_with_claude(self, text: str) -> str:
        """
        Use Claude to classify the topic of the given text.

        Args:
            text: Text to classify

        Returns:
            Classified topic
        """
        try:
            prompt = f"""Analyze the following text and assign a single specific topic category that best describes it. The topic should be specific enough to be meaningful but general enough to group similar items.

Text to analyze:
{text}

Respond with ONLY the topic category - no other text, explanation or punctuation. Some example categories (but don't feel limited to these):

- Academic Research
- Business & Finance  
- Computer Science
- Education
- Entertainment
- Job Search
- Maps & Navigation
- News & Current Events
- Online Shopping
- Personal Communication
- Programming & Development 
- Social Media
- Software & Tools
- Travel & Transportation
- Video Streaming
"""

            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=100,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            topic = message.content[0].text.strip()
            return topic

        except Exception as e:
            print(f"Classification error: {str(e)}")
            return "Uncategorized"

    def process_row(self, row: pd.Series) -> str:
        """
        Process a single row of browser history data.

        Args:
            row: DataFrame row containing 'title' and 'url'

        Returns:
            Classified topic
        """
        content_parts = []

        # Add title if meaningful
        if row['title'] and row['title'] != '.' and not row['title'].startswith('http'):
            content_parts.append(f"Page title: {row['title']}")

        url = row['url']

        # Handle different URL types
        if 'google.com/search' in url:
            query = self.extract_search_query(url)
            if query:
                content_parts.append(f"Search query: {query}")

        elif 'maps.google' in url or '/maps/' in url:
            return "Maps & Navigation"

        elif any(x in url for x in ['youtube.com', 'vimeo.com', 'dailymotion']):
            webpage_info = self.get_webpage_info(url)
            if webpage_info:
                content_parts.append(webpage_info)

        else:
            webpage_info = self.get_webpage_info(url)
            if webpage_info:
                content_parts.append(webpage_info)

        if not content_parts:
            return "Uncategorized"

        # Combine all content and classify
        text_to_classify = "\n".join(content_parts)
        return self.classify_with_claude(text_to_classify)

    def analyze_topics(self, df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
        """
        Analyze topics in the DataFrame.

        Args:
            df: Input DataFrame with 'title' and 'url' columns
            batch_size: Number of rows to process before saving checkpoint

        Returns:
            DataFrame with added 'topic' column
        """
        results_df = df.copy()
        results_df['topic'] = 'Uncategorized'

        print(f"Processing {len(df)} rows...")

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]

            # Process batch
            for idx, row in batch.iterrows():
                try:
                    topic = self.process_row(row)
                    results_df.at[idx, 'topic'] = topic
                    print(f"Processed {idx + 1}/{len(df)}: {topic}")
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"Error processing row {idx}: {str(e)}")

            # Save checkpoint
            results_df.to_csv('topic_analysis_checkpoint.csv', index=False)
            print(f"Saved checkpoint after {i + len(batch)} rows")

        return results_df


class BatchTopicAnalyzer:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def extract_search_query(self, url: str) -> Optional[str]:
        """Extract search terms from Google URL."""
        if 'google.com/search' in url:
            query = re.findall(r'q=([^&]+)', url)
            if query:
                return unquote(query[0]).replace('+', ' ')
        return None

    def get_webpage_info(self, url: str) -> Optional[str]:
        """Extract information from webpage."""
        try:
            if 'maps.google' in url or '/maps/' in url:
                return "Map location"

            response = requests.get(url, headers=self.headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')

            title = soup.title.string if soup.title else ''
            meta_desc = soup.find('meta', {'name': 'description'})
            description = meta_desc['content'] if meta_desc else ''

            return f"{title} - {description}"
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def prepare_batch_data(self, df: pd.DataFrame, batch_size: int = 50) -> List[List[Dict]]:
        """Prepare data in batches for Claude."""
        all_batches = []
        entries = []

        for idx, row in df.iterrows():
            entry_data = {
                'id': idx,
                'title': row['title'] if row['title'] and row['title'] != '.' else '',
                'url': row['url']
            }

            # Add search query if it's a Google search
            if 'google.com/search' in row['url']:
                query = self.extract_search_query(row['url'])
                if query:
                    entry_data['search_query'] = query

            # Add webpage info for non-search URLs
            elif not any(x in row['url'] for x in ['google.com/search', 'maps.google']):
                webpage_info = self.get_webpage_info(row['url'])
                if webpage_info:
                    entry_data['webpage_info'] = webpage_info

            entries.append(entry_data)

            # Create new batch when reaching batch_size
            if len(entries) == batch_size:
                all_batches.append(entries)
                entries = []

        # Add remaining entries
        if entries:
            all_batches.append(entries)

        return all_batches

    def classify_batch(self, batch: List[Dict]) -> Dict[int, str]:
        """Classify a batch of entries using Claude."""
        prompt = """You are a topic classifier for browser history data. Analyze each entry below and assign a single specific topic category. Reply in JSON format where the key is the entry ID and the value is the topic category.

Example categories (but don't feel limited to these):
- Academic Research
- Business & Finance
- Computer Science
- Education
- Entertainment
- Job Search
- Maps & Navigation
- News & Current Events
- Online Shopping
- Personal Communication
- Programming & Development
- Social Media
- Software & Tools
- Travel & Transportation
- Video Streaming

Entries to classify:
"""
        # Add each entry to the prompt
        for entry in batch:
            prompt += f"\nEntry ID {entry['id']}:\n"
            if entry.get('title'):
                prompt += f"Title: {entry['title']}\n"
            if entry.get('search_query'):
                prompt += f"Search Query: {entry['search_query']}\n"
            if entry.get('webpage_info'):
                prompt += f"Page Info: {entry['webpage_info']}\n"
            prompt += f"URL: {entry['url']}\n"

        prompt += "\nRespond with ONLY a JSON object mapping entry IDs to topics. Example format:\n"
        prompt += '{\n  "0": "Education",\n  "1": "Social Media"\n}'

        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Parse JSON response
            result = json.loads(response.content[0].text.strip())
            return {int(k): v for k, v in result.items()}

        except Exception as e:
            print(f"Classification error: {str(e)}")
            return {entry['id']: "Uncategorized" for entry in batch}

    def analyze_topics(self, df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
        """Analyze topics in the DataFrame using batched processing."""
        results_df = df.copy()
        results_df['topic'] = 'Uncategorized'

        print(f"Processing {len(df)} rows in batches of {batch_size}...")

        # Prepare batches
        batches = self.prepare_batch_data(df, batch_size)

        # Process each batch
        for i, batch in enumerate(batches):
            print(f"Processing batch {i + 1}/{len(batches)}")

            # Get classifications for batch
            classifications = self.classify_batch(batch)

            # Update DataFrame with results
            for entry_id, topic in classifications.items():
                results_df.at[entry_id, 'topic'] = topic

            # Save checkpoint
            results_df.to_csv('topic_analysis_checkpoint.csv', index=False)
            print(f"Saved checkpoint after batch {i + 1}")

        return results_df


class ChatGPTTopicAnalyzer:
    def __init__(self, api_key: str, n_topics=10):
        self.api_key = api_key

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.n_topics = n_topics
        self.candidate_labels = [
            "Education & Academia",
            "Technology & Software",
            "Search & Navigation",
            "Social Media & Communication",
            "Business & Professional",
            "Entertainment & Media",
            "Shopping & Commerce",
            "Travel & Transportation",
            "News & Current Events",
            "Personal & Lifestyle"
        ]

    def get_webpage_content(self, url):
        """Get content from webpage."""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get title and content
            title = soup.title.string if soup.title else ''
            paragraphs = soup.find_all('p')
            content = ' '.join([p.text.strip() for p in paragraphs[:5]])

            return f"{title} {content}"
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def extract_search_terms(self, url):
        """Extract search terms from URL."""
        query = re.findall(r'q=([^&]+)', url)
        if query:
            return unquote(query[0]).replace('+', ' ')
        return None

    def classify_content(self, text):
        """Classify content using ChatGPT API."""
        if not text or text.strip() == '.' or len(text.strip()) < 3:
            return "Uncategorized"

        try:
            prompt = f"""
            Classify the following text into one of the following categories:
            - Education & Academia
            - Technology & Software
            - Search & Navigation
            - Social Media & Communication
            - Business & Professional
            - Entertainment & Media
            - Shopping & Commerce
            - Travel & Transportation
            - News & Current Events
            - Personal & Lifestyle

            Text to classify:
            {text}

            Respond with ONLY the category name.
            """

            response = client.chat.completions.create(model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.2)

            topic = response.choices[0].message.content.strip()
            return topic
        except Exception as e:
            print(f"Classification error: {str(e)}")
            return "Uncategorized"

    def process_row(self, row):
        """Process a single row of data."""
        content = []

        # Add title if meaningful
        if row['title'] and row['title'] != '.' and not row['title'].startswith('http'):
            content.append(row['title'])

        # Handle different URL types
        url = row['url']

        # Case 1: Google Search
        if 'google.com/search' in url:
            query = self.extract_search_terms(url)
            if query:
                content.append(query)

        # Case 2: Maps
        elif 'maps.google' in url or '/maps/' in url:
            return "Travel & Navigation"

        # Case 3: Video platforms
        elif any(x in url for x in ['youtube.com', 'vimeo.com', 'dailymotion']):
            webpage_info = self.get_webpage_content(url)
            if webpage_info:
                content.append(webpage_info)

        # Case 4: Regular websites
        else:
            webpage_info = self.get_webpage_content(url)
            if webpage_info:
                content.append(webpage_info)

        # Classify combined content
        text_to_classify = ' '.join(content)
        return self.classify_content(text_to_classify)

    def analyze_topics(self, df):
        """Analyze topics in the DataFrame."""
        print("Processing rows...")
        df['topic'] = df.apply(self.process_row, axis=1)
        return df


def process_csv(input_data, api_key=None, batch_size=100):
    """Process the CSV content or DataFrame and analyze topics."""
    try:
        # Handle input data
        if isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif isinstance(input_data, str):
            cleaned_content = input_data.replace('‚ÄØ', ' ')
            cleaned_content = cleaned_content.replace('timestamp', 'timestamp,')
            cleaned_content = cleaned_content.replace('title', 'title,')
            cleaned_content = cleaned_content.replace('url', 'url,')
            cleaned_content = cleaned_content.replace('details', 'details\n')
            df = pd.read_csv(StringIO(cleaned_content))
        else:
            raise ValueError("Input must be either a DataFrame or a string")

        # Initialize and run analyzer
        # analyzer = TopicAnalyzer(n_topics=10)  # Adjust number of topics as needed
        # analyzer = ModernTopicAnalyzer()
        # analyzer = ClaudeTopicAnalyzer(api_key)
        analyzer = BatchTopicAnalyzer(api_key)
        # analyzer = ChatGPTTopicAnalyzer(api_key)
        results = analyzer.analyze_topics(df, batch_size)

        # Save results
        output_file = 'data/topic_modeled_output.csv'
        results.to_csv(output_file, index=False)
        print(f"\nFile saved as: {output_file}")

        return results
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # For DataFrame input
    df = pd.read_csv('data/parsed_output.csv')
    API_KEY = os.getenv('CLAUDE_API_KEY', '')

    results = process_csv(df, API_KEY, 300)

    if results is not None:
        output_file = 'data/categorized_output.csv'
        results.to_csv(output_file, index=False)

        print("\nTopic Distribution:")
        print(results['topic'].value_counts())