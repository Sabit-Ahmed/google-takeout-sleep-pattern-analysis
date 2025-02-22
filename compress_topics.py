import pandas as pd

def find_best_category(topic):
    patterns = {
        'Technology & Software': [
            'software', 'technology', 'programming', 'development', 'computer', 'web', 'cloud',
            'code', 'api', 'app', 'robotics', 'artificial intelligence', 'version control'
        ],
        'Data & Analytics': [
            'data', 'analytics', 'statistics', 'analysis', 'machine learning', 'ai',
            'prediction', 'visualization', 'graph theory'
        ],
        'Digital Media & Entertainment': [
            'media', 'entertainment', 'streaming', 'video', 'music', 'gaming', 'film',
            'movie', 'television', 'sports', 'sports betting', 'fishing', 'anime'
        ],
        'Education & Learning': [
            'education', 'learning', 'academic', 'university', 'school', 'course', 'study',
            'research', 'psychology', 'history', 'philosophy', 'mathematics', 'reference',
            'biography', 'literature', 'books', 'vocabulary', 'language'
        ],
        'Business & Finance': [
            'business', 'finance', 'banking', 'investment', 'financial', 'commerce',
            'trading', 'market', 'taxes', 'employee benefits'
        ],
        'Healthcare & Medical': [
            'health', 'medical', 'healthcare', 'wellness', 'clinical', 'patient',
            'disease', 'treatment', 'epidemiology'
        ],
        'Communication & Social': [
            'communication', 'social', 'message', 'chat', 'network', 'community',
            'sharing', 'social media'
        ],
        'E-commerce & Retail': [
            'shopping', 'retail', 'ecommerce', 'store', 'product', 'sale', 'purchase',
            'consumer electronics'
        ],
        'Transportation & Automotive': [
            'automotive', 'vehicle', 'car', 'traffic', 'maintenance',
            'parking', 'auto parts', 'repair', 'motor'
        ],
        'Travel & Tourism': [
            'travel', 'tourism', 'tourist', 'vacation', 'hotel', 'flight',
            'accommodation', 'booking', 'destination'
        ],
        'Science & Research': [
            'science', 'research', 'scientific', 'physics', 'chemistry', 'biology',
            'quantum', 'cosmology'
        ],
        'Professional Services': [
            'service', 'professional', 'consulting', 'legal', 'insurance', 'accounting',
            'human resources', 'customer support'
        ],
        'Marketing & Advertising': [
            'marketing', 'advertising', 'promotion', 'brand', 'campaign', 'ads'
        ],
        'Content & Publishing': [
            'content', 'publishing', 'blog', 'article', 'news', 'editorial'
        ],
        'Government & Public Services': [
            'government', 'public', 'service', 'administration', 'policy', 'immigration',
            'military', 'human rights', 'politics', 'international relations'
        ],
        'Lifestyle & Personal': [
            'lifestyle', 'personal', 'home', 'family', 'hobby', 'recreation',
            'religion', 'arts', 'culture', 'fashion', 'relationships', 'holidays',
            'parenting'
        ],
        'Food & Culinary': [
            'food', 'cooking', 'recipe', 'nutrition', 'cuisine', 'dining', 'beverage',
            'drink', 'fruit', 'grocery'
        ],
        'Tools & Utilities': [
            'search engine', 'unit conversion', 'maps', 'navigation', 'tools', 'utility',
            'calculator', 'weather', 'login', 'user account'
        ]
    }

    topic_lower = str(topic).lower()
    best_match = {
        'category': 'Miscellaneous',
        'score': 0
    }

    for category, keywords in patterns.items():
        score = sum(1 for keyword in keywords if keyword in topic_lower)
        if score > best_match['score']:
            best_match = {
                'category': category,
                'score': score
            }

    return best_match['category']

# Read the original CSV
df = pd.read_csv('data/categorized_output.csv')

# Add the new compressed topic column
df['topic_compressed'] = df['topic'].apply(find_best_category)

# Save the updated DataFrame to a new CSV
df.to_csv('data/categorized_output_compressed.csv', index=False)

# Create a mapping CSV showing original to compressed topics
mapping_df = df[['topic', 'topic_compressed']].drop_duplicates().sort_values('topic')
mapping_df.to_csv('topic_compressed.csv', index=False)

# Print some statistics
print("\nCategory Distribution:")
print(df['topic_compressed'].value_counts())

print("\nNumber of original topics:", df['topic'].nunique())
print("Number of compressed topics:", df['topic_compressed'].nunique())