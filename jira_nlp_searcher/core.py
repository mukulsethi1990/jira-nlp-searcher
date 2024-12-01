import os
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from jira import JIRA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

class JiraNLPSearcher:
    """
    A class that implements NLP-based search functionality for Jira issues.
    """
    
    def __init__(self, jira_url: str, username: str, api_token: str):
        """
        Initialize the Jira NLP searcher with connection details.
        
        Args:
            jira_url: The URL of your Jira instance
            username: Your Jira username (usually email)
            api_token: Your Jira API token
        """
        self.jira = JIRA(server=jira_url, basic_auth=(username, api_token))
        
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        
        # Cache for storing processed issues
        self.issues_df = None
        self.tfidf_matrix = None
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for NLP operations.
        
        Args:
            text: Input text to process
            
        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def fetch_issues(self, jql_query: str = "project in (KEY)", max_results: int = 100) -> pd.DataFrame:
        """
        Fetch issues from Jira and create a dataframe.
        
        Args:
            jql_query: JQL query to filter issues
            max_results: Maximum number of issues to fetch
            
        Returns:
            DataFrame containing issue details
        """
        issues = []
        
        for issue in self.jira.search_issues(jql_query, maxResults=max_results):
            issue_dict = {
                'key': issue.key,
                'summary': issue.fields.summary,
                'description': issue.fields.description or "",
                'status': issue.fields.status.name,
                'created': issue.fields.created,
                'updated': issue.fields.updated
            }
            issues.append(issue_dict)
        
        self.issues_df = pd.DataFrame(issues)
        return self.issues_df
    
    def build_search_index(self, jql_query: str = "project in (KEY)", max_results: int = 100):
        """
        Build the search index from Jira issues.
        
        Args:
            jql_query: JQL query to filter issues
            max_results: Maximum number of issues to fetch
        """
        if self.issues_df is None:
            self.fetch_issues(jql_query, max_results)
        
        # Combine summary and description for searching
        combined_text = self.issues_df['summary'] + ' ' + self.issues_df['description'].fillna('')
        processed_text = combined_text.apply(self.preprocess_text)
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_text)
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for issues using NLP.
        
        Args:
            query: Natural language search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching issues with similarity scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("Search index not built. Call build_search_index() first.")
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Transform query using existing vocabulary
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top matching indices
        top_indices = similarities.argsort()[-limit:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            issue = self.issues_df.iloc[idx]
            results.append({
                'key': issue['key'],
                'summary': issue['summary'],
                'status': issue['status'],
                'similarity': similarities[idx],
                'updated': issue['updated']
            })
        
        return results