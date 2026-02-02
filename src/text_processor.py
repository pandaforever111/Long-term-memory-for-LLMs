#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text Processor Module

This module handles text analysis, memory extraction, and semantic processing
for the GPT Memory Agent.
"""

import re
import logging
from typing import Dict, List, Optional, Union, Any, Set
import string

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from src.config import Config


class TextProcessor:
    """Text processing and analysis for memory extraction and retrieval."""

    def __init__(self, config: Config):
        """Initialize the text processor with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger("memory_agent.text_processor")
        
        # Initialize NLP components based on availability
        self.nlp = None
        self.stop_words = set()
        
        self._initialize_nlp()

    def _initialize_nlp(self) -> None:
        """Initialize NLP components based on available libraries."""
        # Try to load spaCy if available
        if SPACY_AVAILABLE and self.config.use_spacy:
            try:
                self.logger.info("Initializing spaCy NLP model")
                self.nlp = spacy.load(self.config.spacy_model)
                self.logger.debug("spaCy NLP model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load spaCy model: {e}")
                self.logger.warning("Falling back to basic text processing")
        
        # Try to load NLTK resources if available
        if NLTK_AVAILABLE and self.config.use_nltk:
            try:
                self.logger.info("Initializing NLTK resources")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
                self.logger.debug("NLTK resources loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load NLTK resources: {e}")
                self.logger.warning("Using basic stopword list")
                
        # If neither is available, use a basic stopword list
        if not self.stop_words:
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
                'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
                'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
                'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
                'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
                'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
                "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
                "weren't", 'won', "won't", 'wouldn', "wouldn't"
            }

    def extract_memory_candidates(self, text: str) -> List[str]:
        """Extract potential memory candidates from text.
        
        This function identifies statements that might be worth remembering
        based on various heuristics and patterns.
        
        Args:
            text: The input text to analyze
            
        Returns:
            List of potential memory statements
        """
        self.logger.debug(f"Extracting memory candidates from text: {text[:50]}...")
        
        # Clean and normalize the text
        cleaned_text = self._clean_text(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(cleaned_text)
        
        # Apply memory extraction rules to find candidates
        candidates = []
        
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                continue
                
            # Check for personal information patterns
            if self._contains_personal_info(sentence):
                candidates.append(sentence)
                continue
                
            # Check for preference patterns
            if self._contains_preference(sentence):
                candidates.append(sentence)
                continue
                
            # Check for factual statements
            if self._is_factual_statement(sentence):
                candidates.append(sentence)
                continue
        
        # If using spaCy, add entities as potential memories
        if self.nlp is not None:
            doc = self.nlp(cleaned_text)
            for ent in doc.ents:
                # Only consider certain entity types as memories
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
                    # Get the sentence containing this entity
                    for sent in doc.sents:
                        if ent.start >= sent.start and ent.end <= sent.end:
                            if sent.text not in candidates:
                                candidates.append(sent.text)
        
        self.logger.debug(f"Extracted {len(candidates)} memory candidates")
        return candidates

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text.
        
        Args:
            text: The input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters except punctuation needed for sentence splitting
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        
        return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: The input text
            
        Returns:
            List of sentences
        """
        if self.nlp is not None:
            # Use spaCy for sentence splitting if available
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Simple rule-based sentence splitting
            # This is a simplified approach and won't handle all cases correctly
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def _contains_personal_info(self, sentence: str) -> bool:
        """Check if a sentence contains personal information.
        
        Args:
            sentence: The sentence to check
            
        Returns:
            True if the sentence contains personal information
        """
        # Check for first-person pronouns followed by potential personal info
        personal_patterns = [
            r'\b(i|my|we|our)\b.*\b(name|live|work|from|born|age|birthday|address|email|phone|number)\b',
            r'\b(i am|i\'m)\b.*\b(from|a|an|the|working|studying)\b',
            r'\b(i|we)\b.*\b(like|love|hate|enjoy|prefer|use|have|own)\b',
            r'\b(my|our)\b.*\b(favorite|hobby|interest|passion|job|profession|career)\b'
        ]
        
        for pattern in personal_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
                
        return False

    def _contains_preference(self, sentence: str) -> bool:
        """Check if a sentence contains a preference statement.
        
        Args:
            sentence: The sentence to check
            
        Returns:
            True if the sentence contains a preference
        """
        # Check for preference patterns
        preference_patterns = [
            r'\b(i|we)\b.*\b(like|love|hate|enjoy|prefer|favorite)\b',
            r'\b(i|we)\b.*\b(don\'t|do not|doesn\'t|does not)\b.*\b(like|love|enjoy|want)\b',
            r'\b(i|we)\b.*\b(would|wouldn\'t|would not)\b.*\b(like|love|enjoy|want|prefer)\b'
        ]
        
        for pattern in preference_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
                
        return False

    def _is_factual_statement(self, sentence: str) -> bool:
        """Check if a sentence appears to be a factual statement.
        
        Args:
            sentence: The sentence to check
            
        Returns:
            True if the sentence appears to be a factual statement
        """
        # This is a simplified heuristic
        # Check if the sentence starts with a capital letter and ends with a period
        # and doesn't contain first-person pronouns
        
        if not re.search(r'\b(i|my|we|our|me)\b', sentence, re.IGNORECASE):
            # Check if it contains a verb
            verb_patterns = [
                r'\b(is|are|was|were|has|have|had|do|does|did)\b',
                r'\b\w+s\b',  # Simple check for present tense verbs
                r'\b\w+ed\b'  # Simple check for past tense verbs
            ]
            
            for pattern in verb_patterns:
                if re.search(pattern, sentence):
                    return True
                    
        return False

    def is_valid_memory(self, text: str) -> bool:
        """Determine if a text is worth storing as a memory.
        
        Args:
            text: The text to evaluate
            
        Returns:
            True if the text should be stored as a memory
        """
        # Skip very short or very long texts
        word_count = len(text.split())
        if word_count < 3 or word_count > 30:
            return False
            
        # Skip if it contains too many stopwords
        words = [w.lower() for w in text.split()]
        stopword_ratio = sum(1 for w in words if w in self.stop_words) / len(words)
        if stopword_ratio > 0.7:
            return False
            
        # Skip if it doesn't contain any content words
        content_words = [w for w in words if w not in self.stop_words and w not in string.punctuation]
        if not content_words:
            return False
            
        # Check if it contains personal information or preferences
        if self._contains_personal_info(text) or self._contains_preference(text):
            return True
            
        # Use additional heuristics to determine if it's worth remembering
        # For example, check if it contains named entities
        if self.nlp is not None:
            doc = self.nlp(text)
            if doc.ents:
                return True
                
        # Default to True for candidate memories that passed the initial filters
        return True

    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text for memory retrieval.
        
        Args:
            text: The input text
            
        Returns:
            List of key concepts
        """
        # Clean the text
        cleaned_text = self._clean_text(text)
        
        if self.nlp is not None:
            # Use spaCy for concept extraction
            doc = self.nlp(cleaned_text)
            
            # Extract named entities
            entities = [ent.text.lower() for ent in doc.ents]
            
            # Extract noun phrases
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
            
            # Extract important words (nouns, verbs, adjectives)
            important_words = [token.lemma_.lower() for token in doc 
                              if token.pos_ in ['NOUN', 'VERB', 'ADJ'] 
                              and not token.is_stop 
                              and len(token.text) > 2]
            
            # Combine all concepts and remove duplicates
            all_concepts = entities + noun_phrases + important_words
            return list(set(all_concepts))
        else:
            # Fallback to basic word tokenization and filtering
            if NLTK_AVAILABLE:
                words = word_tokenize(cleaned_text)
            else:
                # Very basic tokenization
                words = re.findall(r'\b\w+\b', cleaned_text)
            
            # Filter out stopwords and short words
            concepts = [word.lower() for word in words 
                       if word.lower() not in self.stop_words 
                       and len(word) > 2]
            
            return list(set(concepts))

    def extract_deletion_requests(self, text: str) -> List[str]:
        """Extract requests to delete or forget memories.
        
        Args:
            text: The input text
            
        Returns:
            List of content patterns to delete
        """
        # Look for phrases indicating a request to forget information
        forget_patterns = [
            r'forget\s+(?:about|that)?\s+(.*?)(?:\.|$)',
            r'don\'t\s+remember\s+(.*?)(?:\.|$)',
            r'remove\s+(?:the)?\s+(?:memory|information|data)\s+(?:about|that)?\s+(.*?)(?:\.|$)',
            r'delete\s+(?:the)?\s+(?:memory|information|data)\s+(?:about|that)?\s+(.*?)(?:\.|$)'
        ]
        
        deletion_requests = []
        
        for pattern in forget_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group(1).strip():
                    deletion_requests.append(match.group(1).strip())
        
        return deletion_requests

    def calculate_text_importance(self, text: str) -> float:
        """Calculate the importance score for a piece of text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            Importance score between 0.0 and 1.0
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated analysis
        
        # Base score
        score = 0.5
        
        # Adjust based on presence of personal information
        if self._contains_personal_info(text):
            score += 0.2
        
        # Adjust based on presence of preferences
        if self._contains_preference(text):
            score += 0.15
        
        # Adjust based on named entities if spaCy is available
        if self.nlp is not None:
            doc = self.nlp(text)
            entity_count = len(doc.ents)
            if entity_count > 0:
                score += min(0.1 * entity_count, 0.3)  # Cap at 0.3
        
        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, score))