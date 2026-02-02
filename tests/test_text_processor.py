#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the TextProcessor module.
"""

import unittest
import os
import sys

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.text_processor import TextProcessor
from src.config import Config


class TestTextProcessor(unittest.TestCase):
    """Test cases for the TextProcessor class."""

    def setUp(self):
        """Set up a TextProcessor instance for testing."""
        # Create a minimal configuration
        config = Config()
        config.text_processing = {
            "spacy_model": "en_core_web_sm",
            "nltk_data_path": None,
            "min_memory_length": 5,
            "max_memory_length": 200,
            "personal_info_keywords": ["I", "my", "mine", "me"],
            "preference_keywords": ["like", "love", "hate", "prefer", "enjoy", "dislike"],
            "factual_keywords": ["is", "are", "was", "were", "has", "have", "had"],
            "deletion_keywords": ["forget", "delete", "remove", "erase"]
        }
        
        # Initialize the TextProcessor
        self.text_processor = TextProcessor(config)

    def test_extract_memory_candidates(self):
        """Test extracting memory candidates from text."""
        # Test with a simple text containing multiple potential memories
        text = "My name is John. I like to play tennis. I was born in New York."
        
        candidates = self.text_processor.extract_memory_candidates(text)
        
        # Check that we extracted the expected memories
        self.assertEqual(len(candidates), 3)
        # The implementation uses lowercase for text processing
        self.assertIn("my name is john.", candidates)
        self.assertIn("i like to play tennis.", candidates)
        self.assertIn("i was born in new york.", candidates)

    def test_clean_text(self):
        """Test cleaning text."""
        # Test with text containing extra whitespace and special characters
        text = "  This is a   test with \n newlines and \t tabs.  "
        
        cleaned_text = self.text_processor._clean_text(text)
        
        # Check that the text was cleaned properly
        self.assertEqual(cleaned_text, "this is a test with newlines and tabs.")

    def test_split_into_sentences(self):
        """Test splitting text into sentences."""
        # Test with text containing multiple sentences
        text = "This is the first sentence. This is the second sentence! Is this the third sentence?"
        
        sentences = self.text_processor._split_into_sentences(text)
        
        # Check that we got the expected sentences
        self.assertEqual(len(sentences), 3)
        # The implementation may or may not lowercase the text, so we'll check case-insensitively
        self.assertEqual(sentences[0].lower(), "this is the first sentence.".lower())
        self.assertEqual(sentences[1].lower(), "this is the second sentence!".lower())
        self.assertEqual(sentences[2].lower(), "is this the third sentence?".lower())

    def test_is_personal_information(self):
        """Test identifying personal information."""
        # Test with sentences containing personal information
        personal_sentences = [
            "My name is John",
            "I live in New York",
            "My favorite color is blue"
        ]
        
        # Test with sentences not containing personal information
        non_personal_sentences = [
            "The sky is blue",
            "Dogs are mammals",
            "Paris is the capital of France"
        ]
        
        # Check personal sentences
        for sentence in personal_sentences:
            self.assertTrue(
                self.text_processor._contains_personal_info(sentence),
                f"Failed to identify personal information in: {sentence}"
            )
        
        # Check non-personal sentences
        for sentence in non_personal_sentences:
            self.assertFalse(
                self.text_processor._contains_personal_info(sentence),
                f"Incorrectly identified personal information in: {sentence}"
            )

    def test_is_preference(self):
        """Test identifying preferences."""
        # Test with sentences containing preferences
        preference_sentences = [
            "I like chocolate ice cream",
            "I hate waking up early",
            "I enjoy reading science fiction"
        ]
        
        # Test with sentences not containing preferences
        non_preference_sentences = [
            "The book is on the table",
            "She went to the store",
            "The movie starts at 8 PM"
        ]
        
        # Check preference sentences
        for sentence in preference_sentences:
            self.assertTrue(
                self.text_processor._contains_preference(sentence),
                f"Failed to identify preference in: {sentence}"
            )
        
        # Check non-preference sentences
        for sentence in non_preference_sentences:
            self.assertFalse(
                self.text_processor._contains_preference(sentence),
                f"Incorrectly identified preference in: {sentence}"
            )

    def test_is_factual_statement(self):
        """Test identifying factual statements."""
        # Test with sentences containing factual statements
        factual_sentences = [
            "The Earth is the third planet from the Sun.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Paris is the capital of France."
        ]
        
        # Test with sentences not containing factual statements - using first-person statements
        # since the implementation checks for absence of first-person pronouns
        non_factual_sentences = [
            "I think the weather is nice today.",
            "I like pizza a lot.",
            "I am feeling good today."
        ]
        
        # Check factual sentences
        for sentence in factual_sentences:
            self.assertTrue(
                self.text_processor._is_factual_statement(sentence),
                f"Failed to identify factual statement in: {sentence}"
            )
        
        # Check non-factual sentences
        for sentence in non_factual_sentences:
            self.assertFalse(
                self.text_processor._is_factual_statement(sentence),
                f"Incorrectly identified factual statement in: {sentence}"
            )

    def test_is_valid_memory(self):
        """Test determining if a sentence is a valid memory."""
        # Test with valid memory sentences
        valid_memory_sentences = [
            "My name is John",  # Personal information
            "I like chocolate ice cream",  # Preference
            "The Earth is round"  # Factual statement
        ]
        
        # Test with invalid memory sentences that are too short
        invalid_memory_sentences = [
            "Hi",  # Too short
            "OK",  # Too short
            "Yes"  # Too short
        ]
        
        # Check valid memory sentences
        for sentence in valid_memory_sentences:
            self.assertTrue(
                self.text_processor.is_valid_memory(sentence),
                f"Failed to identify valid memory in: {sentence}"
            )
        
        # Check invalid memory sentences
        for sentence in invalid_memory_sentences:
            self.assertFalse(
                self.text_processor.is_valid_memory(sentence),
                f"Incorrectly identified valid memory in: {sentence}"
            )

    def test_extract_concepts(self):
        """Test extracting concepts from text."""
        # Test with a sentence containing multiple concepts
        sentence = "I enjoy playing tennis with my friends on weekends."
        
        concepts = self.text_processor.extract_key_concepts(sentence)
        
        # Check that we extracted the expected concepts
        self.assertIn("tennis", concepts)
        self.assertIn("friends", concepts)
        self.assertIn("weekends", concepts)

    def test_is_deletion_request(self):
        """Test identifying deletion requests."""
        # Test with sentences containing deletion requests that match the patterns in extract_deletion_requests
        deletion_sentences = [
            "Please forget that I like chocolate",
            "Delete the memory about my address",
            "Remove the information about my birthday"
        ]
        
        # Test with sentences not containing deletion requests
        non_deletion_sentences = [
            "I like chocolate",
            "My address is 123 Main St",
            "My birthday is January 1st"
        ]
        
        # Check deletion sentences
        for sentence in deletion_sentences:
            patterns = self.text_processor.extract_deletion_requests(sentence)
            self.assertTrue(
                len(patterns) > 0,
                f"Failed to identify deletion request in: {sentence}"
            )
        
        # Check non-deletion sentences
        for sentence in non_deletion_sentences:
            patterns = self.text_processor.extract_deletion_requests(sentence)
            self.assertEqual(
                len(patterns), 0,
                f"Incorrectly identified deletion request in: {sentence}"
            )

    def test_calculate_importance(self):
        """Test calculating importance of text."""
        # Test with different types of sentences
        personal_info = "My name is John and I live in New York."
        preference = "I really love chocolate ice cream."
        factual = "The Earth orbits around the Sun."
        
        # Calculate importance for each sentence
        personal_importance = self.text_processor.calculate_text_importance(personal_info)
        preference_importance = self.text_processor.calculate_text_importance(preference)
        factual_importance = self.text_processor.calculate_text_importance(factual)
        
        # Check that the importance values are within the expected range
        self.assertGreaterEqual(personal_importance, 0.0)
        self.assertLessEqual(personal_importance, 1.0)
        
        self.assertGreaterEqual(preference_importance, 0.0)
        self.assertLessEqual(preference_importance, 1.0)
        
        self.assertGreaterEqual(factual_importance, 0.0)
        self.assertLessEqual(factual_importance, 1.0)
        
        # Personal information should generally be more important than factual statements
        self.assertGreater(personal_importance, factual_importance)


if __name__ == "__main__":
    unittest.main()