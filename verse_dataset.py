import os
import json
from typing import Dict, Optional
from datasets import load_dataset
import time

class VerseDataset:
    """
    Utility class to fetch and manage verse and shabad IDs from the HuggingFace dataset
    https://huggingface.co/datasets/jssaluja/verse_dataset
    """
    
    def __init__(self, cache_path="verse_data_cache.json"):
        """
        Initialize the dataset handler
        
        Args:
            cache_path: Path to cache the dataset locally
        """
        self.cache_path = cache_path
        self.verse_to_shabad_map = {}
        self.hf_dataset_url = "https://huggingface.co/datasets/jssaluja/verse_dataset"
        self._load_data()
        
    def _load_data(self):
        """Load verse-to-shabad mapping from cache or fetch from HuggingFace"""
        if os.path.exists(self.cache_path):
            # Load from cache
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    self.verse_to_shabad_map = json.load(f)
                print(f"Loaded verse-to-shabad mapping from cache ({len(self.verse_to_shabad_map)} entries)")
                return
            except Exception as e:
                print(f"Error loading from cache: {str(e)}")
        
        # Fetch from HuggingFace or build from LINE_STORE
        self._build_mapping()
        
    def _build_mapping(self):
        """
        Build verse-to-shabad mapping from the HuggingFace dataset
        https://huggingface.co/datasets/jssaluja/verse_dataset
        
        The dataset contains the following fields:
        - verse_id (string): Unique identifier for the verse
        - shabad_id (string): Unique identifier for the shabad that the verse belongs to
        - page (string): Page number in the Guru Granth Sahib
        - line (string): Line number on the page
        - orig_text (string): Original text of the verse
        - asr_text (string): ASR recognized text
        - en_meaning (string): English meaning of the verse
        """
        print("Building verse-to-shabad mapping from HuggingFace dataset...")
        
        try:
            # Load the dataset directly using the datasets library
            print("Loading dataset from HuggingFace using datasets library...")
            dataset = load_dataset("jssaluja/verse_dataset")
            
            # The dataset should have a 'train' split
            if 'train' in dataset:
                # Process all examples in the dataset
                total_mappings = 0
                for example in dataset['train']:
                    verse_id = example.get('verse_id')
                    shabad_id = example.get('shabad_id')
                    
                    if verse_id and shabad_id:
                        self.verse_to_shabad_map[str(verse_id)] = str(shabad_id)
                        total_mappings += 1
                        
                        # Print progress every 1000 entries
                        if total_mappings % 1000 == 0:
                            print(f"Processed {total_mappings} mappings...")
                
                print(f"Successfully built verse-to-shabad mapping with {total_mappings} entries")
            else:
                raise ValueError("Dataset does not contain a 'train' split")
            
        except Exception as e:
            print(f"Error loading data from HuggingFace: {str(e)}")
            print("Falling back to default mappings...")
            
            # Create a basic mapping for commonly used verse IDs as fallback
            default_mappings = {
                # Add mappings for the verses we've seen in the logs
                "30955": "4082",
                "30956": "4082",
                "30957": "4082",
                "30958": "4082", 
                "30959": "4082",
                "30960": "4082",
                "30961": "4082",
                "30962": "4082", 
                "30963": "4082",
                "30964": "4082",
                "30965": "4082",
                "30966": "4083",
                "30967": "4083",
                "30968": "4083",
                "30969": "4083",
                "30970": "4083",
                "30971": "4083",
                "30972": "4083",
                "30973": "4083",
                "30974": "4083"
            }
            
            # Add these mappings to our verse_to_shabad_map
            self.verse_to_shabad_map.update(default_mappings)
            
            print(f"Added {len(default_mappings)} default verse-to-shabad mappings")
        
        # Save to cache
        self._save_cache()
        
    def _save_cache(self):
        """Save the mapping to cache file"""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.verse_to_shabad_map, f, ensure_ascii=False)
            print(f"Saved verse-to-shabad mapping to cache ({len(self.verse_to_shabad_map)} entries)")
        except Exception as e:
            print(f"Error saving to cache: {str(e)}")
    
    def get_shabad_id(self, verse_id: str) -> Optional[str]:
        """
        Get shabad ID for a given verse ID
        
        Args:
            verse_id: The verse ID to look up
            
        Returns:
            str: Shabad ID if found, None otherwise
        """
        # Convert to string if it's not already
        verse_id = str(verse_id)
        
        # Check if we have it in our mapping
        if verse_id in self.verse_to_shabad_map:
            return self.verse_to_shabad_map[verse_id]
            
        # If not found, try to fetch it or return placeholder
        # TODO: Implement on-demand fetching if needed
        print(f"⚠️ Shabad ID not found for verse ID: {verse_id}")
        return None
        
    def add_mapping(self, verse_id: str, shabad_id: str):
        """
        Add a verse-to-shabad mapping
        
        Args:
            verse_id: The verse ID
            shabad_id: The shabad ID
        """
        self.verse_to_shabad_map[str(verse_id)] = str(shabad_id)
        # Optionally save to cache after each addition
        # self._save_cache()
        
# Singleton instance to be used across the application
verse_dataset = VerseDataset()

def get_shabad_id_for_verse(verse_id: str) -> Optional[str]:
    """
    Utility function to get shabad ID for a verse ID
    
    Args:
        verse_id: The verse ID to look up
        
    Returns:
        str: Shabad ID if found, None otherwise
    """
    return verse_dataset.get_shabad_id(verse_id)
