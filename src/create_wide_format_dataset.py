import pandas as pd
import ast
import os
from IPython.display import display

class WideDatasetCreator:
    """
    A class to create a "wide" format dataset where each row represents a sentence
    and contains parallel lists of its tokens and their attributes from a lexicon.
    """
    def __init__(self, processed_path: str, lexicon_path: str):
        """Initializes the creator with paths to the processed data and raw lexicon."""
        self.processed_path = processed_path
        self.lexicon_path = lexicon_path
        self.lexicon_map = self._create_lookup_map()

    def _create_lookup_map(self) -> dict:
        """Creates a word-to-attribute dictionary from the raw lexicon file for fast lookups."""
        print("Creating word-to-attribute lookup map...")
        lexicon_df = pd.read_csv(self.lexicon_path)
        lexicon_df.dropna(subset=['word'], inplace=True)
        lexicon_df['word'] = lexicon_df['word'].str.lower()
        lexicon_df.drop_duplicates(subset=['word'], keep='first', inplace=True)
        
        # Creates a dictionary like: {'word': {'partOfSpeech': 'noun', ...}}
        return lexicon_df.set_index('word')[['partOfSpeech', 'sentimentTag', 'sentimentStrength']].to_dict('index')

    def create_and_save(self, output_path: str):
        """
        Executes the full process of creating and saving the wide format dataset.
        """
        print(f"Loading processed sentences from {self.processed_path}...")
        sentences_df = pd.read_csv(self.processed_path)
        
        wide_format_data = []
        
        print("Processing each sentence to create parallel attribute lists...")
        for _, row in sentences_df.iterrows():
            try:
                tokens = ast.literal_eval(row['tokens'])
            except (ValueError, SyntaxError):
                continue

            token_pos_list = []
            token_sentiment_tag_list = []
            token_sentiment_strength_list = []

            for token in tokens:
                # Use placeholders for words not found in the lexicon
                attributes = self.lexicon_map.get(token, {
                    'partOfSpeech': 'unknown',
                    'sentimentTag': 'unknown',
                    'sentimentStrength': 0.0
                })
                
                token_pos_list.append(attributes.get('partOfSpeech', 'unknown'))
                token_sentiment_tag_list.append(attributes.get('sentimentTag', 'unknown'))
                token_sentiment_strength_list.append(attributes.get('sentimentStrength', 0.0))
            
            wide_format_data.append({
                'tokens': tokens,
                'partOfSpeech': token_pos_list,
                'sentimentTag': token_sentiment_tag_list,
                'sentimentStrength': token_sentiment_strength_list,
                'overallSentenceTag': row['sentimentTag'],
                'overallSentenceStrength': row['sentimentStrength']
            })

        wide_df = pd.DataFrame(wide_format_data)
        
        print(f"Saving wide format dataset to {output_path}...")
        wide_df.to_csv(output_path, index=False)
        print("Process complete.")
        return wide_df

# This block allows the script to be run directly from the command line
if __name__ == '__main__':
    # --- Configuration ---
    PROJECT_ROOT = r"C:\Users\soozh\AAAfyp"
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "preprocessed_for_graph.csv")
    LEXICON_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "word_sentiment_lexicon.csv")
    WIDE_FORMAT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "sentence_wide_format_dataset.csv")
    
    # --- Execution ---
    # 1. Create an instance of the creator class
    dataset_creator = WideDatasetCreator(PROCESSED_DATA_PATH, LEXICON_DATA_PATH)
    
    # 2. Run the creation and saving process
    final_wide_df = dataset_creator.create_and_save(WIDE_FORMAT_OUTPUT_PATH)
    
    print("\nSample of the new wide format dataset:")
    with pd.option_context('display.max_colwidth', 100):
        display(final_wide_df.head())