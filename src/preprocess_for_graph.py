import pandas as pd
import re
import spacy
import emoji
import ast # Added for the new wide format dataset script

class LexiconPreprocessor:
    """
    A class to handle the loading, cleaning, and tokenization of the raw lexicon data.
    """
    def __init__(self, raw_filepath: str):
        """Initializes the preprocessor with the path to the raw data."""
        self.raw_filepath = raw_filepath
        self.df = None  # This will hold the DataFrame as we process it
        
        # Load the spacy model once during initialization for efficiency
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def _demojize_and_clean(self, text: str) -> str:
        """Helper method to translate emojis and clean the resulting text."""
        s = emoji.demojize(text, language='en')
        return re.sub(r":(.*?):", lambda m: m.group(1).replace("_", " "), s)

    def _clean_and_normalize_punctuation(self, text: str) -> str:
        """Helper method for improved punctuation handling."""
        text = re.sub(r'([!?.,])\1+', r'\1', text) # Collapse repeated punctuation
        text = re.sub(r'([!?.,])', r' \1 ', text)  # Add space around punctuation
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
        return text

    def process_and_save(self, output_filepath: str):
        """
        Executes the full preprocessing pipeline and saves the result to a CSV file.
        This is the main public method to call.
        """
        print(f"Loading raw dataset from {self.raw_filepath}...")
        self.df = pd.read_csv(self.raw_filepath)

        print("Step 1: Filtering columns and dropping empty rows...")
        self.df = self.df[["partOfSpeech", "sentimentTag", "sentimentStrength", "exampleSentence"]]
        self.df.dropna(subset=['exampleSentence'], inplace=True)
        self.df = self.df[self.df["exampleSentence"].str.strip() != ""]
        
        print("Step 2: Translating emojis...")
        self.df["cleanSentence"] = self.df["exampleSentence"].astype(str).apply(self._demojize_and_clean)

        print("Step 3: Lowercasing text...")
        self.df["cleanSentence"] = self.df["cleanSentence"].str.lower()
        
        print("Step 4: Cleaning and normalizing punctuation...")
        self.df["cleanSentence"] = self.df["cleanSentence"].apply(self._clean_and_normalize_punctuation)

        print("Step 5: Tokenizing sentences...")
        tokens_list = []
        for doc in self.nlp.pipe(self.df["cleanSentence"].astype(str), batch_size=50):
            tokens_list.append([token.text for token in doc if not token.is_space])
        self.df["tokens"] = tokens_list

        # Prepare the final dataframe for export
        final_df = self.df[["exampleSentence", "partOfSpeech", "sentimentTag", "sentimentStrength", "tokens"]]
        
        print(f"Preprocessing complete. Saving to {output_filepath}...")
        final_df.to_csv(output_filepath, index=False)
        print("Intermediate file created successfully.")
        
        return final_df

# This block allows the script to be run directly from the command line
if __name__ == '__main__':
    PROJECT_ROOT = r"C:\Users\soozh\AAAfyp"
    RAW_DATA_PATH = f"{PROJECT_ROOT}/data/word_sentiment_lexicon.csv"
    INTERMEDIATE_OUTPUT_PATH = f"{PROJECT_ROOT}/data/preprocessed_for_graph.csv"
    
    # How to use the class:
    # 1. Create an instance of the preprocessor
    preprocessor = LexiconPreprocessor(RAW_DATA_PATH)
    
    # 2. Run the processing pipeline and save the output
    processed_dataframe = preprocessor.process_and_save(INTERMEDIATE_OUTPUT_PATH)

    print("\n--- Preprocessing Finished ---")
    print("Sample of the processed data:")
    print(processed_dataframe.head())