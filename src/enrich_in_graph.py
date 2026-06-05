# File: src/enrich_in_graph.py (Complete and Correct Version)

import pandas as pd
from neo4j import GraphDatabase
import ast
from itertools import combinations
import os

class GraphEnricher:
    """
    A class to manage the connection to Neo4j and populate it with sentence,
    token, and relationship data, including enriching tokens with lexicon attributes.
    """
    def __init__(self, uri, user, password):
        """Initializes the enricher with Neo4j connection details."""
        print("Connecting to Neo4j...")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.df_sentences = None  # This will hold the processed sentence data

    def close(self):
        """Closes the database connection."""
        print("Closing Neo4j connection.")
        self.driver.close()

    def clear_database(self):
        """Wipes the entire database. Use with caution."""
        print("Clearing all data from the database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")
        
    def create_constraints(self):
        """Creates uniqueness constraints for faster merges and lookups."""
        print("Creating constraints...")
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sentence) REQUIRE s.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Token) REQUIRE t.text IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (wl:WordLexicon) REQUIRE wl.word IS UNIQUE")
        print("Constraints created.")

    def ingest_processed_sentences(self, filepath):
        """
        Loads preprocessed_for_graph.csv, creates Sentence and Token nodes,
        and the relationships between them. Stores the loaded DataFrame in the class.
        """
        print(f"Ingesting sentences and tokens from {filepath}...")
        self.df_sentences = pd.read_csv(filepath) 

        query = """
        UNWIND $rows AS row
        MERGE (s:Sentence {id: row.id})
        SET s.text = row.exampleSentence,
            s.sentimentTag = row.sentimentTag,
            s.strength = row.sentimentStrength
        
        WITH s, row.tokens AS tokenList
        UNWIND tokenList AS tokenText
        MERGE (t:Token {text: tokenText})
        MERGE (s)-[:HAS_TOKEN]->(t)
        """
        rows_to_ingest = []
        for index, row in self.df_sentences.iterrows():
            try:
                tokens = ast.literal_eval(row['tokens'])
                rows_to_ingest.append({
                    "id": index,
                    "exampleSentence": row['exampleSentence'],
                    "sentimentTag": row['sentimentTag'],
                    "sentimentStrength": row['sentimentStrength'],
                    "tokens": tokens
                })
            except (ValueError, SyntaxError):
                continue
        
        with self.driver.session() as session:
            session.run(query, rows=rows_to_ingest)
        print("Ingestion of sentences and tokens complete.")

    def ingest_word_lexicon_for_lookup(self, filepath):
        """Loads the word_sentiment_lexicon.csv into temporary :WordLexicon nodes."""
        print(f"Ingesting word lookup data from {filepath}...")
        df_lexicon = pd.read_csv(filepath)
        df_lexicon.dropna(subset=['word'], inplace=True)
        df_lexicon['word'] = df_lexicon['word'].str.lower()
        df_lexicon.drop_duplicates(subset=['word'], keep='first', inplace=True)

        query = """
        UNWIND $rows AS row
        MERGE (wl:WordLexicon {word: row.word})
        SET wl.partOfSpeech = row.partOfSpeech,
            wl.sentimentTag = row.sentimentTag,
            wl.sentimentStrength = row.sentimentStrength
        """
        rows_to_ingest = df_lexicon.to_dict('records')

        with self.driver.session() as session:
            session.run(query, rows=rows_to_ingest)
        print("Word lookup data ingested.")

    def create_cooccurrence_edges(self):
        """For each sentence, link each pair of tokens that co-occur."""
        print("Creating co-occurrence relationships between tokens...")
        if self.df_sentences is None:
            print("Error: DataFrame not loaded. Run ingest_processed_sentences first.")
            return

        pairs = []
        for _, row in self.df_sentences.iterrows():
            try:
                toks = [t for t in ast.literal_eval(row['tokens']) if t]
                for a, b in combinations(set(toks), 2):
                    pairs.append({'source': a, 'target': b})
                    pairs.append({'source': b, 'target': a})
            except (ValueError, SyntaxError):
                continue

        query = """
        UNWIND $pairs AS p
        MATCH (a:Token {text: p.source}), (b:Token {text: p.target})
        MERGE (a)-[r:CO_OCCURS]->(b)
        ON CREATE SET r.count = 1
        ON MATCH SET r.count = r.count + 1
        """
        with self.driver.session() as session:
            session.run(query, pairs=pairs)
        print("Co-occurrence relationships created.")

    def enrich_token_nodes(self):
        """Matches Token nodes with WordLexicon nodes and copies the properties."""
        print("Enriching Token nodes with sentiment and POS data...")
        query = """
        MATCH (t:Token), (wl:WordLexicon)
        WHERE t.text = wl.word
        SET t.partOfSpeech = wl.partOfSpeech,
            t.sentimentTag = wl.sentimentTag,
            t.strength = wl.sentimentStrength
        RETURN count(t) AS updated_tokens
        """
        with self.driver.session() as session:
            result = session.run(query)
            count = result.single()['updated_tokens']
        print(f"Enriched {count} token nodes.")

    def cleanup_lookup_nodes(self):
        """Removes the temporary :WordLexicon nodes."""
        print("Cleaning up temporary lookup nodes...")
        with self.driver.session() as session:
            session.run("MATCH (wl:WordLexicon) DELETE wl")
        print("Cleanup complete.")


if __name__ == '__main__':
    # --- Configuration ---
    NEO4J_URI = "neo4j+s://651ad0cf.databases.neo4j.io"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Ih60kt7LGhfYDav6F-gQl1HftueVl-uVlJbxI0pmb20"
    
    PROJECT_ROOT = r"C:\Users\soozh\AAAfyp"
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "preprocessed_for_graph.csv")
    LEXICON_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "word_sentiment_lexicon.csv")

    # --- Execution Pipeline ---
    enricher = GraphEnricher(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    enricher.clear_database()
    enricher.create_constraints()
    # This function loads the sentence data into enricher.df_sentences
    enricher.ingest_processed_sentences(PROCESSED_DATA_PATH)
    enricher.ingest_word_lexicon_for_lookup(LEXICON_DATA_PATH)
    enricher.enrich_token_nodes()
    # This function now correctly uses the already-loaded DataFrame
    enricher.create_cooccurrence_edges()
    enricher.cleanup_lookup_nodes()

    print("\nProcess finished. Your graph is now enriched and complete.")
    enricher.close()