# app.py

# import streamlit as st
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from neo4j import GraphDatabase
# import networkx as nx
# from pyvis.network import Network
# import streamlit.components.v1 as components
# import re
# import tempfile

# # --- STREAMLIT PAGE CONFIGURATION (MOVED TO THE TOP) ---
# st.set_page_config(page_title="Malay Sentiment & Context Analyzer", layout="wide")

# # --- CONFIGURATION ---
# MODEL_PATH = "./results/best_indobert_model"
# NEO4J_URI = "neo4j+s://651ad0cf.databases.neo4j.io"
# # Use Streamlit's secrets for credentials in a real application
# # For this example, we'll use placeholders but you should use st.secrets
# NEO4J_USER = st.secrets.get("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD", "Ih60kt7LGhfYDav6F-gQl1HftueVl-uVlJbxI0pmb20")


# # --- MODEL LOADING (Cached for performance) ---
# @st.cache_resource
# def load_model_and_tokenizer():
#     """Loads the fine-tuned IndoBERT model and tokenizer."""
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#         model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
#         return tokenizer, model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None, None

# tokenizer, model = load_model_and_tokenizer()
# # Define the label mapping based on the LabelEncoder during training
# # This is crucial for interpreting the model's output
# label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}


# # --- NEO4J GRAPH FUNCTIONS (Cached for performance) ---
# @st.cache_resource
# def get_neo4j_driver():
#     """Creates and returns a Neo4j driver instance."""
#     try:
#         driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
#         driver.verify_connectivity()
#         return driver
#     except Exception as e:
#         st.error(f"Could not connect to Neo4j database: {e}")
#         return None

# driver = get_neo4j_driver()

# @st.cache_data(ttl=3600) # Cache data for 1 hour
# def get_cooccurrence_data(_driver, word):
#     """Queries Neo4j for a word and its co-occurring neighbors."""
#     if not _driver:
#         return []
    
#     # Sanitize the input word: lowercase and strip whitespace
#     word = word.lower().strip()

#     query = """
#     MATCH (t1:Token {text: $word})-[r:CO_OCCURS]-(t2:Token)
#     RETURN t1.text AS source, t2.text AS target, r.count AS weight
#     ORDER BY r.count DESC
#     LIMIT 20
#     """
#     with _driver.session() as session:
#         result = session.run(query, word=word)
#         return [record.data() for record in result]

# def generate_graph_html(data, central_word):
#     """Generates an interactive pyvis graph from Neo4j data."""
#     if not data:
#         return None

#     net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    
#     # Add nodes and edges
#     central_word = central_word.lower().strip()
#     net.add_node(central_word, label=central_word, color='#FF4B4B', size=30)
    
#     for record in data:
#         target_node = record['target']
#         weight = record['weight']
        
#         if target_node not in net.nodes:
#              net.add_node(target_node, label=target_node, color='#00a1cb', size=15)
        
#         net.add_edge(central_word, target_node, value=weight, title=f"Co-occurred: {weight} times")

#     net.set_options("""
#     var options = {
#       "physics": {
#         "forceAtlas2Based": {
#           "gravitationalConstant": -50,
#           "centralGravity": 0.01,
#           "springLength": 100,
#           "springConstant": 0.08
#         },
#         "minVelocity": 0.75,
#         "solver": "forceAtlas2Based"
#       }
#     }
#     """)
    
#     # Generate the HTML content in memory and write it to a temporary file with explicit UTF-8 encoding.
#     try:
#         # Step 1: Generate HTML content as a string
#         html_content = net.generate_html(name='cooccurrence_graph.html', local=True, notebook=True)

#         # Step 2: Save this string to a temporary file
#         with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as tmpfile:
#             tmpfile.write(html_content)
#             html_path = tmpfile.name
        
#         # Step 3: Read the content back from the temporary file to display it
#         with open(html_path, 'r', encoding='utf-8') as f:
#             final_html = f.read()
            
#         return final_html
        
#     except Exception as e:
#         return f"<p>Error generating graph: {e}</p>"

# # --- SENTIMENT PREDICTION FUNCTION ---
# def predict_sentiment(text):
#     """Predicts sentiment for a given text using the loaded model."""
#     if not tokenizer or not model:
#         return None, None

#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     logits = outputs.logits
#     probabilities = torch.nn.functional.softmax(logits, dim=-1)
#     prediction = torch.argmax(probabilities, dim=-1).item()
    
#     # Create a DataFrame for easy visualization
#     prob_df = pd.DataFrame({
#         'Sentiment': [label_map[i] for i in range(len(label_map))],
#         'Probability': probabilities.flatten().numpy()
#     })
    
#     return label_map[prediction], prob_df

# # --- STREAMLIT USER INTERFACE (UI) ---

# st.title("Malay Sentiment and Context Analysis")
# st.markdown("This web application utilizes a Transformer model (IndoBERT) for sentiment analysis and a Neo4j graph database to explore word contexts.")

# st.sidebar.header("Settings")
# st.sidebar.info("This app demonstrates two key functionalities from the research project: (1) Live Sentiment Prediction and (2) Graph-Based Word Context Explorer.")

# # --- 1. Live Sentiment Prediction ---
# st.header("1. Live Sentiment Prediction")
# user_input = st.text_area("Enter a Malay sentence here:", "Saya sangat gembira dengan perkhidmatan yang diberikan.", height=100)

# if st.button("Analyze Sentiment"):
#     if user_input.strip() and model:
#         with st.spinner("Analyzing..."):
#             predicted_label, probs_df = predict_sentiment(user_input)
            
#             if predicted_label == 'positive':
#                 st.success(f"**Predicted Sentiment: Positive**")
#             elif predicted_label == 'negative':
#                 st.error(f"**Predicted Sentiment: Negative**")
#             else:
#                 st.info(f"**Predicted Sentiment: Neutral**")

#             st.write("Probability Distribution:")
#             st.bar_chart(probs_df.set_index('Sentiment'))
#     else:
#         st.warning("Please enter a sentence to analyze.")

# # --- 2. Interactive Co-occurrence Graph ---
# st.header("2. Word Context Explorer (Co-occurrence Graph)")
# word_input = st.text_input("Enter a single word to explore its context:", "makan")

# if st.button("Generate Context Graph"):
#     if not driver:
#         st.error("Connection to the graph database failed. This feature is unavailable.")
#     elif word_input.strip():
#         # Sanitize input: ensure it's a single word
#         word_to_search = word_input.strip().split()[0]
#         with st.spinner(f"Fetching context for '{word_to_search}'..."):
#             graph_data = get_cooccurrence_data(driver, word_to_search)
#             if graph_data:
#                 html_content = generate_graph_html(graph_data, word_to_search)
#                 if html_content:
#                     st.subheader(f"Co-occurrence Graph for '{word_to_search}'")
#                     st.markdown("This graph shows words that most frequently appear alongside your input word in the dataset. The thickness of the lines indicates higher co-occurrence frequency.")
#                     components.html(html_content, height=520)
#                 else:
#                     st.error("Failed to generate the graph visualization.")
#             else:
#                 st.warning(f"No co-occurrence data found for '{word_to_search}'. Try another common word like 'kereta' or 'sangat'.")
#     else:
#         st.warning("Please enter a single word.")


# app.py (Baseline Model with Advanced Graph Visualization)

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(page_title="Malay Sentiment & Context Analyzer", layout="wide")

# --- CONFIGURATION ---
MODEL_PATH = "./results/best_indobert_model" # Using the baseline model
NEO4J_URI = "neo4j+s://651ad0cf.databases.neo4j.io"
NEO4J_USER = st.secrets.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD", "Ih60kt7LGhfYDav6F-gQl1HftueVl-uVlJbxI0pmb20")

# --- MODEL LOADING ---
@st.cache_resource
def load_model_and_tokenizer():
    """Loads the fine-tuned IndoBERT model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure the model is saved in '{MODEL_PATH}'.")
        return None, None

tokenizer, model = load_model_and_tokenizer()
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'} # This must match your baseline's LabelEncoder

# --- NEO4J GRAPH FUNCTIONS ---
@st.cache_resource
def get_neo4j_driver():
    """Creates and returns a Neo4j driver instance."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.error(f"Could not connect to Neo4j database: {e}")
        return None
driver = get_neo4j_driver()

# --- IMPROVEMENT: Function now handles multiple hops ---
@st.cache_data(ttl=3600)
def get_cooccurrence_data(_driver, word, hops=1):
    """
    Queries Neo4j for a word and its top 20 co-occurring neighbors at each hop.
    """
    if not _driver: return []
    word = word.lower().strip()

    if hops == 1:
        query = """
        MATCH (t1:Token {text: $word})-[r:CO_OCCURS]-(t2:Token)
        WITH t1, r, t2 ORDER BY r.count DESC LIMIT 20
        RETURN t1.text AS source, t2.text AS target, r.count AS weight
        """
        with _driver.session() as session:
            result = session.run(query, word=word)
            return [record.data() for record in result]
    
    # --- FINAL FIX IS HERE: A simpler and more robust query for 2 hops ---
    elif hops == 2:
        query = """
        // Find the top 20 neighbors of the central word
        MATCH (t1:Token {text: $word})-[r1:CO_OCCURS]-(t2:Token)
        WITH t1, r1, t2 ORDER BY r1.count DESC LIMIT 20
        
        // From that set of neighbors, find THEIR top neighbors
        MATCH (t2)-[r2:CO_OCCURS]-(t3:Token)
        WHERE t3 <> t1 // Exclude the original central word
        
        // Return both sets of relationships
        RETURN t1.text AS source1, t2.text AS target1, r1.count AS weight1,
               t2.text AS source2, t3.text AS target2, r2.count AS weight2
        """
        records = []
        with _driver.session() as session:
            result = session.run(query, word=word)
            # We need to process the result to combine the two sets of relationships
            # and limit the second hop results manually
            first_hop_added = set()
            second_hop_counts = {}

            for record in result:
                # Add the first-degree relationship if not already added
                if record['target1'] not in first_hop_added:
                    records.append({'source': record['source1'], 'target': record['target1'], 'weight': record['weight1']})
                    first_hop_added.add(record['target1'])

                # Store second-degree relationships and their weights
                if record['source2']:
                    neighbor = record['source2']
                    if neighbor not in second_hop_counts:
                        second_hop_counts[neighbor] = []
                    second_hop_counts[neighbor].append(
                        {'source': record['source2'], 'target': record['target2'], 'weight': record['weight2']}
                    )
            
            # For each first-level neighbor, add only its top 20 second-level connections
            for neighbor, relations in second_hop_counts.items():
                # Sort relations by weight and take the top 20
                sorted_relations = sorted(relations, key=lambda x: x['weight'], reverse=True)[:20]
                records.extend(sorted_relations)

        # Remove duplicate relationships that might have been added
        unique_records = [dict(t) for t in {tuple(d.items()) for d in records}]
        return unique_records
    # -------------------------------------------------------------------
            
    return []

# --- IMPROVEMENT: Function now visualizes multi-level graphs ---
def generate_graph_html(data, central_word):
    """Generates an interactive pyvis graph from multi-level Neo4j data."""
    if not data: return None

    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    central_word = central_word.lower().strip()
    
    first_degree_neighbors = {record['target'] for record in data if record['source'] == central_word} | {record['source'] for record in data if record['target'] == central_word}
    net.add_node(central_word, label=central_word, color='#FF4B4B', size=30)
    
    for record in data:
        source, target, weight = record['source'], record['target'], record['weight']
        for node in [source, target]:
            if node not in net.nodes:
                if node in first_degree_neighbors:
                    color, size = '#00a1cb', 18
                else:
                    color, size = '#E0E0E0', 12
                net.add_node(node, label=node, color=color, size=size)
        net.add_edge(source, target, value=weight, title=f"Co-occurred: {weight} times")

    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.01,
          "springLength": 100
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    
    # --- FIX IS HERE: Manually handle HTML generation with UTF-8 ---
    try:
        # Step 1: Generate the HTML content directly into a string variable
        html_content_string = net.generate_html(notebook=True)

        # Step 2: Write this string to a temporary file, explicitly using UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as tmpfile:
            tmpfile.write(html_content_string)
            html_path = tmpfile.name
        
        # Step 3: Read the content back from the temporary file to be displayed by Streamlit
        with open(html_path, 'r', encoding='utf-8') as f:
            final_html = f.read()
            
        return final_html
        
    except Exception as e:
        return f"<p>Error generating graph: {e}</p>"

# --- SENTIMENT PREDICTION FUNCTION (for baseline model) ---
def predict_sentiment(text):
    if not tokenizer or not model: return None, None
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    prob_df = pd.DataFrame({
        'Sentiment': [label_map[i] for i in range(len(label_map))],
        'Probability': probabilities.flatten().numpy()
    })
    return label_map[prediction], prob_df

# --- STREAMLIT USER INTERFACE (UI) ---

st.title("Malay Sentiment and Context Analysis")
st.markdown("This web application utilizes a **fine-tuned IndoBERT model** for sentiment analysis and a Neo4j graph database to explore word contexts.")

st.sidebar.header("About")
st.sidebar.info("This app demonstrates two key functionalities from the research project: (1) Live Sentiment Prediction with the best text-only model and (2) An advanced, interactive Graph-Based Word Context Explorer.")

# --- 1. Live Sentiment Prediction ---
st.header("1. Live Sentiment Prediction")
user_input = st.text_area("Enter a Malay sentence here:", "Saya sangat gembira dengan perkhidmatan yang diberikan.", height=100)

if st.button("Analyze Sentiment"):
    if user_input.strip() and model:
        with st.spinner("Analyzing..."):
            predicted_label, probs_df = predict_sentiment(user_input)
            if predicted_label == 'positive': st.success(f"**Predicted Sentiment: Positive**")
            elif predicted_label == 'negative': st.error(f"**Predicted Sentiment: Negative**")
            else: st.info(f"**Predicted Sentiment: Neutral**")
            st.write("Probability Distribution:")
            st.bar_chart(probs_df.set_index('Sentiment'))
    else:
        st.warning("Please enter a sentence to analyze.")

# --- 2. Interactive Co-occurrence Graph ---
st.header("2. Word Context Explorer")
word_input = st.text_input("Enter a single word to explore its context:", "makan")

# --- IMPROVEMENT: Add a slider to select number of hops ---
hop_slider = st.slider("Select neighborhood levels (hops):", min_value=1, max_value=2, value=1)

if st.button("Generate Context Graph"):
    if not driver:
        st.error("Connection to the graph database failed. This feature is unavailable.")
    elif word_input.strip():
        word_to_search = word_input.strip().split()[0]
        with st.spinner(f"Fetching context for '{word_to_search}'..."):
            graph_data = get_cooccurrence_data(driver, word_to_search, hops=hop_slider)
            if graph_data:
                html_content = generate_graph_html(graph_data, word_to_search)
                if html_content:
                    st.subheader(f"Co-occurrence Graph for '{word_to_search}' ({hop_slider} level(s))")
                    components.html(html_content, height=620, scrolling=True)
                else:
                    st.error("Failed to generate the graph visualization.")
            else:
                st.warning(f"No co-occurrence data found for '{word_to_search}'. Try another common word.")
    else:
        st.warning("Please enter a single word.")