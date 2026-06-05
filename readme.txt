# ===================================================================
# MLex Project: System Access and Credentials
# ===================================================================

This document contains the necessary credentials and information to access
the project's database and run the associated applications.

-------------------------------------------------------------------

### 1. Neo4j AuraDB (Graph Database)

This is the cloud-hosted graph database containing the co-occurrence graph,
token properties, and sentence data.

- **Type:** Neo4j AuraDB (Cloud)
- **URI:** neo4j+s://651ad0cf.databases.neo4j.io
- **Username:** neo4j
- **Password:** Ih60kt7LGhfYDav6F-gQl1HftueVl-uVlJbxI0pmb20

You can use these credentials to connect to the database via the Neo4j Browser
or any official Neo4j driver.

-------------------------------------------------------------------

### 2. Streamlit Application (Local Deployment)

To run the deployed web application, navigate to the project's root 
directory (`AAAfyp`) in your terminal and run the following command:

`streamlit run app.py`

The application is configured to connect to the Neo4j database using the
credentials listed above, which are hardcoded in the `app.py` script for
demonstration purposes. In a production environment, these would be managed
as secure environment variables or secrets.

-------------------------------------------------------------------