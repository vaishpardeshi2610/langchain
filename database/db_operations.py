import os
from dotenv import load_dotenv
import asyncpg
import json

def load_env_variables():
    """Load environment variables from a .env file."""
    load_dotenv()

def get_db_params():
    """Return database connection parameters."""
    return {
        "database": os.getenv("DBNAME"),  # Changed from "dbname" to "database"
        "user": os.getenv("USER"),
        "password": os.getenv("PASSWORD"),
        "host": os.getenv("HOST"),
        "port": os.getenv("PORT"),
    }



class DatabaseManager:
    """Manages PostgreSQL connections and operations."""

    def __init__(self, db_params):
        self.db_params = db_params

    async def store_embedding(self, video_id, transcript, embedding):
        """
        Stores the transcript and embedding into PostgreSQL.
        """
        conn = None
        try:
            conn = await asyncpg.connect(
                user=self.db_params["user"],
                password=self.db_params["password"],
                database=self.db_params["database"],  # Corrected this line
                host=self.db_params["host"],
                port=self.db_params["port"]
            )

            # Create the table if it doesn't exist
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_embeddings (
                id SERIAL PRIMARY KEY,
                video_id VARCHAR(255),
                transcript TEXT,
                embedding JSON
            );
            """)

            # Convert embedding to JSON and insert into the table
            embedding_json = json.dumps(embedding)
            await conn.execute(
                "INSERT INTO transcript_embeddings (video_id, transcript, embedding) VALUES ($1, $2, $3)",
                video_id, transcript, embedding_json
            )
            print("Embedding stored successfully!")
        except Exception as e:
            print("Error storing embedding:", e)
        finally:
            if conn:
                await conn.close()