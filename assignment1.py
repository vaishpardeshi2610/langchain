from pydantic import BaseModel, Field
from typing import List
from langchain_groq.chat_models import ChatGroq
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Define a Pydantic model for a single movie
class Movie(BaseModel):
    movie_name: str = Field(description="Name of the movie")
    movie_director: str = Field(description="Name of the director")

# Define a model for the list of movies
class HollywoodMovies(BaseModel):
    movies: List[Movie]

# Initialize the JsonOutputParser with Pydantic model
output_parser = JsonOutputParser(pydantic_object=HollywoodMovies)

# Example context and format instructions
context = """Hollywood's movies showcase the pinnacle of cinematic achievement, blending exceptional storytelling, groundbreaking visuals, and stellar performances. Topping the list, Avatar remains a masterpiece of technological innovation, transporting audiences to Pandora with its immersive 3D visuals and a story centered on environmentalism and interplanetary conflict. Titanic, another James Cameron classic, is a timeless tale of love and tragedy aboard the ill-fated ship, capturing hearts globally with its emotional depth and unforgettable soundtrack. Christopher Nolan's The Dark Knight redefined superhero movies, offering a gritty and complex narrative, highlighted by Heath Ledger's Oscar-winning portrayal of the Joker. Marvel Studios' Avengers: Endgame became a cultural phenomenon, masterfully concluding a decade-long saga with its thrilling battles and heartfelt moments, making it the highest-grossing superhero film ever. Lastly, Inception, another Nolan creation, is a mind-bending journey into dreams within dreams, celebrated for its intricate plot, stunning visuals, and Hans Zimmer's iconic score. These films not only dominated the box office but also left indelible marks on pop culture, pushing the boundaries of what cinema can achieve. From romance and action to fantasy and innovation, Hollywood's finest have set benchmarks that continue to inspire filmmakers and captivate audiences worldwide."""

format_instructions = """
Please extract the following details from the context:
- Name of the movie
- Name of the director
The output should only include a list with the following structure, and if the movie_name and movie_director are not given in the context, use "NULL":
{
"movies":
[
    {
    "movie_name": "Movie Name",
    "movie_director": "Director Name"
    },
    {
    "movie_name": "Movie Name",
    "movie_director": "Director Name"
    }
]
}
"""

# Define the prompt template
prompt = """
    You are an assistant that has to extract the following required information from the context provided.
    Context: {context}
    Format Instructions: {format_instructions}
    Please return only the JSON output without any additional text.
"""

prompt_template = PromptTemplate(
    template=prompt, 
    input_variables=["context", "format_instructions"]
)

# Initialize the ChatGroq model with Groq API key
llm = ChatGroq(
    model_name="llama3-70b-8192",  
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Set up the chain
chain = prompt_template | llm | output_parser

# Invoke the chain
response = chain.invoke({"context": context, "format_instructions": format_instructions})

# Output the parsed response
print(response)
