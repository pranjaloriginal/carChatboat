
from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
import psycopg2
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings  
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import re
import google.generativeai as genai


load_dotenv()


app = Flask(__name__)


openai_api_key = os.getenv('OPENAI_API_KEY')


llm = OpenAI(api_key=openai_api_key)


def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST')
        )
        print("Database connection successful!")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def fetch_car_data():
    conn = get_db_connection()
    if conn is None:
        return []
    
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM cars_table;")
        rows = cursor.fetchall()
        
        
        for row in rows:
            print(f"Fetched row: {row}")

        return rows
    except Exception as e:
        print(f"Error fetching car data: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def create_docsearch():
    documents = []
    car_data = fetch_car_data()
    
    
    if not car_data:
        print("No car data found!")
        return None

    for row in car_data:
        
        if len(row) != 9: 
            print(f"Row has unexpected structure: {row}")
            continue

        
        car_info = {
            "text": f"{row[1]} ({row[2]} {row[3]}), Year: {row[4]}, Engine: {row[5]}, Starting Price: {row[6]} Lakhs, Ending Price: {row[7]} Lakhs, Fuel: {row[8]}",
            "id": str(row[0])
        }
        
     
        print(f"Document created: {car_info}")
        documents.append(car_info)

   
    try:
        embeddings = OpenAIEmbeddings()
        texts = [doc["text"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        print("Creating Chroma vector store...")
        return Chroma.from_texts(texts, embeddings, ids=ids)
    except Exception as e:
        print(f"Error creating Chroma vector store: {e}")
        return None


docsearch = create_docsearch()
if docsearch:
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    print("QA chain initialized successfully.")
else:
    print("Document search creation failed!")
    
def generate_sql(user_input):
    genai.configure(api_key="genai-key")

    
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="You are an SQL Agent who can create sql query based on user input\n\nI have a table cars_table with column names: car_name, make, model, year, engine_type, fuel_type, starting_price, ending_price.\n\n I want to find car between price range from 50 to 80 lakhs the SQL query is = SELECT car_name,starting_price,ending_price  FROM cars_table  WHERE starting_price>=50 AND ending_price<=80;\n\n Give me Details of car = SELECT * FROM car_table WHERE car_name in ('car') ",
    )

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    "which is the most expensive car",
                ],
            },
            {
                "role": "model",
                "parts": [
                    "```sql\nSELECT car_name\nFROM car_table\nORDER BY ending_price DESC\nLIMIT 1;\n```",
                ],
            },
        ]
    )

    response = chat_session.send_message(user_input)

    print(response.text)
    return response

def formatter(user_input):
    genai.configure(api_key="genai-key")

 
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        system_instruction="You are an Agent who can create my responses in a proper english phrase how a ai bot should reponse\n\n [('Toyota Land Cruiser',), ('Toyota Vellfire',), ('Toyota Sequoia',)] in a way like a proper english answer   \n\n and present * **Toyota Tundra** like Toyota Tundra \n\n and their price currency should replace $ with Rs. and in lakhs\n\n if asked a general question ask by yourself from the internet",
    )

    chat_session = model.start_chat(
        history=[
        ]
    )

    response = chat_session.send_message(user_input)

    print(response.text)
    return response.text


@app.route('/', methods=['GET', 'POST'])
def index():
    conn = get_db_connection()
    if conn is None:
        return []
    
    cursor = conn.cursor()
        
    reply = ""
    if request.method == 'POST':
        user_input = request.form.get('message')

        if not user_input:
            reply = "Please enter a message."
        else:
            print(f"User query: {user_input}")
            
            generated_response = generate_sql(user_input)
            query_regex = r"(SELECT[\s\S]+?;)"
            match = re.search(query_regex, generated_response.text)
            # generated_query = re.search(r'(SELECT .*?;)', generated_response, re.IGNORECASE)
            # print("generated_query : ", generated_query)
            
            if match:
                generated_query = match.group(1)
                print(generated_query)
                try:
                    cursor.execute(generated_query)
                    rows = cursor.fetchall()
                    print(rows)
                    k = formatter(str(rows))
                    reply = k  
                except Exception as e:
                    print(f"Error executing query: {e}")
                    reply = "There was an error executing the SQL query."
            else:
                print("No query found.")
                reply = "Please ask questions about cars."
                

            # if generated_query:
            #     # generated_query = generated_query.group(1)
            #     try:
            #         cursor.execute(generated_query)
            #         rows = cursor.fetchall()
            #         reply = rows  # Modify this if you want to format the output
            #     except Exception as e:
            #         print(f"Error executing query: {e}")
            #         reply = "There was an error executing the SQL query."
            # else:
            #     reply = "Could not generate a valid SQL query."

    return render_template('index.html', reply=reply)

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)

