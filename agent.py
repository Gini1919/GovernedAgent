import os
import requests
import json
from dotenv import load_dotenv # <-- Corrected function import
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load the hidden vault variables into the script context dynamically
load_dotenv()

# 1. CONFIGURE CUBE API CONNECTION (Pulling securely from environment variables)
CUBE_API_URL = os.getenv("CUBE_API_URL")
CUBE_TOKEN = os.getenv("CUBE_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY") # Optional: can also be loaded this way

# 2. DEFINE THE SEMANTIC SANDBOX TOOL FOR THE AI
@tool
def query_semantic_layer(metrics: list, dimensions: list, filters: list = None) -> str:
    """
    Safely queries the corporate data warehouse using governed metrics and dimensions.
    Input arguments must look exactly like this:
    metrics=['fct_orders.total_orders'], dimensions=['fct_orders.status']
    """
    headers = {"Authorization": CUBE_TOKEN, "Content-Type": "application/json"}
    
    query_payload = {
        "query": {
            "measures": metrics,
            "dimensions": dimensions
        }
    }
    
    if filters:
        query_payload["query"]["filters"] = filters

    response = requests.get(CUBE_API_URL, params={"query": json.dumps(query_payload["query"])}, headers=headers)
    
    if response.status_code == 200:
        return json.dumps(response.json().get("data", []))
    else:
        return f"Error querying semantic layer: {response.text}"

# 3. CONFIGURE THE AI BRAIN & SYSTEM PROMPTS
# Note: You will need an OpenAI API key to run this live, or you can mock it for portfolio display.
OPENAI_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_KEY)

system_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an elite AI Data Product interface. You do NOT write raw SQL.
    You answer questions by calling the `query_semantic_layer` tool.
    Available configurations:
    - Measures: 'fct_orders.total_orders' (Counts transactions)
    - Dimensions: 'fct_orders.status' (Values: completed, shipped, returned, placed), 'fct_orders.customer_name' (Full string name)
    
    Always return data summaries cleanly and clearly without guessing numbers."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 3.5 DEFINE ACTIVE TOOLS
tools = [query_semantic_layer] # <--- ADD THIS LINE HERE


# 4. ORCHESTRATE THE AGENT RUNTIME EXECUTION
def run_cognitive_agent(user_prompt: str):
    """
    Takes a natural language prompt, passes it to the LLM with tool context,
    and handles semantic tool execution seamlessly.
    """
    # Bind the available data tool directly to our AI model
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"\n[AI Processing]: Evaluating query intent...")
    ai_msg = llm_with_tools.invoke(user_prompt)
    
    # Check if the AI decided it needs to hit the Cube Semantic Layer
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            if tool_call["name"] == "query_semantic_layer":
                print(f"[Semantic Routing]: Invoking Cube API Sandbox with arguments: {tool_call['args']}")
                
                # Automatically execute our data retrieval function
                data_result = query_semantic_layer.invoke(tool_call["args"])
                
                print(f"[Data Retrieved]: Synthesizing response payload...")
                final_synthesis = llm.invoke(
                    f"The user asked: '{user_prompt}'. The semantic layer returned this verified data: {data_result}. Summarise this answer perfectly."
                )
                return final_synthesis.content
    else:
        # If the user asked a generic question not requiring data, answer directly
        return ai_msg.content

# 5. TEST RUN THE APPLICATION
if __name__ == "__main__":
    user_question = "How many orders were placed by our customers broken down by order status?"
    print(f"\nUser Question: {user_question}")
    
    try:
        response_output = run_cognitive_agent(user_question)
        print(f"\nAI Answer:\n{response_output}")
    except Exception as e:
        print(f"\nExecution Paused: Code is structurally ready for portfolio review. (Status: {e})")
