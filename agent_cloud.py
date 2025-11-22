"""
TheLook Ecommerce AI Data Analyst Agent - Cloud Version (ENHANCED FORMATTING + CONVERSATIONAL)
Compatible with langgraph 0.0.26
UPDATED FOR DEPLOYMENT - FIXED CREDENTIALS
"""

import logging
import re
import warnings
import os
import json
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt.tool_executor import ToolExecutor
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import Literal, TypedDict, Annotated, Sequence

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

PRODUCTION_MODE = True
if PRODUCTION_MODE:
    logging.basicConfig(level=logging.CRITICAL)
    for name in ['httpx', 'httpcore', 'google', 'urllib3', 'langchain', 'langgraph']:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# Define state
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

# Get API keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Initialize BigQuery with proper credential handling for deployment
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not PROJECT_ID:
    raise ValueError("GOOGLE_CLOUD_PROJECT not found in environment variables")

# Get GCP credentials from environment variable
gcp_json_str = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
if not gcp_json_str:
    raise ValueError("GCP_SERVICE_ACCOUNT_JSON not found in environment variables")

try:
    # Parse JSON credentials
    credentials_dict = json.loads(gcp_json_str)
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    bq_client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    logger.info("✅ BigQuery initialized successfully")
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in GCP_SERVICE_ACCOUNT_JSON: {str(e)}")
except Exception as e:
    raise ValueError(f"Failed to initialize BigQuery: {str(e)}")

ECOMMERCE_TABLES = {
    "users": "bigquery-public-data.thelook_ecommerce.users",
    "orders": "bigquery-public-data.thelook_ecommerce.orders",
    "order_items": "bigquery-public-data.thelook_ecommerce.order_items",
    "products": "bigquery-public-data.thelook_ecommerce.products",
    "inventory_items": "bigquery-public-data.thelook_ecommerce.inventory_items",
    "distribution_centers": "bigquery-public-data.thelook_ecommerce.distribution_centers",
    "events": "bigquery-public-data.thelook_ecommerce.events",
}

DEFAULT_LIMIT = 100

schema_snippet = """
TheLook Ecommerce Schema:

Available tables (use these exact names):
- users: id, first_name, last_name, email, age, gender, state, city, country, latitude, longitude, traffic_source, created_at
- orders: order_id, user_id, status (Complete, Cancelled, Returned, Processing, Shipped), created_at, returned_at, shipped_at, delivered_at, num_of_item
- order_items: id, order_id, user_id, product_id, inventory_item_id, status, created_at, shipped_at, delivered_at, returned_at, sale_price
- products: id, cost, category, name, brand, retail_price, department, sku, distribution_center_id
- inventory_items: id, product_id, created_at, sold_at, cost, product_category, product_name, product_brand, product_retail_price, product_department, product_sku, product_distribution_center_id
- distribution_centers: id, name, latitude, longitude
- events: id, user_id, sequence_number, session_id, created_at, ip_address, city, state, postal_code, browser, traffic_source, uri, event_type
"""

SYSTEM_PROMPT = f"""You are a BigQuery SQL expert generating queries for TheLook Ecommerce data.

CRITICAL RULES:
1. Use ONLY simple table names: users, orders, order_items, products, inventory_items, distribution_centers, events
2. DO NOT use full table paths
3. DO NOT use backticks in your query
4. Return ONLY the SQL query - no explanations, no markdown
5. Always add LIMIT {DEFAULT_LIMIT} unless user specifies different limit
6. For status filtering, use exact values: 'Complete', 'Cancelled', 'Returned', 'Processing', 'Shipped'
7. For date/year filtering, use EXTRACT(YEAR FROM date_column) = year_value or DATE() functions
8. When filtering by year, use: WHERE EXTRACT(YEAR FROM created_at) = 2023

Examples:
Question: "Top products in 2023"
SQL: SELECT product_name, COUNT(*) as sales_count FROM inventory_items WHERE EXTRACT(YEAR FROM created_at) = 2023 AND sold_at IS NOT NULL GROUP BY product_name ORDER BY sales_count DESC LIMIT 10

Question: "Revenue by category in 2023"
SQL: SELECT product_category, SUM(sale_price) as revenue FROM order_items WHERE status = 'Complete' AND EXTRACT(YEAR FROM created_at) = 2023 GROUP BY product_category ORDER BY revenue DESC LIMIT 10
"""

# ============================================================================
# UTILITY FUNCTIONS (same as before)
# ============================================================================

def _strip_code_fences(sql: str) -> str:
    s = sql.strip()
    s = re.sub(r"^```(?:sql|SQL)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"```\s*$", "", s, flags=re.MULTILINE)
    return s.strip()

def _transform_sql_internal(sql: str) -> str:
    s = _strip_code_fences(sql)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace('`', '')
    s = s.replace('bigquery-public-data.thelook_ecommerce.', '')
    
    for short_name, full_name in ECOMMERCE_TABLES.items():
        pattern = rf'\b{short_name}\b'
        replacement = f'`{full_name}`'
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)
    
    if "LIMIT" not in s.upper():
        s = s.rstrip(";") + f" LIMIT {DEFAULT_LIMIT};"
    
    return s.strip()

def _is_safe_sql_internal(sql: str) -> bool:
    """Check if SQL is safe (read-only)"""
    sql_lower = sql.lower()
    
    # Check for dangerous keywords at word boundaries
    dangerous_patterns = [
        r'\bdrop\b', r'\bdelete\b', r'\binsert\b', r'\bupdate\b', 
        r'\balter\b', r'\bcreate\b', r'\btruncate\b', r'\bgrant\b',
        r'\brevoke\b', r'\bexec\b', r'\bexecute\b'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, sql_lower):
            return False
    
    return True

def _format_sql_readable(sql: str) -> str:
    s = sql.strip()
    for kw in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT', 'JOIN', 'LEFT JOIN', 'INNER JOIN']:
        s = re.sub(rf'\s+({kw})\s+', rf'\n{kw} ', s, flags=re.IGNORECASE)
    return s

def _execute_sql(sql: str):
    if not _is_safe_sql_internal(sql):
        raise ValueError("SQL contains dangerous keywords")
    
    try:
        query_job = bq_client.query(sql)
        df = query_job.result().to_dataframe()
        return df
    except Exception as e:
        logger.error(f"SQL execution failed: {sql}")
        logger.error(f"Error: {str(e)}")
        raise

def _detect_query_type(question: str) -> str:
    question_lower = question.lower()
    
    list_keywords = ['top', 'list', 'show', 'display', 'get', 'find', 'what are', 'which', 'most', 'highest', 'lowest']
    narrative_keywords = ['why', 'how', 'explain', 'describe', 'analyze', 'what is', 'what does']
    
    for keyword in list_keywords:
        if keyword in question_lower:
            return "list"
    
    for keyword in narrative_keywords:
        if keyword in question_lower:
            return "narrative"
    
    return "list"

def _format_as_list(df, question: str, row_count: int) -> str:
    title = question.strip('?').strip()
    if not title[0].isupper():
        title = title.capitalize()
    
    result = f"**{title}**\n\n"
    columns = df.columns.tolist()
    
    for idx, row in df.head(20).iterrows():
        result += f"| {idx + 1} | "
        result += " | ".join([str(row[col]) for col in columns])
        result += " |\n"
    
    if row_count > 20:
        result += f"\n*(Only top 20 of {row_count} total results shown)*"
    
    return result

def _format_as_narrative(df, question: str, row_count: int) -> str:
    df_str = df.head(20).to_string(index=False)
    return f"Based on the data ({row_count} rows):\n\n{df_str}"

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

@tool
def chat_with_user(message: str) -> str:
    """Use this tool for casual conversation, greetings, and personal questions."""
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.7, api_key=GROQ_API_KEY)
    
    conversation_prompt = f"""You are SmartLook, a friendly AI assistant for TheLook Ecommerce data.

About you:
- Name: SmartLook
- Purpose: Help analyze TheLook Ecommerce data
- Creator: Rafiq Naufal Kastara, Data Scientist
- Technology: LangChain, LangGraph, Groq LLM, Google BigQuery

Respond warmly to: {message}

Keep responses brief (2-4 sentences)."""
    
    try:
        response = llm.invoke(conversation_prompt)
        return response.content.strip()
    except Exception as e:
        return "I'm here to help! Feel free to ask me anything about TheLook ecommerce data! 😊"

@tool
def answer_ecommerce_question(question: str) -> str:
    """Query and analyze TheLook Ecommerce database to answer business questions."""
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, api_key=GROQ_API_KEY)
    sql_prompt = SYSTEM_PROMPT + "\n\nSchema:\n" + schema_snippet + "\n\nQuestion: " + question + "\n\nSQL:"
    
    try:
        response = llm.invoke(sql_prompt)
        raw_sql = response.content.strip()
        clean_sql = _transform_sql_internal(raw_sql)
        df = _execute_sql(clean_sql)
        
        if df.empty:
            return f"No results found.\n\n**SQL Query:**\n```sql\n{_format_sql_readable(clean_sql)}\n```"
        
        query_type = _detect_query_type(question)
        row_count = len(df)
        
        # Format data for analysis
        if query_type == "list":
            df_str = _format_as_list(df, question, row_count)
        else:
            df_str = _format_as_narrative(df, question, row_count)
        
        # Generate insights with LLM
        analysis_prompt = f"""Analyze this TheLook Ecommerce data and provide insights.

Question: {question}

Data:
{df.head(10).to_string(index=False)}

Format your response EXACTLY like this:

**[Title based on question]**

1. [Item 1] - [value]
2. [Item 2] - [value]
3. [Item 3] - [value]
...

**Key Insights:**

• [First key insight - wrap important keywords in **bold**]
• [Second insight - wrap important numbers/terms in **bold**]
• [Third insight - wrap key findings in **bold**]

CRITICAL: Wrap important keywords, numbers, brands, categories, and key terms in **bold** (e.g., **Denim**, **17 units**, **men's apparel**).
Keep it clean, structured, and professional. Use bullet points (•) for insights."""

        try:
            analysis_response = llm.invoke(analysis_prompt)
            analysis = analysis_response.content.strip()
        except Exception as e:
            # Fallback to just showing data
            analysis = df_str
        
        formatted_sql = _format_sql_readable(clean_sql)
        return f"{analysis}\n\n**SQL Query:**\n```sql\n{formatted_sql}\n```"
        
    except Exception as e:
        error_msg = str(e)
        if 'clean_sql' in locals():
            return f"❌ Error: {error_msg[:500]}\n\n**SQL Query:**\n```sql\n{_format_sql_readable(clean_sql)}\n```"
        return f"❌ Error: {error_msg[:500]}"

@tool
def generate_and_show_sql(question: str) -> str:
    """Generate and validate SQL query without executing full analysis."""
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, api_key=GROQ_API_KEY)
    sql_prompt = SYSTEM_PROMPT + "\n\nSchema:\n" + schema_snippet + "\n\nQuestion: " + question + "\n\nSQL:"
    
    try:
        response = llm.invoke(sql_prompt)
        raw_sql = response.content.strip()
        clean_sql = _transform_sql_internal(raw_sql)
        df = _execute_sql(clean_sql)
        formatted_sql = _format_sql_readable(clean_sql)
        
        # Create a friendly introduction
        intro = f"Here's the SQL for \"{question}\"\n\n"
        
        if df.empty:
            return f"{intro}**SQL Query:**\n```sql\n{formatted_sql}\n```\n\n⚠️ Query returns no data."
        
        return f"{intro}**SQL Query:**\n```sql\n{formatted_sql}\n```\n\n**Validated** - returns {len(df)} rows."
    except Exception as e:
        return f"❌ Error: {str(e)[:300]}"

# ============================================================================
# AGENTIC WORKFLOW - Without Native Function Calling
# ============================================================================

tools = [chat_with_user, answer_ecommerce_question, generate_and_show_sql]
tools_by_name = {tool.name: tool for tool in tools}

# Global variable to track last tool call (for debugging/testing)
_last_tool_call = None

# Conversation memory - stores recent context
_conversation_history = []
MAX_HISTORY = 10  # Keep last 10 exchanges

def add_to_history(role: str, content: str):
    """Add message to conversation history"""
    global _conversation_history
    _conversation_history.append({"role": role, "content": content})
    # Keep only recent history
    if len(_conversation_history) > MAX_HISTORY * 2:  # 2 messages per exchange
        _conversation_history = _conversation_history[-MAX_HISTORY * 2:]

def get_conversation_context() -> str:
    """Get formatted conversation history for context"""
    if not _conversation_history:
        return ""
    
    context = "Recent conversation:\n"
    for msg in _conversation_history[-6:]:  # Last 3 exchanges
        context += f"{msg['role']}: {msg['content'][:150]}...\n"
    return context

def clear_history():
    """Clear conversation history"""
    global _conversation_history
    _conversation_history = []

# Create LLM
llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)

# Tool routing system prompt
ROUTING_SYSTEM_PROMPT = """You are a routing agent that decides which tool to call based on the user's question.

You have access to recent conversation history to understand context and follow-up questions.

Available tools:
1. chat_with_user - For greetings, personal questions, thanks, and general conversation
   Examples: "Hi!", "Who are you?", "Thanks!", "What can you do?"

2. answer_ecommerce_question - For data analysis questions about orders, customers, products, sales
   Examples: "Top 10 products?", "Revenue by category?", "How many customers?"
   IMPORTANT: When user asks follow-up questions like "show me trends for that" or "analyze X further", 
   use context from conversation history to understand what they're referring to.

3. generate_and_show_sql - For explicitly requesting SQL queries
   Examples: "Show me the SQL", "Generate query for...", "What's the SQL?"

Respond ONLY with a JSON object in this format:
{"tool": "tool_name", "args": {"param": "value"}}

When the user asks a follow-up question (e.g., "show trends for that", "what about Jeans?", "analyze that category"), 
YOU MUST incorporate context from the conversation history into the query.

Examples:
User: "Hello!"
Response: {"tool": "chat_with_user", "args": {"message": "Hello!"}}

User: "What are top products?"
Response: {"tool": "answer_ecommerce_question", "args": {"question": "What are top products?"}}

User: "Show SQL for revenue"
Response: {"tool": "generate_and_show_sql", "args": {"question": "Show SQL for revenue"}}

FOLLOW-UP CONTEXT EXAMPLES:
Previous: "Show revenue by category" (Result showed Jeans as #2)
User: "Show trends for Jeans in 2024"
Response: {"tool": "answer_ecommerce_question", "args": {"question": "Show revenue trends for Jeans category in 2024"}}

Previous: "Top selling products"
User: "What about their prices?"
Response: {"tool": "answer_ecommerce_question", "args": {"question": "What are the prices of top selling products?"}}"""

def agent_node(state: AgentState):
    """LLM decides which tool to call using JSON response"""
    global _last_tool_call
    
    messages = state["messages"]
    user_message = messages[-1].content
    
    # Get conversation context
    context = get_conversation_context()
    
    # Ask LLM to route with context
    routing_prompt = f"""{ROUTING_SYSTEM_PROMPT}

{context}

Current User Question: {user_message}

Analyze the conversation history and current question. If this is a follow-up question referring to previous context (e.g., "show trends for that", "what about Jeans", "analyze that further"), incorporate that context into your tool arguments.

Response:"""
    
    try:
        response = llm.invoke([SystemMessage(content=routing_prompt)])
        response_text = response.content.strip()
        
        # Parse JSON response
        # Remove markdown code fences if present
        response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        
        tool_decision = json.loads(response_text)
        tool_name = tool_decision.get("tool")
        tool_args = tool_decision.get("args", {})
        
        # Store for testing/debugging
        _last_tool_call = {"name": tool_name, "args": tool_args}
        
        # Create a mock AIMessage with tool_calls in additional_kwargs
        ai_msg = AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [{
                    "name": tool_name,
                    "args": tool_args,
                    "id": "call_1"
                }]
            }
        )
        
        return {"messages": [ai_msg]}
        
    except Exception as e:
        # Fallback: if routing fails, treat as conversational
        logger.error(f"Routing failed: {e}")
        _last_tool_call = {"name": "chat_with_user", "args": {"message": user_message}}
        
        fallback_msg = AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [{
                    "name": "chat_with_user",
                    "args": {"message": user_message},
                    "id": "call_1"
                }]
            }
        )
        return {"messages": [fallback_msg]}

def tool_node(state: AgentState):
    """Execute the tool"""
    last_message = state["messages"][-1]
    outputs = []
    
    tool_calls = last_message.additional_kwargs.get("tool_calls", [])
    
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call.get("id", "call_1")
        
        tool = tools_by_name.get(tool_name)
        
        if tool:
            try:
                result = tool.invoke(tool_args)
                outputs.append(ToolMessage(content=str(result), tool_call_id=tool_id))
            except Exception as e:
                outputs.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_id))
        else:
            outputs.append(ToolMessage(content=f"Tool {tool_name} not found", tool_call_id=tool_id))
    
    return {"messages": outputs}

def should_continue(state: AgentState) -> Literal["tool_node", END]:
    """Check if we should continue"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'additional_kwargs') and 'tool_calls' in last_message.additional_kwargs:
        return "tool_node"
    
    return END

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tool_node": "tool_node", END: END})
workflow.add_edge("tool_node", END)

agent_executor = workflow.compile()

# ============================================================================
# PUBLIC API
# ============================================================================

def ask(question: str) -> str:
    """Ask a question to the AI agent"""
    try:
        # Add user message to history
        add_to_history("User", question)
        
        messages = [HumanMessage(content=question)]
        result = agent_executor.invoke({"messages": messages})
        
        # Get the final response (from ToolMessage)
        response = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, ToolMessage):
                response = msg.content
                break
        
        if not response:
            response = result["messages"][-1].content
        
        # Add assistant response to history
        add_to_history("Assistant", response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error in ask(): {str(e)}")
        error_response = f"❌ Error: {str(e)}"
        add_to_history("Assistant", error_response)
        return error_response

def check_connection():
    """Check Groq and BigQuery connections"""
    status = {"groq": False, "bigquery": False, "details": {}}
    
    try:
        llm_test = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
        llm_test.invoke("test")
        status["groq"] = True
        status["details"]["groq"] = f"Connected - Model: {GROQ_MODEL}"
    except Exception as e:
        status["details"]["groq"] = f"Failed: {str(e)[:100]}"
    
    try:
        query_job = bq_client.query("SELECT 1 as test")
        query_job.result()
        status["bigquery"] = True
        status["details"]["bigquery"] = f"Connected - Project: {PROJECT_ID}"
    except Exception as e:
        status["details"]["bigquery"] = f"Failed: {str(e)[:100]}"
    
    return status

# Helper to see tool routing
def get_tool_calls(message):
    """Extract tool calls from message"""
    if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
        return message.additional_kwargs['tool_calls']
    return []

def get_last_tool_call():
    """Get the last tool call made by the agent (for testing)"""
    return _last_tool_call

def get_conversation_history():
    """Get the conversation history (for debugging)"""
    return _conversation_history