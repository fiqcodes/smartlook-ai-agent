"""
TheLook Ecommerce AI Data Analyst Agent - Cloud Version (ENHANCED FORMATTING + CONVERSATIONAL)
Compatible with langgraph 0.0.26
UPDATED FOR DEPLOYMENT
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

# Try multiple credential methods
try:
    # Method 1: From JSON string environment variable (Render/Railway)
    gcp_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if gcp_json:
        credentials_dict = json.loads(gcp_json)
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        bq_client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        logger.info("✅ BigQuery initialized with JSON credentials")
    else:
        # Method 2: Default credentials (for local development)
        bq_client = bigquery.Client(project=PROJECT_ID)
        logger.info("✅ BigQuery initialized with default credentials")
except Exception as e:
    logger.error(f"❌ Failed to initialize BigQuery: {str(e)}")
    raise

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

Important notes:
- Order status values: 'Complete', 'Cancelled', 'Returned', 'Processing', 'Shipped'
- Product categories: Accessories, Active, Blazers & Jackets, Clothing Sets, Dresses, Intimates, Jeans, Jumpsuits & Rompers, Leggings, Maternity, Outerwear & Coats, Pants, Pants & Capris, Plus, Shorts, Skirts, Sleep & Lounge, Socks, Socks & Hosiery, Suits, Suits & Sport Coats, Sweaters, Swim, Tops & Tees, Underwear
- For date filtering, use DATE() or TIMESTAMP functions
- Use simple table names like 'users', not full paths
"""

SYSTEM_PROMPT = f"""You are a BigQuery SQL expert generating queries for TheLook Ecommerce data.

CRITICAL RULES:
1. Use ONLY simple table names: users, orders, order_items, products, inventory_items, distribution_centers, events
2. DO NOT use full table paths like bigquery-public-data.thelook_ecommerce.users
3. DO NOT use backticks in your query
4. Return ONLY the SQL query - no explanations, no markdown
5. Always add LIMIT {DEFAULT_LIMIT} unless user specifies different limit
6. For status filtering, use exact values: 'Complete', 'Cancelled', 'Returned', 'Processing', 'Shipped'
7. For date comparisons, use DATE() or TIMESTAMP() functions

Examples:
Question: "Top 5 selling products"
SQL: SELECT product_name, COUNT(*) as sales_count FROM inventory_items WHERE sold_at IS NOT NULL GROUP BY product_name ORDER BY sales_count DESC LIMIT 5

Question: "Revenue by product category"
SQL: SELECT product_category, SUM(sale_price) as total_revenue FROM order_items WHERE status = 'Complete' GROUP BY product_category ORDER BY total_revenue DESC LIMIT 10
"""

CONVERSATIONAL_SYSTEM_PROMPT = """You are SmartLook, a friendly AI assistant specializing in TheLook Ecommerce data analysis.

Your personality:
- Friendly, helpful, and enthusiastic about e-commerce data
- Professional but approachable
- Expert in ecommerce metrics and SQL queries

About you:
- Name: SmartLook
- Purpose: Help users analyze TheLook Ecommerce data through natural language queries
- Capabilities: Query and analyze customers, orders, products, inventory, sales, revenue, and user behavior
- Data source: TheLook Ecommerce public dataset on Google BigQuery (a fictitious ecommerce clothing store)
- Creator: Rafiq Naufal Kastara, Data Scientist
- Technology: Built using LangChain, Groq LLM (openai/gpt-oss-120b), and Google BigQuery

When users ask about you:
- Be concise and friendly
- Mention your ability to analyze ecommerce data
- When asked about your creator, proudly mention: "I was created by Rafiq Naufal Kastara, a Data Scientist who built me to help people explore TheLook Ecommerce data easily!"
- Encourage them to ask data-related questions
- Use a conversational tone

Respond naturally to:
- Greetings (hello, hi, hey)
- Thanks/gratitude
- Questions about your identity
- Questions about your capabilities
- Questions about the data you work with
- Questions about your creator/developer
- General chitchat

Keep responses brief (2-4 sentences) unless more detail is requested."""

def _strip_code_fences(sql: str) -> str:
    """Remove markdown code fences from SQL"""
    s = sql.strip()
    s = re.sub(r"^```(?:sql|SQL)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"```\s*$", "", s, flags=re.MULTILINE)
    return s.strip()

def _transform_sql_internal(sql: str) -> str:
    """Transform and validate SQL query"""
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
    dangerous = ['drop', 'delete', 'insert', 'update', 'alter', 'create', 'truncate']
    return not any(kw in sql.lower() for kw in dangerous)

def _format_sql_readable(sql: str) -> str:
    """Format SQL for display"""
    s = sql.strip()
    for kw in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT', 'JOIN', 'LEFT JOIN', 'INNER JOIN']:
        s = re.sub(rf'\s+({kw})\s+', rf'\n{kw} ', s, flags=re.IGNORECASE)
    return s

def _execute_sql(sql: str):
    """Execute SQL query on BigQuery"""
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
    """Detect if query should return list format or narrative format"""
    question_lower = question.lower()
    
    # List triggers
    list_keywords = [
        'top', 'list', 'show', 'display', 'get', 'find',
        'what are', 'which', 'who are', 'give me',
        'most', 'highest', 'lowest', 'best', 'worst',
        'all', 'many'
    ]
    
    # Narrative triggers
    narrative_keywords = [
        'why', 'how', 'explain', 'describe', 'analyze',
        'what is', 'what does', 'tell me about',
        'compare', 'difference', 'relationship'
    ]
    
    # Check for list indicators
    for keyword in list_keywords:
        if keyword in question_lower:
            return "list"
    
    # Check for narrative indicators
    for keyword in narrative_keywords:
        if keyword in question_lower:
            return "narrative"
    
    # Default to list if unclear
    return "list"

def _format_as_list(df, question: str, row_count: int) -> str:
    """Format data as a structured list with title and items"""
    
    # Create title based on question
    title = question.strip('?').strip()
    if not title[0].isupper():
        title = title.capitalize()
    
    # Format the data as ranked list
    result = f"**{title}**\n\n"
    
    # Add column headers
    columns = df.columns.tolist()
    
    # Format each row as list item
    for idx, row in df.head(20).iterrows():
        result += f"| {idx + 1} | "
        result += " | ".join([str(row[col]) for col in columns])
        result += " |\n"
    
    if row_count > 20:
        result += f"\n*(Only top 20 of {row_count} total results shown)*"
    
    return result

def _format_as_narrative(df, question: str, row_count: int) -> str:
    """Format data as narrative paragraphs with insights"""
    df_str = df.head(20).to_string(index=False)
    return f"Based on the data ({row_count} rows):\n\n{df_str}"

@tool
def chat_with_user(message: str) -> str:
    """Handle general conversation, greetings, and questions about SmartLook's identity and capabilities.
    Use this tool when users:
    - Greet you (hi, hello, hey)
    - Thank you
    - Ask who you are or what you do
    - Ask about your capabilities
    - Ask about your creator
    - Ask what data you can access
    - Make general chitchat
    
    DO NOT use this for data analysis questions - use answer_ecommerce_question instead.
    """
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.7, api_key=GROQ_API_KEY)
    
    conversation_prompt = f"""{CONVERSATIONAL_SYSTEM_PROMPT}

User message: {message}

Respond naturally and helpfully. Keep it concise but friendly."""
    
    try:
        response = llm.invoke(conversation_prompt)
        return response.content.strip()
    except Exception as e:
        return "I'm here to help! Feel free to ask me anything about TheLook data and performance, or just say hello! 😊"

@tool
def answer_ecommerce_question(question: str) -> str:
    """Answer questions about TheLook Ecommerce data with analysis.
    Use this tool ONLY for questions that require querying the database:
    - Questions about customers, orders, products, inventory, sales, revenue
    - Statistical queries (top, most, highest, revenue, etc.)
    - Data analysis questions
    
    DO NOT use this for greetings or general conversation.
    """
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, api_key=GROQ_API_KEY)
    sql_prompt = SYSTEM_PROMPT + "\n\nSchema:\n" + schema_snippet + "\n\nQuestion: " + question + "\n\nSQL:"
    
    try:
        # Generate SQL
        response = llm.invoke(sql_prompt)
        raw_sql = response.content.strip()
        clean_sql = _transform_sql_internal(raw_sql)
        
        # Execute query
        df = _execute_sql(clean_sql)
        
        if df.empty:
            return f"No results found.\n\n{'='*60}\n**SQL Query:**\n{'='*60}\n```sql\n{_format_sql_readable(clean_sql)}\n```"
        
        # Detect query type
        query_type = _detect_query_type(question)
        row_count = len(df)
        
        # Prepare data for analysis
        if query_type == "list":
            df_str = _format_as_list(df, question, row_count)
        else:
            df_str = _format_as_narrative(df, question, row_count)
        
    except Exception as e:
        error_msg = str(e)
        if 'clean_sql' in locals():
            return f"❌ Error executing query: {error_msg[:500]}\n\n**SQL Query:**\n```sql\n{_format_sql_readable(clean_sql)}\n```"
        else:
            return f"❌ Error generating SQL: {error_msg[:500]}"
    
    # Generate analysis with improved prompt
    if query_type == "list":
        analysis_prompt = f"""Analyze this TheLook Ecommerce data and provide a well-structured response.

Question: {question}

Data:
{df_str}

FORMATTING INSTRUCTIONS:
1. Start with a clear title in **bold** (use the question as basis)
2. Present the data as a numbered/ranked list with clear formatting
3. THIS IS CRITICAL: After the main list, you MUST add exactly 5 empty lines (press Enter 5 times) before starting the Key Insights section
4. Then add a "Key Insights" section (also in **bold**) with 3-5 bullet points using • symbol highlighting:
   • Patterns or trends in the data
   • Notable observations
   • Interesting facts about customer behavior, sales, or products
5. DO NOT use ##, ###, --- or - for bullet points
6. DO NOT use asterisks (*) or any special characters around text - write cleanly
7. Use • (bullet point symbol) for all list items in Key Insights
8. Use only **bold** for titles and section headers
9. Mark important keywords/phrases in **bold** to highlight (these will be colored orange in HTML)
10. Keep insights concise and data-driven

Remember: This is a LIST query - present data in structured table/list format first, then add 5 BLANK LINES, then insights.

Example format you MUST follow:
**Top 5 Products**
1. Product one - $1000 revenue
2. Product two - $900 revenue

**Key Insights**
(No blank lines here)
• First insight
• Second insight"""
    else:
        analysis_prompt = f"""Analyze this TheLook Ecommerce data and provide a narrative explanation.

Question: {question}

Data:
{df_str}

FORMATTING INSTRUCTIONS:
1. Write in clear, well-structured paragraphs
2. Start with a direct answer to the question
3. Support your points with specific data from the results
4. Organize your response with clear sections using **bold** headers
5. DO NOT use ##, ###, or --- symbols
6. Use only **bold** for titles and section headers
7. Mark important keywords/phrases in **bold** to highlight (these will be colored orange in HTML)
8. End with a brief conclusion or summary
9. Don't add the blank lines between Key Insights: and the bullet points
10. Add the blank lines between last Key Insights: and the bullet points and the SQL Query: title

Remember: This is a NARRATIVE query - focus on explanation and insights, not just listing data."""
    
    try:
        response = llm.invoke(analysis_prompt)
        analysis = response.content.strip()

        # REMOVE ALL STRAY ASTERISKS (***, **, *)
        analysis = re.sub(r'\n\s*[\*]{1,3}\s*\n', '\n\n', analysis)
        analysis = re.sub(r'^\s*[\*]{1,3}\s*\n', '\n', analysis, flags=re.MULTILINE)
        analysis = re.sub(r'\n\s*[\*]{1,3}\s*$', '\n', analysis)

        # Force Key Insights
        analysis = re.sub(r'\*\*Key Insights\*\*:?', 'KEY_INSIGHTS_MARKER', analysis, flags=re.IGNORECASE)
        analysis = re.sub(r'Key Insights:?', 'KEY_INSIGHTS_MARKER', analysis, flags=re.IGNORECASE)
        analysis = analysis.replace('KEY_INSIGHTS_MARKER', '\n\n\n\n\n**Key Insights:**')
        
        # --- Add SQL section with title and spacing ---
        formatted_sql = _format_sql_readable(clean_sql)
        analysis += f"\n\n\n\n\n**SQL Query:**\n\n{'='*60}\n```sql\n{formatted_sql}\n```"
        
        return analysis
    except Exception as e:
        # If analysis fails, return formatted data with SQL
        return f"{df_str}\n\n{'='*60}\n**SQL Query:**\n{'='*60}\n```sql\n{_format_sql_readable(clean_sql)}\n```\n\n⚠️ Analysis error: {str(e)}"

@tool
def generate_and_show_sql(question: str) -> str:
    """Generate and validate SQL query without full analysis."""
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, api_key=GROQ_API_KEY)
    sql_prompt = SYSTEM_PROMPT + "\n\nSchema:\n" + schema_snippet + "\n\nQuestion: " + question + "\n\nSQL:"
    
    try:
        response = llm.invoke(sql_prompt)
        raw_sql = response.content.strip()
        clean_sql = _transform_sql_internal(raw_sql)
        
        # Test query
        df = _execute_sql(clean_sql)
        formatted_sql = _format_sql_readable(clean_sql)
        
        if df.empty:
            return f"✅ **SQL Query Generated:**\n```sql\n{formatted_sql}\n```\n\n⚠️ Query returns no data."
        
        return f"✅ **SQL Query Generated:**\n```sql\n{formatted_sql}\n```\n\n✓ Query validated - returns {len(df)} rows."
    except Exception as e:
        if 'clean_sql' in locals():
            return f"❌ Error: {str(e)[:300]}\n\n**Generated SQL:**\n```sql\n{_format_sql_readable(clean_sql)}\n```"
        return f"❌ Error generating SQL: {str(e)[:300]}"

# Setup tools and agent
tools = [chat_with_user, answer_ecommerce_question, generate_and_show_sql]
tool_executor = ToolExecutor(tools)
llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, api_key=GROQ_API_KEY)
llm_with_tools = llm.bind_tools(tools)

def call_model(state: AgentState):
    messages = state["messages"]
    
    if messages and isinstance(messages[-1], ToolMessage):
        response = llm.invoke(messages)
    else:
        response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

def call_tool(state: AgentState):
    last_message = state["messages"][-1]
    outputs = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
        tool_args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", {})
        tool_id = tool_call.get("id", "")
        
        tool = None
        for t in tools:
            if t.name == tool_name:
                tool = t
                break
        
        if tool:
            try:
                tool_result = tool.invoke(tool_args)
                outputs.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))
            except Exception as e:
                outputs.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_id))
        else:
            outputs.append(ToolMessage(content=f"Tool {tool_name} not found", tool_call_id=tool_id))
    
    return {"messages": outputs}

def should_continue(state: AgentState) -> Literal["continue", "end"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
    
    return "end"

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
workflow.add_edge("action", "agent")
agent_executor = workflow.compile()

def ask(question: str) -> str:
    """Ask question to agent - Uses LangGraph to route to appropriate tool."""
    try:
        # Simple keyword-based routing for better reliability
        question_lower = question.lower().strip()
        
        # Conversational triggers - expanded list
        conversational_keywords = [
            # Greetings
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
            # Gratitude
            'thanks', 'thank you', 'appreciate', 'grateful',
            # Personal questions about the bot
            'how are you', 'how do you do', 'what are you', 'who are you', 
            'what can you do', 'what do you do', 'your name', 'your creator',
            'what is your', 'tell me about yourself', 'introduce yourself',
            # Capability questions
            'what data', 'which data', 'what database', 'what kind of data',
            'what can you help', 'what can you analyze', 'your capabilities',
            'how do you work', 'how were you built', 'who created you',
            'who made you', 'what model', 'what llm',
            # Casual conversation
            'how is it going', 'whats up', "what's up", 'howdy',
            'nice to meet', 'pleasure to meet',
            # Reactions and acknowledgments
            'cool', 'awesome', 'nice', 'great', 'interesting', 'wow',
            'ok', 'okay', 'i see', 'got it', 'alright', 'sounds good',
            'perfect', 'excellent', 'amazing', 'wonderful',
            # Affirmations
            'yes', 'yeah', 'yep', 'sure', 'of course',
            'no', 'nah', 'nope',
            # Action prompts (starting conversation)
            'lets go', "let's go", 'lets start', "let's start",
            'lets begin', "let's begin", 'lets diggin', "let's diggin",
            'lets dive', "let's dive", 'ready', 'oke lets',
            'ok lets', 'okay lets',
            # Goodbye
            'bye', 'goodbye', 'see you', 'later', 'catch you'
        ]
        
        # Data-related keywords that should trigger data analysis mode
        data_keywords = [
            'order', 'customer', 'product', 'revenue', 'sales', 'user', 'inventory',
            'top', 'most', 'highest', 'lowest', 'count', 'show', 'find',
            'list', 'get', 'display', 'fetch', 'retrieve', 'total',
            'category', 'brand', 'price', 'profit', 'status', 'shipped',
            'how many', 'which are', 'what are the', 'give me',
            'ecommerce', 'thelook', 'purchase', 'buy', 'sold',
            'traffic', 'source', 'age', 'gender', 'country', 'state'
        ]
        
        # Check for very short responses (likely conversational)
        if len(question_lower.split()) <= 3 and '?' not in question:
            # Short phrases without questions are likely conversational
            result = chat_with_user.invoke({"message": question})
            return result
        
        # Check if it contains data keywords
        has_data_keywords = any(keyword in question_lower for keyword in data_keywords)
        
        # Check if it's a conversational question
        is_conversational = any(keyword in question_lower for keyword in conversational_keywords)
        
        # Route directly to appropriate tool
        # Data questions take priority if they have data keywords + question mark
        if has_data_keywords and '?' in question:
            result = answer_ecommerce_question.invoke({"question": question})
            return result
        elif is_conversational:
            result = chat_with_user.invoke({"message": question})
            return result
        elif has_data_keywords:
            result = answer_ecommerce_question.invoke({"question": question})
            return result
        else:
            # For ambiguous cases without clear indicators, use conversational
            # This prevents hallucination on casual responses
            result = chat_with_user.invoke({"message": question})
            return result
            
    except Exception as e:
        logger.error(f"Error in ask(): {str(e)}")
        return f"❌ Error: {str(e)}"

def check_connection():
    """Check Groq and BigQuery connections."""
    status = {"groq": False, "bigquery": False, "details": {}}
    
    # Check Groq
    try:
        llm_test = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
        llm_test.invoke("test")
        status["groq"] = True
        status["details"]["groq"] = f"Connected - Model: {GROQ_MODEL}"
    except Exception as e:
        status["details"]["groq"] = f"Failed: {str(e)[:100]}"
    
    # Check BigQuery
    try:
        query_job = bq_client.query("SELECT 1 as test")
        query_job.result()
        status["bigquery"] = True
        status["details"]["bigquery"] = f"Connected - Project: {PROJECT_ID}"
    except Exception as e:
        status["details"]["bigquery"] = f"Failed: {str(e)[:100]}"
    
    return status