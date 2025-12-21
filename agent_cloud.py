import logging
import re
import warnings
import os
import colorsys
import random
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langgraph.graph import StateGraph, END
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import Literal, TypedDict, Sequence
import json
import plotly.graph_objects as go
import pandas as pd

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

SYSTEM_PROMPT = f"""You are a BigQuery SQL expert for TheLook Ecommerce data.

RULES:
1. Use simple table names (users, orders, order_items, products, etc.)
2. Return ONLY SQL - no explanations
3. For stacked bar charts: ALWAYS use LIMIT 500 (not 100)
4. For simple queries: Use LIMIT {DEFAULT_LIMIT}
5. Use CROSS JOIN to generate ALL time periods

TIME PERIOD GENERATION:
- Monthly: CROSS JOIN with all 12 months
- Quarterly: CROSS JOIN with quarters 1,2,3,4
- Yearly: CROSS JOIN with array of years

COLUMN NAMING FOR STACKED CHARTS:
Always name columns: x_axis, stack_category, value

CRITICAL: PARSE USER REQUEST CORRECTLY
Read the user's question word by word to identify:
1. TIME dimension (monthly/quarterly/yearly) → becomes x_axis
2. BREAKDOWN dimension (the word after "by") → becomes stack_category

Examples of parsing:
- "monthly revenue by category" → time=month, breakdown=CATEGORY
- "yearly revenue by brand" → time=year, breakdown=BRAND
- "quarterly sales by country" → time=quarter, breakdown=COUNTRY
- "yearly revenue by category" → time=year, breakdown=CATEGORY

DO NOT use brand when user says category!
DO NOT use country when user says category!
DO NOT add extra dimensions!

TEMPLATE - Monthly by Category:
WITH all_months AS (
  SELECT DATE_TRUNC(DATE_ADD(DATE '2025-01-01', INTERVAL month MONTH), MONTH) AS month 
  FROM UNNEST(GENERATE_ARRAY(0, 11)) AS month
),
categories AS (
  SELECT DISTINCT category FROM products LIMIT 10
),
revenue_data AS (
  SELECT 
    DATE_TRUNC(DATE(oi.created_at), MONTH) as month,
    p.category,
    SUM(oi.sale_price) as revenue
  FROM order_items oi
  JOIN products p ON oi.product_id = p.id
  WHERE oi.status = 'Complete' AND EXTRACT(YEAR FROM oi.created_at) = 2025
  GROUP BY month, p.category
)
SELECT 
  FORMAT_DATE('%Y-%m-%d', am.month) as x_axis,
  c.category as stack_category,
  COALESCE(rd.revenue, 0) as value
FROM all_months am
CROSS JOIN categories c
LEFT JOIN revenue_data rd ON am.month = rd.month AND c.category = rd.category
ORDER BY am.month, value DESC
LIMIT 500

TEMPLATE - Yearly by Category:
WITH all_years AS (
  SELECT year FROM UNNEST([2022, 2023, 2024, 2025]) AS year
),
categories AS (
  SELECT DISTINCT category FROM products LIMIT 10
),
revenue_data AS (
  SELECT 
    EXTRACT(YEAR FROM oi.created_at) as year,
    p.category,
    SUM(oi.sale_price) as revenue
  FROM order_items oi
  JOIN products p ON oi.product_id = p.id
  WHERE oi.status = 'Complete' 
    AND EXTRACT(YEAR FROM oi.created_at) BETWEEN 2022 AND 2025
  GROUP BY year, p.category
)
SELECT 
  CAST(ay.year AS STRING) as x_axis,
  c.category as stack_category,
  COALESCE(rd.revenue, 0) as value
FROM all_years ay
CROSS JOIN categories c
LEFT JOIN revenue_data rd ON ay.year = rd.year AND c.category = rd.category
ORDER BY ay.year, value DESC
LIMIT 500

TEMPLATE - Yearly by Brand:
WITH all_years AS (
  SELECT year FROM UNNEST([2022, 2023, 2024, 2025]) AS year
),
brands AS (
  SELECT DISTINCT brand FROM products LIMIT 10
),
revenue_data AS (
  SELECT 
    EXTRACT(YEAR FROM oi.created_at) as year,
    p.brand,
    SUM(oi.sale_price) as revenue
  FROM order_items oi
  JOIN products p ON oi.product_id = p.id
  WHERE oi.status = 'Complete' 
    AND EXTRACT(YEAR FROM oi.created_at) BETWEEN 2022 AND 2025
  GROUP BY year, p.brand
)
SELECT 
  CAST(ay.year AS STRING) as x_axis,
  b.brand as stack_category,
  COALESCE(rd.revenue, 0) as value
FROM all_years ay
CROSS JOIN brands b
LEFT JOIN revenue_data rd ON ay.year = rd.year AND b.brand = rd.brand
ORDER BY ay.year, value DESC
LIMIT 500

TEMPLATE - Quarterly by Category:
WITH all_quarters AS (
  SELECT quarter FROM UNNEST([1, 2, 3, 4]) AS quarter
),
categories AS (
  SELECT DISTINCT category FROM products LIMIT 10
),
revenue_data AS (
  SELECT 
    EXTRACT(QUARTER FROM oi.created_at) as quarter,
    p.category,
    SUM(oi.sale_price) as revenue
  FROM order_items oi
  JOIN products p ON oi.product_id = p.id
  WHERE oi.status = 'Complete' AND EXTRACT(YEAR FROM oi.created_at) = 2025
  GROUP BY quarter, p.category
)
SELECT 
  CONCAT('Q', CAST(aq.quarter AS STRING), ' 2025') as x_axis,
  c.category as stack_category,
  COALESCE(rd.revenue, 0) as value
FROM all_quarters aq
CROSS JOIN categories c
LEFT JOIN revenue_data rd ON aq.quarter = rd.quarter AND c.category = rd.category
ORDER BY aq.quarter, value DESC
LIMIT 500

TEMPLATE - Quarterly by Brand:
WITH all_quarters AS (
  SELECT quarter FROM UNNEST([1, 2, 3, 4]) AS quarter
),
brands AS (
  SELECT DISTINCT brand FROM products LIMIT 10
),
revenue_data AS (
  SELECT 
    EXTRACT(QUARTER FROM oi.created_at) as quarter,
    p.brand,
    SUM(oi.sale_price) as revenue
  FROM order_items oi
  JOIN products p ON oi.product_id = p.id
  WHERE oi.status = 'Complete' AND EXTRACT(YEAR FROM oi.created_at) = 2025
  GROUP BY quarter, p.brand
)
SELECT 
  CONCAT('Q', CAST(aq.quarter AS STRING), ' 2025') as x_axis,
  b.brand as stack_category,
  COALESCE(rd.revenue, 0) as value
FROM all_quarters aq
CROSS JOIN brands b
LEFT JOIN revenue_data rd ON aq.quarter = rd.quarter AND b.brand = rd.brand
ORDER BY aq.quarter, value DESC
LIMIT 500

CRITICAL REMINDER:
If user says "by category" → use p.category
If user says "by brand" → use p.brand
If user says "by country" → use u.country
DO NOT mix them! Use ONLY what user requested!
"""

# Utility Functons
def _strip_code_fences(sql: str) -> str:
    """Remove markdown code fences and clean up the SQL string"""
    s = sql.strip()
    
    # Remove markdown code fences with optional sql language identifier
    s = re.sub(r'^```(?:sql|SQL)?\s*', '', s, flags=re.MULTILINE)
    s = re.sub(r'```\s*$', '', s, flags=re.MULTILINE)
    s = re.sub(r'```(?:sql|SQL)?', '', s)  # Remove any remaining code fences
    
    # Remove any remaining backticks
    s = s.replace('`', '')
    
    # Remove literal \n characters (not actual newlines) - CRITICAL FIX
    s = s.replace('\\n', ' ')
    
    # Remove the word "sql" if it appears at the start
    s = re.sub(r'^\s*sql\s+', '', s, flags=re.IGNORECASE)
    
    # Replace actual newlines with spaces
    s = s.replace('\n', ' ')
    s = s.replace('\r', ' ')
    
    # Normalize whitespace - replace multiple spaces with single space
    s = re.sub(r'\s+', ' ', s)
    
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
        # Check if this is a stacked bar query (has CROSS JOIN and stack_category)
        if "CROSS JOIN" in s.upper() and "stack_category" in s.lower():
            s = s.rstrip(";") + f" LIMIT 500;"
        else:
            s = s.rstrip(";") + f" LIMIT {DEFAULT_LIMIT};"
    
    return s.strip()

def _is_safe_sql_internal(sql: str) -> bool:
    """Check if SQL is safe (read-only)"""
    sql_lower = sql.lower()
    
    # Check for dangerous keywords at word boundaries
    # FIXED: Changed \\b to \b here as well
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
        # FIXED: Changed \\s and \\n to \s and \n
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

# Visualization Functions

# Custom color palette
CUSTOM_COLORS = ['#fe5208', '#36cdc3', '#5886e8', '#aae8f4', '#ffc600', '#ff9804']

def hex_to_rgb(hex_color):
    """Convert hex color to RGB"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """Convert RGB to hex color"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def generate_similar_color(base_color):
    """Generate a color similar to the base color by varying hue, saturation, and lightness"""
    # Convert hex to RGB
    r, g, b = hex_to_rgb(base_color)
    
    # Convert RGB to HSL (Hue, Saturation, Lightness)
    h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
    
    # Vary the hue slightly (±15 degrees)
    h = (h + random.uniform(-0.04, 0.04)) % 1.0
    
    # Vary saturation slightly (±10%)
    s = max(0.0, min(1.0, s + random.uniform(-0.1, 0.1)))
    
    # Vary lightness slightly (±10%)
    l = max(0.0, min(1.0, l + random.uniform(-0.1, 0.1)))
    
    # Convert back to RGB
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    
    return rgb_to_hex((r * 255, g * 255, b * 255))

def get_color_palette(num_colors):
    """Get a color palette with the specified number of colors.
    First 6 are from CUSTOM_COLORS, rest are similar variations"""
    
    if num_colors <= len(CUSTOM_COLORS):
        return CUSTOM_COLORS[:num_colors]
    
    palette = CUSTOM_COLORS.copy()
    
    # Generate additional colors based on the custom palette
    remaining = num_colors - len(CUSTOM_COLORS)
    
    for i in range(remaining):
        # Cycle through custom colors as base for similar colors
        base_color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        similar_color = generate_similar_color(base_color)
        palette.append(similar_color)
    
    return palette


def _detect_chart_type(question: str, df: pd.DataFrame) -> str:
    """Detect the appropriate chart type based on question and data"""
    question_lower = question.lower()
    
    # Check for explicit chart type mentions
    if any(word in question_lower for word in ['stacked bar', 'stacked chart', 'breakdown by', 'by quarter', 'by year and', 'share %', 'share by']):
        # Check if data has 3 columns (x, category, value) - typical stacked bar structure
        if len(df.columns) == 3:
            return 'stacked_bar'
    elif any(word in question_lower for word in ['bar chart', 'bar graph', 'bars']):
        return 'bar'
    elif any(word in question_lower for word in ['line chart', 'line graph', 'trend', 'over time', 'monthly', 'yearly']):
        return 'line'
    elif any(word in question_lower for word in ['pie chart', 'pie graph', 'percentage', 'proportion', 'distribution']):
        return 'pie'
    elif any(word in question_lower for word in ['scatter', 'correlation']):
        return 'scatter'
    
    # Auto-detect based on data structure
    if len(df.columns) == 3:
        # Check if it looks like stacked bar data (x_axis, category, value)
        third_col = df.columns[2]
        if any(word in str(third_col).lower() for word in ['value', 'revenue', 'sales', 'count', 'amount']):
            return 'stacked_bar'
    
    if len(df) <= 10 and len(df.columns) == 2:
        if any(word in question_lower for word in ['percentage', 'share', 'distribution']):
            return 'pie'
        return 'bar'
    elif 'date' in str(df.columns).lower() or 'month' in str(df.columns).lower() or 'year' in str(df.columns).lower():
        return 'line'
    else:
        return 'bar'

def _detect_x_axis_label(x_col: str, question: str, df: pd.DataFrame) -> str:
    """
    Dynamically detect X-axis label based on column name, question context, and actual data.
    """
    x_col_lower = x_col.lower()
    question_lower = question.lower()
    
    # Analyze the actual data in the column to understand its type
    sample_values = df[x_col].head(5).astype(str).tolist()
    sample_str = ' '.join(sample_values).lower()
    
    # Strategy 1: Check COLUMN NAME patterns first (highest priority)
    import re
    column_patterns = {
        r'month|monthly': 'Month',
        r'quarter': 'Quarter',
        r'year|yearly': 'Year',
        r'date|day|created_at': 'Date',
        r'week': 'Week',
    }
    
    for pattern, label in column_patterns.items():
        if re.search(pattern, x_col_lower):
            return label
    
    # Strategy 2: Analyze actual DATA patterns (second priority)
    try:
        # Check if data looks like dates (YYYY-MM-DD format)
        if re.search(r'\d{4}-\d{2}-\d{2}', sample_str):
            # Check if question asks for monthly/quarterly/yearly
            if any(word in question_lower for word in ['month', 'monthly']):
                return 'Month'
            elif any(word in question_lower for word in ['quarter', 'quarterly']):
                return 'Quarter'
            elif any(word in question_lower for word in ['year', 'yearly', 'annual']):
                return 'Year'
            else:
                return 'Month'  # Default for dates
        
        # Check if data looks like quarters (Q1, Q2, etc.)
        if any(f'q{i}' in sample_str for i in range(1, 5)):
            return 'Quarter'
        
        # Check if data looks like years (4-digit numbers)
        if all(val.isdigit() and len(val) == 4 for val in sample_values if val.isdigit()):
            return 'Year'
        
        # Check if data looks like country names
        if any(country in sample_str for country in ['china', 'united states', 'brasil', 'india', 'japan']):
            return 'Country'
        
        # Check if data looks like brands
        if any(brand in sample_str for brand in ['nike', 'adidas', 'calvin', 'diesel']):
            return 'Brand'
            
    except Exception:
        pass
    
    # Strategy 3: Check question keywords for TIME dimensions (only if column/data didn't match)
    if any(word in question_lower for word in ['month', 'monthly']):
        return 'Month'
    elif any(word in question_lower for word in ['quarter', 'quarterly', 'q1', 'q2', 'q3', 'q4']):
        return 'Quarter'
    elif any(word in question_lower for word in ['year', 'yearly', 'annual']):
        return 'Year'
    elif any(word in question_lower for word in ['day', 'daily', 'date']):
        return 'Date'
    elif any(word in question_lower for word in ['week', 'weekly']):
        return 'Week'
    
    # Strategy 4: Check question keywords for CATEGORICAL dimensions (lowest priority)
    elif any(word in question_lower for word in ['category', 'categories']):
        return 'Product Category'
    elif any(word in question_lower for word in ['brand', 'brands']):
        return 'Brand'
    elif any(word in question_lower for word in ['country', 'countries']):
        return 'Country'
    elif any(word in question_lower for word in ['state', 'states']):
        return 'State'
    elif any(word in question_lower for word in ['city', 'cities']):
        return 'City'
    elif any(word in question_lower for word in ['department', 'dept']):
        return 'Department'
    elif any(word in question_lower for word in ['product', 'products']):
        return 'Product Name'
    elif any(word in question_lower for word in ['customer', 'user', 'buyer']):
        return 'Customer'
    
    # Fallback: Format column name nicely
    cleaned = x_col.replace('_', ' ').replace('-', ' ')
    cleaned = re.sub(r'^(the|a|an)\s+', '', cleaned, flags=re.IGNORECASE)
    return cleaned.title()

def _detect_y_axis_label(y_col: str, question: str, df: pd.DataFrame) -> tuple:
    """
    Dynamically detect Y-axis label and format based on data type.
    
    Returns:
        tuple: (label, is_revenue, is_percentage, is_count)
    """
    y_col_lower = y_col.lower()
    question_lower = question.lower()
    
    # Check actual data to infer type
    sample_values = df[y_col].head(10)
    
    # Detect percentage (values between 0-1 or 0-100)
    is_percentage = False
    if 'percentage' in question_lower or 'share' in question_lower or '%' in question_lower or 'proportion' in question_lower:
        is_percentage = True
    elif 'percentage' in y_col_lower or 'share' in y_col_lower or 'percent' in y_col_lower:
        is_percentage = True
    elif sample_values.max() <= 1.0 and sample_values.min() >= 0:
        # Values between 0-1 likely percentage
        is_percentage = True
    
    # Detect revenue/money
    is_revenue = False
    if any(word in question_lower for word in ['revenue', 'sales', 'price', 'cost', 'profit', 'income']):
        is_revenue = True
    elif any(word in y_col_lower for word in ['revenue', 'sale_price', 'price', 'cost', 'profit']):
        is_revenue = True
    elif sample_values.max() > 1000 and not is_percentage:
        # Large numbers likely revenue
        is_revenue = True
    
    # Detect count
    is_count = False
    if not is_revenue and not is_percentage:
        if any(word in question_lower for word in ['number of', 'count', 'quantity', 'how many', 'total']):
            is_count = True
        elif any(word in y_col_lower for word in ['count', 'quantity', 'total', 'num_']):
            is_count = True
        elif all(val == int(val) for val in sample_values if not pd.isna(val)):
            # All integer values likely counts
            is_count = True
    
    # Generate label
    if is_percentage:
        if 'product' in question_lower:
            label = 'Product Share (%)'
        elif 'revenue' in question_lower:
            label = 'Revenue Share (%)'
        elif 'order' in question_lower:
            label = 'Order Share (%)'
        else:
            label = 'Share (%)'
    elif is_revenue:
        label = 'Revenue ($)'
    elif is_count:
        if 'product' in question_lower:
            label = 'Products Sold'
        elif 'order' in question_lower:
            label = 'Orders'
        elif 'customer' in question_lower or 'user' in question_lower:
            label = 'Customers'
        else:
            label = y_col.replace('_', ' ').title()
    else:
        label = y_col.replace('_', ' ').title()
    
    return label, is_revenue, is_percentage, is_count

def _create_stacked_bar_chart(df: pd.DataFrame, question: str) -> dict:
    """Create stacked bar chart visualization with custom colors"""
    
    if df.empty or len(df.columns) < 3:
        return {"error": "Stacked bar chart requires at least 3 columns (x_axis, category, value)"}
    
    columns = df.columns.tolist()
    x_col = columns[0]  
    stack_col = columns[1] 
    value_col = columns[2] 
    
    try:
        question_lower = question.lower()
        value_col_lower = value_col.lower()
        
        # Use smart X-axis detection
        x_axis_label = _detect_x_axis_label(x_col, question, df)

        # Smart metric detection using sample values
        sample_values = df[value_col].head(10)

        is_percentage = (
            'percentage' in question_lower or 'share' in question_lower or '%' in question_lower or
            'percentage' in value_col_lower or (sample_values.max() <= 1.0 and sample_values.min() >= 0)
        )

        is_revenue = (
            not is_percentage and 
            (any(word in question_lower for word in ['revenue', 'sales', 'price']) or
             any(word in value_col_lower for word in ['revenue', 'price', 'sale']))
        )

        is_count = (
            not is_percentage and not is_revenue and
            (any(word in question_lower for word in ['number of', 'count', 'sold', 'quantity']) or
             'count' in value_col_lower)
        )

        # Generate metric name
        if is_percentage:
            metric_name = 'Share'
        elif is_revenue:
            metric_name = 'Revenue'
        elif is_count:
            if 'product' in question_lower:
                metric_name = 'Products Sold'
            elif 'order' in question_lower:
                metric_name = 'Orders'
            else:
                metric_name = 'Count'
        else:
            metric_name = value_col.replace('_', ' ').title()

        y_axis_label = f'{metric_name} (%)' if is_percentage else f'{metric_name} ($)' if is_revenue else metric_name

        # Formatting Setup based on detection
        if is_percentage:
            y_tickformat = ',.1%'
            hover_format = '%{y:.1%}'
            text_template = '%{text:.1%}'
        elif is_revenue:
            y_tickformat = '$,.0f'
            hover_format = '$%{y:,.0f}'
            text_template = '$%{text:,.0f}'
        elif is_count:
            y_tickformat = ',.0f'
            hover_format = '%{y:,.0f}'
            text_template = '%{text:,.0f}'
        else:
            y_tickformat = ',.2f'
            hover_format = '%{y:,.2f}'
            text_template = '%{text:.2f}'

        all_x_values = df[x_col].unique()
        
        # Sort x-axis values properly
        try:
            # Try to sort as dates first
            if 'q' in str(all_x_values[0]).lower():
                # Quarters: Q1 2025, Q2 2025, etc.
                all_x_values = sorted(all_x_values, key=lambda x: (x.split()[-1], x.split()[0]))
            elif '-' in str(all_x_values[0]) and len(str(all_x_values[0])) == 10:
                # Dates: 2025-01-01, 2025-02-01, etc.
                all_x_values = sorted(all_x_values)
            elif all(str(x).isdigit() for x in all_x_values):
                # Years: 2022, 2023, 2024, 2025
                all_x_values = sorted(all_x_values)
            else:
                # Keep original order
                all_x_values = sorted(all_x_values)
        except:
            # If sorting fails, keep original order
            pass

        # Pivot and Filter Data
        pivot_df = df.pivot_table(
            index=x_col, 
            columns=stack_col, 
            values=value_col, 
            aggfunc='sum',
            fill_value=0
        )
        
        pivot_df = pivot_df.reindex(all_x_values, fill_value=0)
        
        # Select top categories by total value
        category_totals = pivot_df.sum().sort_values(ascending=False)
        top_categories = category_totals.head(10).index.tolist()
        pivot_df = pivot_df[top_categories]
        
        if is_percentage:
            # Normalize percentage charts to 100%
            pivot_df = pivot_df.apply(lambda x: x / x.sum() if x.sum() > 0 else x, axis=1)
            pivot_df = pivot_df.fillna(0) 
            
        # Chart Creation
        fig = go.Figure()
        
        colors = get_color_palette(len(pivot_df.columns))

        for idx, category in enumerate(pivot_df.columns):
            color = colors[idx]
            fig.add_trace(go.Bar(
                name=str(category),
                x=pivot_df.index,
                y=pivot_df[category],
                marker_color=color,
                hovertemplate=f'<b>{category}</b><br>%{{x}}<br>Value: {hover_format}<extra></extra>',
                text=pivot_df[category],
                texttemplate=text_template,
                textposition='inside',
                textfont=dict(size=10, color='white')
            ))
        
        # Layout
        fig.update_layout(
            title=question,
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            barmode='stack',
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=1
            ),
            height=500,
            margin=dict(r=150)
        )
        
        # Apply Y-axis formatting based on detection
        fig.update_yaxes(tickformat=y_tickformat)
        
        return {
            "chart_json": fig.to_json(), 
            "chart_type": "stacked_bar", 
            "data_points": len(df),
            "categories": len(top_categories), 
            "x_axis": x_col, 
            "stack_by": stack_col
        }
        
    except Exception as e:
        logger.error(f"Stacked bar chart error: {str(e)}")
        return {"error": f"Failed to create stacked bar chart: {str(e)}"}


def _create_visualization(df: pd.DataFrame, question: str, chart_type: str = None) -> dict:
    """Create interactive Plotly visualization with custom colors"""
    
    if df.empty:
        return {"error": "No data to visualize"}
    
    if not chart_type:
        chart_type = _detect_chart_type(question, df)
    
    # Handle stacked bar chart
    if chart_type == 'stacked_bar':
        return _create_stacked_bar_chart(df, question)
    
    columns = df.columns.tolist()
    
    if len(columns) < 2:
        return {"error": "Need at least 2 columns for visualization"}
    
    x_col = columns[0]
    y_col = columns[1]
    
    df_viz = df.head(20)
    
    try:
        # --- Use smart detections ---
        x_axis_label = _detect_x_axis_label(x_col, question, df_viz)
        y_axis_label, is_revenue, is_percentage, is_count = _detect_y_axis_label(y_col, question, df_viz)
        
        # --- Formatting Setup based on detection ---
        if is_percentage:
            y_tickformat = ',.1%'
            hover_format = '%{y:.1%}'
            text_template = '%{text:.1%}'
        elif is_revenue:
            y_tickformat = '$,.0f'
            hover_format = '$%{y:,.0f}'
            text_template = '$%{text:,.0f}'
        elif is_count:
            y_tickformat = ',.0f'
            hover_format = '%{y:,.0f}'
            text_template = '%{text:,.0f}'
        else:
            y_tickformat = ',.2f'
            hover_format = '%{y:,.2f}'
            text_template = '%{text:.2f}'
        
        # --- Chart Creation ---
        if chart_type == 'bar':
            # Generate colors for bar chart
            num_bars = len(df_viz)
            bar_colors = get_color_palette(num_bars)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df_viz[x_col],
                    y=df_viz[y_col],
                    marker=dict(
                        color=bar_colors,
                        showscale=False
                    ),
                    text=df_viz[y_col],
                    texttemplate=text_template,
                    textposition='outside',
                    hovertemplate=f'<b>%{{x}}</b><br>{y_axis_label}: {hover_format}<extra></extra>'
                )
            ])
            fig.update_layout(
                title=question,
                xaxis_title=x_axis_label,
                yaxis_title=y_axis_label,
                template='plotly_white',
                hovermode='x',
                showlegend=False
            )
            fig.update_yaxes(tickformat=y_tickformat)
            
        elif chart_type == 'line':
            fig = go.Figure(data=[
                go.Scatter(
                    x=df_viz[x_col],
                    y=df_viz[y_col],
                    mode='lines+markers',
                    line=dict(color=CUSTOM_COLORS[0], width=3),  # #fe5208 (orange-red)
                    marker=dict(size=8, color=CUSTOM_COLORS[0]),  # #fe5208 (orange-red)
                    hovertemplate=f'<b>%{{x}}</b><br>{y_axis_label}: {hover_format}<extra></extra>'
                )
            ])
            fig.update_layout(
                title=question,
                xaxis_title=x_axis_label,
                yaxis_title=y_axis_label,
                template='plotly_white',
                hovermode='x'
            )
            fig.update_yaxes(tickformat=y_tickformat)
            
        elif chart_type == 'pie':
            # For pie charts, use different formatting for labels
            if is_percentage:
                texttemplate = '%{label}<br>%{percent}'
            elif is_revenue:
                texttemplate = '%{label}<br>$%{value:,.0f}'
            else:
                texttemplate = '%{label}<br>%{value:,.0f}'
            
            pie_colors = get_color_palette(len(df_viz))

            fig = go.Figure(data=[
                go.Pie(
                    labels=df_viz[x_col],
                    values=df_viz[y_col],
                    hole=0.3,
                    marker=dict(colors=pie_colors),
                    texttemplate=texttemplate,
                    hovertemplate=f'<b>%{{label}}</b><br>{y_axis_label}: {hover_format}<br>Percentage: %{{percent}}<extra></extra>'
                )
            ])
            fig.update_layout(
                title=question,
                template='plotly_white'
            )
            
        elif chart_type == 'scatter':
            fig = go.Figure(data=[
                go.Scatter(
                    x=df_viz[x_col],
                    y=df_viz[y_col],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=df_viz[y_col],
                        colorscale=[
                            [0, CUSTOM_COLORS[0]],
                            [0.2, CUSTOM_COLORS[2]],
                            [0.4, CUSTOM_COLORS[1]],
                            [0.6, CUSTOM_COLORS[3]],
                            [0.8, CUSTOM_COLORS[4]],
                            [1.0, CUSTOM_COLORS[5]]
                        ],
                        showscale=True,
                        colorbar=dict(
                            title=y_axis_label,
                            tickformat=y_tickformat
                        )
                    ),
                    hovertemplate=f'<b>%{{x}}</b><br>{y_axis_label}: {hover_format}<extra></extra>'
                )
            ])
            fig.update_layout(
                title=question,
                xaxis_title=x_axis_label,
                yaxis_title=y_axis_label,
                template='plotly_white'
            )
            fig.update_yaxes(tickformat=y_tickformat)
        
        return {
            "chart_json": fig.to_json(),
            "chart_type": chart_type,
            "data_points": len(df_viz)
        }
        
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return {"error": f"Failed to create visualization: {str(e)}"}

def _create_cohort_heatmap(df: pd.DataFrame, question: str) -> dict:
    """Create cohort retention heatmap visualization"""
    
    try:
        # Pivot data for heatmap
        pivot_data = df.pivot(
            index='cohort_month', 
            columns='month_number', 
            values='percentage'
        )
        
        # Fill NaN values with 0 for missing data
        pivot_data = pivot_data.fillna(0)
        
        # Create custom hovertext with all details BEFORE converting dates
        hover_text = []
        for cohort_idx, cohort in enumerate(pivot_data.index):
            hover_row = []
            for col_idx, month_num in enumerate(pivot_data.columns):
                # Use iloc for safe positional indexing
                percentage = pivot_data.iloc[cohort_idx, col_idx]
                
                # Get additional details from original df
                cohort_data = df[
                    (df['cohort_month'] == cohort) & 
                    (df['month_number'] == month_num)
                ]
                
                if not cohort_data.empty and not pd.isna(percentage):
                    total_users = int(cohort_data['total_users'].values[0])
                    cohort_size = int(cohort_data['cohort_size'].values[0])
                    # Format cohort date nicely for hover
                    cohort_display = pd.to_datetime(cohort).strftime('%b %Y')
                    hover_row.append(
                        f"Cohort: {cohort_display}<br>"
                        f"Month: {month_num}<br>"
                        f"Users: {total_users}/{cohort_size}<br>"
                        f"Retention: {percentage:.1f}%"
                    )
                else:
                    cohort_display = pd.to_datetime(cohort).strftime('%b %Y')
                    hover_row.append(f"Cohort: {cohort_display}<br>Month: {month_num}<br>No data")
            hover_text.append(hover_row)
        
        # NOW convert cohort_month to readable string format for display
        pivot_data.index = pd.to_datetime(pivot_data.index).strftime('%b %Y')
        
        # Create heatmap with intuitive color scale (Red = bad, Yellow = medium, Green = good)
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=[str(col) for col in pivot_data.columns],
            y=pivot_data.index.tolist(),
            colorscale=[
                [0, '#dc2626'],      # Bright red for 0% (worst)
                [0.01, '#ef4444'],   # Red
                [0.02, '#f87171'],   # Light red
                [0.05, '#fb923c'],   # Orange
                [0.1, '#fb923c'],    # Orange
                [0.2, '#fbbf24'],    # Amber
                [0.3, '#fde047'],    # Light yellow
                [0.5, '#fef08a'],    # Bright yellow for 50% (medium)
                [0.6, '#d9f99d'],    # Yellow-green
                [0.7, '#a3e635'],    # Lime
                [0.8, '#4ade80'],    # Light green
                [0.9, '#22c55e'],    # Green
                [1.0, '#16a34a']     # Dark green for 100% (best)
            ],
            text=pivot_data.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10, "color": "white", "family": "Arial Black"},
            hovertext=hover_text,
            hoverinfo='text',
            colorbar=dict(
                title="Retention %",
                ticksuffix="%",
                thickness=20,
                len=0.7,
                x=1.02,
                tickfont=dict(size=11, color="#1e293b")
            ),
            zmin=0,
            zmax=100,
            zauto=False
        ))
        
        fig.update_layout(
            title='',
            xaxis_title='Months Since First Purchase',
            yaxis_title='Cohort Month',
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1,
                side='bottom',
                tickfont=dict(size=12, color="#1e293b", family="Arial"),
                gridcolor='#e2e8f0'
            ),
            yaxis=dict(
                autorange='reversed',
                tickfont=dict(size=12, color="#1e293b", family="Arial"),
                gridcolor='#e2e8f0'
            ),
            template='plotly_white',
            height=max(500, len(pivot_data) * 45),
            margin=dict(l=130, r=100, t=40, b=80),
            font=dict(family="Arial, sans-serif"),
            plot_bgcolor='#f8fafc',
            paper_bgcolor='white'
        )
        
        return {
            "chart_json": fig.to_json(),
            "chart_type": "cohort_heatmap",
            "data_points": len(df),
            "cohorts_count": len(pivot_data),
            "max_months": int(pivot_data.columns.max())
        }
        
    except Exception as e:
        logger.error(f"Cohort heatmap error: {str(e)}")
        return {"error": f"Failed to create cohort heatmap: {str(e)}"}

# Tool Implementations

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
            formatted_sql = _format_sql_readable(clean_sql)
            return json.dumps({
                "is_data_response": True,
                "response_text": f"No results found.\n\n**SQL Query:**\n```sql\n{formatted_sql}\n```",
                "sql": formatted_sql,
                "source_data": None,
                "row_count": 0
            })
        
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
{df.head(500).to_string(index=False)}

Format your response EXACTLY like this:

**[Title based on question]**

1. [Item 1] - [value]
2. [Item 2] - [value]
3. [Item 3] - [value]
...

**Key Insights:**

- [First key insight - wrap important keywords in **bold**]
- [Second insight - wrap important numbers/terms in **bold**]
- [Third insight - wrap key findings in **bold**]

**Actionable Recommendations:**

1. **[Action Title]** — [Detailed recommendation with specific steps]
2. **[Action Title]** — [Detailed recommendation with specific steps]
3. **[Action Title]** — [Detailed recommendation with specific steps]

CRITICAL: 
- DO NOT use \\n in your response - use actual line breaks
- Wrap important keywords, numbers, brands, categories, and key terms in **bold** (e.g., **Denim**, **17 units**, **men's apparel**).
- Make recommendations specific, actionable, and data-driven based on the results.
- Keep it clean, structured, and professional. Use bullet points (•) for insights."""

        try:
            analysis_response = llm.invoke(analysis_prompt)
            analysis = analysis_response.content.strip()
            
            # CRITICAL FIX: Clean up any \n in the LLM response
            analysis = analysis.replace('\\n', '\n')
            analysis = re.sub(r'\n\s*\n\s*\n+', '\n\n', analysis)
            
        except Exception as e:
            # Fallback to just showing data
            analysis = df_str
        
        formatted_sql = _format_sql_readable(clean_sql)
        response_text = f"{analysis}\n\n**SQL Query:**\n```sql\n{formatted_sql}\n```"
        
        # Return JSON with source data for CSV download
        return json.dumps({
            "is_data_response": True,
            "response_text": response_text,
            "sql": formatted_sql,
            "source_data": df.astype(str).to_dict(orient='records'),
            "row_count": row_count,
            "question": question
        })
        
    except Exception as e:
        error_msg = str(e)
        if 'clean_sql' in locals():
            formatted_sql = _format_sql_readable(clean_sql)
            return json.dumps({
                "is_data_response": True,
                "response_text": f"❌ Error: {error_msg[:500]}\n\n**SQL Query:**\n```sql\n{formatted_sql}\n```",
                "sql": formatted_sql,
                "source_data": None,
                "row_count": 0
            })
        return json.dumps({
            "is_data_response": True,
            "response_text": f"❌ Error: {error_msg[:500]}",
            "sql": None,
            "source_data": None,
            "row_count": 0
        })

@tool
def create_visualization(question: str, chart_type: str = None) -> str:
    """Create interactive visualizations (bar, line, pie, scatter, stacked_bar charts) from TheLook Ecommerce data.
    
    Use chart_type='stacked_bar' for breakdowns like:
    - Revenue by Quarter and Country
    - Sales by Category and Brand  
    - Revenue Share % by Year and Product Category
    """
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, api_key=GROQ_API_KEY)
    sql_prompt = SYSTEM_PROMPT + "\\n\\nSchema:\\n" + schema_snippet + "\\n\\nQuestion: " + question + "\\n\\nSQL:"
    
    try:
        # Generate SQL
        response = llm.invoke(sql_prompt)
        raw_sql = response.content.strip()
        clean_sql = _transform_sql_internal(raw_sql)
        
        # Execute query
        df = _execute_sql(clean_sql)
        
        if df.empty:
            return json.dumps({
                "error": "No data found for visualization",
                "sql": _format_sql_readable(clean_sql)
            })
        
        # Create visualization
        viz_result = _create_visualization(df, question, chart_type)
        
        if "error" in viz_result:
            return json.dumps({
                "error": viz_result["error"],
                "sql": _format_sql_readable(clean_sql)
            })
        
        # Add metadata
        viz_result["sql"] = _format_sql_readable(clean_sql)
        viz_result["question"] = question
        viz_result["row_count"] = len(df)
        viz_result["source_data"] = df.astype(str).to_dict(orient='records')
        
        return json.dumps(viz_result)
        
    except Exception as e:
        error_response = {
            "error": f"Failed to create visualization: {str(e)[:300]}"
        }
        if 'clean_sql' in locals():
            error_response["sql"] = _format_sql_readable(clean_sql)
        return json.dumps(error_response)
    
@tool
def create_cohort_analysis(question: str, year: int = None, category: str = None, product: str = None, brand: str = None, country: str = None) -> str:
    """Create cohort retention analysis and heatmap visualization. 
    Use this for cohort analysis, retention analysis, or user lifecycle questions.
    
    Args:
        question: The analysis question
        year: Optional year filter (e.g., 2023, 2024, 2025)
        category: Optional product category filter (e.g., "Intimates", "Jeans", "Outerwear & Coats")
        product: Optional product name filter
        brand: Optional brand filter (e.g., "Calvin Klein")
        country: Optional country name filter (e.g, "China", "United States", "Brazil")
    """

    # Build WHERE clause filters with proper SQL escaping
    filter_lines = []
    
    if year:
        filter_lines.append(f"        AND EXTRACT(YEAR FROM o.created_at) = {year}")
    
    if category:
        # Clean and escape the category name
        escaped_category = str(category).strip().replace("'", "''")
        filter_lines.append(f"        AND p.category = '{escaped_category}'")
        if not PRODUCTION_MODE:
            print(f"[DEBUG] Adding category filter: {escaped_category}")
    
    if product:
        escaped_product = str(product).strip().replace("'", "''")
        filter_lines.append(f"        AND p.name LIKE '%{escaped_product}%'")

    if country:
        escaped_country = str(country).strip().replace("'", "''")
        filter_lines.append(f"        AND u.country = '{escaped_country}'")
    
    if brand:
        escaped_brand = str(brand).strip().replace("'", "''")
        filter_lines.append(f"        AND p.brand = '{escaped_brand}'")
    
    filter_clause = "\n".join(filter_lines) if filter_lines else ""
    
    # Enhanced cohort query with proper table joins
    cohort_query = f"""
    WITH first_purchase AS (
      SELECT
        o.user_id,
        DATE_TRUNC(DATE(MIN(o.created_at)), MONTH) AS cohort_month
      FROM `bigquery-public-data.thelook_ecommerce.orders` o
      JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi
        ON o.order_id = oi.order_id
      JOIN `bigquery-public-data.thelook_ecommerce.products` p
        ON oi.product_id = p.id
      JOIN `bigquery-public-data.thelook_ecommerce.users` u
        ON o.user_id = u.id
      WHERE o.status = 'Complete'{filter_clause}
      GROUP BY o.user_id
    ),
    
    monthly_activity AS (
      SELECT
        o.user_id,
        DATE_TRUNC(DATE(o.created_at), MONTH) AS activity_month
      FROM `bigquery-public-data.thelook_ecommerce.orders` o
      JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi
        ON o.order_id = oi.order_id
      JOIN `bigquery-public-data.thelook_ecommerce.products` p
        ON oi.product_id = p.id
      JOIN `bigquery-public-data.thelook_ecommerce.users` u
        ON o.user_id = u.id
      WHERE o.status = 'Complete'{filter_clause}
      GROUP BY o.user_id, activity_month
    ),
    
    cohorts AS (
      SELECT
        fp.cohort_month,
        ma.activity_month,
        DATE_DIFF(ma.activity_month, fp.cohort_month, MONTH) + 1 AS month_number,
        COUNT(DISTINCT ma.user_id) AS total_users
      FROM first_purchase fp
      JOIN monthly_activity ma
        ON fp.user_id = ma.user_id
      GROUP BY fp.cohort_month, ma.activity_month, month_number
    ),
    
    cohort_sizes AS (
      SELECT
        cohort_month,
        COUNT(DISTINCT user_id) AS cohort_size
      FROM first_purchase
      GROUP BY cohort_month
    )
    
    SELECT
      c.cohort_month,
      cs.cohort_size,
      c.month_number,
      c.total_users,
      ROUND(c.total_users / cs.cohort_size * 100, 2) AS percentage
    FROM cohorts c
    JOIN cohort_sizes cs USING (cohort_month)
    ORDER BY c.cohort_month, c.month_number
    LIMIT 1000
    """
    
    try:
        # Execute query
        df = _execute_sql(cohort_query)
        
        if df.empty:
            # Build filter description for error message
            filter_desc = []
            if year:
                filter_desc.append(f"year {year}")
            if category:
                filter_desc.append(f"category '{category}'")
            if product:
                filter_desc.append(f"product '{product}'")
            if brand:
                filter_desc.append(f"brand '{brand}'")
            if country:
                filter_desc.append(f"country '{country}'")
            
            filter_text = " with filters: " + ", ".join(filter_desc) if filter_desc else ""
            
            return json.dumps({
                "error": f"No cohort data found{filter_text}",
                "sql": cohort_query,
                "suggestion": "Try adjusting your filters or check if data exists for the specified criteria"
            })
        
        # Create cohort heatmap
        viz_result = _create_cohort_heatmap(df, question)
        
        if "error" in viz_result:
            return json.dumps({
                "error": viz_result["error"],
                "sql": cohort_query
            })
        
        # Generate insights
        llm = ChatGroq(model=GROQ_MODEL, temperature=0.3, api_key=GROQ_API_KEY)
        
        # Get summary statistics
        avg_retention_month_1 = df[df['month_number'] == 1]['percentage'].mean()
        avg_retention_month_3 = df[df['month_number'] == 3]['percentage'].mean() if 3 in df['month_number'].values else None
        avg_retention_month_6 = df[df['month_number'] == 6]['percentage'].mean() if 6 in df['month_number'].values else None
        best_cohort = df[df['month_number'] == 1].nlargest(1, 'percentage')
        
        # Build filter description
        filter_desc = []
        if year:
            filter_desc.append(f"year {year}")
        if category:
            filter_desc.append(f"category '{category}'")
        if product:
            filter_desc.append(f"product '{product}'")
        if brand:
            filter_desc.append(f"brand '{brand}'")
        if country:
            filter_desc.append(f"country '{country}'")
        
        filter_text = " for " + ", ".join(filter_desc) if filter_desc else ""
        
        summary_stats = f"""
Cohort Summary{filter_text}:
- Total Cohorts: {df['cohort_month'].nunique()}
- Avg Month 1 Retention: {avg_retention_month_1:.1f}%
"""
        if avg_retention_month_3:
            summary_stats += f"- Avg Month 3 Retention: {avg_retention_month_3:.1f}%\n"
        if avg_retention_month_6:
            summary_stats += f"- Avg Month 6 Retention: {avg_retention_month_6:.1f}%\n"
        
        if not best_cohort.empty:
            summary_stats += f"- Best Performing Cohort: {best_cohort['cohort_month'].values[0]} with {best_cohort['percentage'].values[0]:.1f}% retention\n"
        
        analysis_prompt = f"""Analyze this cohort retention data and provide key insights.

Question: {question}

{summary_stats}

Sample Data (first 10 rows):
{df.head(10).to_string(index=False)}

Format your response EXACTLY like this:

**Cohort Retention Analysis{filter_text}**

**Key Insights:**

- [First key insight - wrap important keywords in **bold**]
- [Second insight - wrap important numbers/terms in **bold**]
- [Third insight - wrap key findings in **bold**]

**Actionable Recommendations:**

1. **[Action Title]** — [Detailed recommendation with specific steps]
2. **[Action Title]** — [Detailed recommendation with specific steps]
3. **[Action Title]** — [Detailed recommendation with specific steps]

CRITICAL: 
- DO NOT use backslash-n in your response
- Wrap important keywords, numbers, percentages, cohorts, and key terms in **bold** (e.g., **Month 1**, **1.6%**, **January 2025 cohort**).
- Make recommendations specific, actionable, and data-driven based on the cohort retention results.
- Keep it clean, structured, and professional. Use bullet points (•) for insights and numbered list for recommendations.
- Focus on retention improvement strategies with measurable outcomes.
- Reference specific months or cohorts from the data."""

        try:
            analysis_response = llm.invoke(analysis_prompt)
            response_text = analysis_response.content.strip()
            
            # Clean up any literal \n in the response
            response_text = response_text.replace('\\n', '\n')
            
        except Exception as e:
            response_text = summary_stats
        
        # Add metadata
        viz_result["sql"] = cohort_query
        viz_result["question"] = question
        viz_result["response_text"] = response_text
        viz_result["row_count"] = len(df)
        viz_result["filters_applied"] = {
            "year": year,
            "category": category,
            "product": product,
            "brand": brand,
            "country": country
        }

        viz_result["source_data"] = df.astype(str).to_dict(orient='records')
        
        return json.dumps(viz_result)
        
    except Exception as e:
        error_response = {
            "error": f"Failed to create cohort analysis: {str(e)[:500]}"
        }
        if 'cohort_query' in locals():
            error_response["sql"] = cohort_query
        return json.dumps(error_response)

@tool
def generate_and_show_sql(question: str) -> str:
    """Generate and validate SQL query without executing full analysis."""
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1, api_key=GROQ_API_KEY)
    
    # Clean the question to remove "generate/show sql" phrases
    cleaned_question = question.lower()
    
    # Remove common SQL request phrases
    patterns_to_remove = [
        r'^generate\s+(sql\s+)?(query\s+)?(to\s+)?(find\s+)?',
        r'^show\s+(me\s+)?(the\s+)?(sql\s+)?(query\s+)?(to\s+)?(find\s+)?',
        r'^create\s+(sql\s+)?(query\s+)?(to\s+)?(find\s+)?',
        r'^write\s+(sql\s+)?(query\s+)?(to\s+)?(find\s+)?',
        r'^give\s+(me\s+)?(sql\s+)?(query\s+)?(to\s+)?(find\s+)?',
    ]
    
    for pattern in patterns_to_remove:
        cleaned_question = re.sub(pattern, '', cleaned_question, flags=re.IGNORECASE)
    
    # Capitalize first letter for display
    cleaned_question = cleaned_question.strip()
    if cleaned_question and cleaned_question[0].islower():
        cleaned_question = cleaned_question[0].upper() + cleaned_question[1:]
    
    # Use original question for SQL generation, cleaned for display
    sql_prompt = SYSTEM_PROMPT + "\n\nSchema:\n" + schema_snippet + "\n\nQuestion: " + question + "\n\nSQL:"
    
    try:
        response = llm.invoke(sql_prompt)
        raw_sql = response.content.strip()
        clean_sql = _transform_sql_internal(raw_sql)
        df = _execute_sql(clean_sql)
        formatted_sql = _format_sql_readable(clean_sql)
        
        # Use cleaned question in the intro
        intro = f"Here's the SQL for \"{cleaned_question}\"\n\n"
        
        if df.empty:
            return f"{intro}**SQL Query:**\n```sql\n{formatted_sql}\n```\n\n⚠️ Query returns no data."
        
        return f"{intro}**SQL Query:**\n```sql\n{formatted_sql}\n```\n\n**Validated** - returns {len(df)} rows."
    except Exception as e:
        return f"❌ Error: {str(e)[:300]}"

# Agent Workflow
tools = [chat_with_user, answer_ecommerce_question, create_visualization, create_cohort_analysis, generate_and_show_sql]
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
    
    context = "Recent conversation:\\n"
    for msg in _conversation_history[-6:]:  # Last 3 exchanges
        context += f"{msg['role']}: {msg['content'][:150]}...\\n"
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

3. create_visualization - For creating charts and graphs (bar, line, pie, scatter, stacked_bar)
   Examples: "Create a bar chart", "Show pie chart", "Line graph of sales"
   IMPORTANT: Use this when user explicitly asks for chart, graph, plot, or visualization!
   
   CRITICAL - Pass the user's question EXACTLY as they wrote it:
   - "monthly revenue by category" → pass as-is
   - "yearly revenue by brand" → pass as-is
   - DO NOT add words like "country" or "brand" if user didn't say them
   
   For STACKED BAR CHARTS specifically, use chart_type='stacked_bar' when:
   - User EXPLICITLY asks for "stacked bar chart" or "stacked chart"
   - User asks for breakdown by a dimension (e.g., "by category", "by brand", "by country")
   - Keywords: "breakdown by", "revenue share %", "comparison", "stacked"
   
   IMPORTANT: When user says "stacked bar chart", automatically add appropriate breakdown dimension:
   - For time-based queries (quarter/year/month) → add "by country" or "by category"
   - For category queries → add "by brand" or "by department"
   - For product queries → add "by category" or "by brand"

   IMPORTANT: Handle STACKED BAR CHARTS logic carefully:
   1. If user provides TWO dimensions (e.g., "Revenue by Quarter AND Category"), use them EXACTLY. DO NOT add more.
      - "Quarterly revenue by category" → args: {"question": "Quarterly revenue by category", "chart_type": "stacked_bar"}
   
   2. Only add a breakdown dimension if the user provided ONLY a time dimension:
      - "Revenue by Quarter" (1 dim) → Add breakdown: "Revenue by Quarter by Category" or "by Country"
      - "Sales by Year" (1 dim) → Add breakdown: "Sales by Year by Brand"
      
   DO NOT add "by country" if the user already asked for "by category"!
   
   Examples: 
     * "create stacked bar chart of 2024 revenue by quarter" → {"tool": "create_visualization", "args": {"question": "2024 revenue by quarter by country", "chart_type": "stacked_bar"}}
     * "stacked bar of sales by category" → {"tool": "create_visualization", "args": {"question": "sales by category by brand", "chart_type": "stacked_bar"}}
     * "2024 revenue by quarter and country" → {"tool": "create_visualization", "args": {"question": "2024 revenue by quarter and country", "chart_type": "stacked_bar"}}
   
   For regular bar/line charts (single dimension):
   - User asks for simple trends without "stacked" keyword
   - Examples:
     * "2024 revenue by quarter" → {"tool": "create_visualization", "args": {"question": "2024 revenue by quarter", "chart_type": "bar"}}
     * "show quarterly revenue" → {"tool": "create_visualization", "args": {"question": "quarterly revenue 2024", "chart_type": "line"}}

4. create_cohort_analysis - For cohort retention analysis and heatmaps
   Examples: "Show cohort analysis", "Retention heatmap", "User retention by cohort"
   Keywords: cohort, retention, lifecycle, repeat customers
   
   CRITICAL - Extract filters from the user's question:
   - year: Look for year numbers like "2024", "in 2025", "for 2023"
   - category: Look for product categories like "Intimates", "Jeans", "Outerwear & Coats", "Accessories"
   - product: Look for specific product names
   - brand: Look for brand names like "Calvin Klein", "Nike", "Adidas"
   
   ALWAYS include extracted parameters in args!
   
   Examples:
   - "cohort retention for Intimates category in 2024" → {"tool": "create_cohort_analysis", "args": {"question": "cohort retention for Intimates", "year": 2024, "category": "Intimates"}}
   - "show retention for Jeans in 2025" → {"tool": "create_cohort_analysis", "args": {"question": "retention for Jeans", "year": 2025, "category": "Jeans"}}
   - "cohort analysis for Calvin Klein brand" → {"tool": "create_cohort_analysis", "args": {"question": "cohort for Calvin Klein", "brand": "Calvin Klein"}}
   - "retention heatmap for Accessories" → {"tool": "create_cohort_analysis", "args": {"question": "retention heatmap", "category": "Accessories"}}

5. generate_and_show_sql - For explicitly requesting SQL queries
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

User: "Create a bar chart of revenue by category"
Response: {"tool": "create_visualization", "args": {"question": "revenue by category", "chart_type": "bar"}}

User: "Show 2024 revenue by quarter and country"
Response: {"tool": "create_visualization", "args": {"question": "2024 revenue by quarter and country", "chart_type": "stacked_bar"}}

User: "Revenue share % by year and product category"
Response: {"tool": "create_visualization", "args": {"question": "revenue share by year and product category", "chart_type": "stacked_bar"}}

User: "Show cohort retention analysis for 2023"
Response: {"tool": "create_cohort_analysis", "args": {"question": "cohort retention analysis", "year": 2023}}

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
        response_text = re.sub(r'```json\\s*|\\s*```', '', response_text).strip()
        
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

# Public API
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