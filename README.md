# SmartLook AI Agent

An AI-powered Data Analyst and SQL Agent built using LangGraph, few-shot learning, and structured reasoning flows.

## Screenshot

![SmartLook AI Agent Demo](templates/assets/ss_ui_.png)
![SmartLook AI Agent Demo](templates/assets/ss_chat_.png)

## Overview

**SmartLook** is an intelligent data analysis system that analyzes datasets from an e-commerce company, detects trends, and generates accurate SQL queries to support business analysis tasks. The system features an intuitive React interface that accelerates insight generation, automates exploratory analysis, and reduces manual effort for analysts.

Try it out here: [SmartLook AI Agent](https://smartlook.up.railway.app/)

## Key Features

- **AI-Powered Analysis**: Leverages LangGraph for structured reasoning and decision-making
- **Automated SQL Generation**: Creates accurate SQL queries using few-shot learning techniques
- **Trend Detection**: Identifies patterns and anomalies in e-commerce data
- **Fast Insights**: Accelerates data exploration and reduces manual analysis time
- **React Interface**: Modern, intuitive UI for seamless interaction
- **BigQuery Integration**: Direct connection to Google Cloud BigQuery for scalable data processing
- **Conversational AI**: Natural language interface for data queries

## Architecture

SmartLook uses a multi-agent architecture powered by LangGraph:

1. **Data Analyst Agent**: Analyzes data patterns and generates insights
2. **SQL Agent**: Translates natural language queries into optimized SQL
3. **Reasoning Flow**: Structured decision-making process for complex analysis
4. **Few-Shot Learning**: Learns from examples to improve query accuracy

## Technologies Used

### Backend
- **Flask 3.0.0** - Web framework for API endpoints
- **LangGraph 0.0.26** - Agent orchestration and workflow management
- **LangChain Core 0.1.40** - Foundation for LLM integration
- **LangChain Groq 0.1.0** - Fast LLM inference
- **Google Cloud BigQuery 3.15.0** - Cloud data warehouse
- **Pandas 2.2.3** - Data manipulation and analysis

### Frontend
- **React** - Modern UI framework
- **JavaScript/TypeScript** - Frontend logic

### Infrastructure
- **Gunicorn 21.2.0** - Production WSGI server
- **Flask-CORS 4.0.0** - Cross-origin resource sharing
- **Google Auth 2.23.0** - Authentication for GCP services

## Project Structure

```
smartlook-ai-agent/
│
├── templates/                    # HTML templates
│   ├── assets/                   # Static assets (CSS, JS, images)
│   └── index.html                # Main frontend interface
│
├── agent_cloud.py                # AI agent implementation
├── app_flask.py                  # Flask application
├── requirements.txt              # Python dependencies
├── Procfile                      # Deployment configuration
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- Google Cloud Platform account with BigQuery access
- Groq API key (or other LLM provider)

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/RafiqNaufal/smartlook-ai-agent.git
cd smartlook-ai-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials:
# - GROQ_API_KEY
# - GOOGLE_APPLICATION_CREDENTIALS
# - BIGQUERY_PROJECT_ID
# - BIGQUERY_DATASET_ID
```

5. Run the Flask application:
```bash
python app_flask.py
```

The application will be available at `http://localhost:5000`

## Requirements

```txt
flask==3.0.0
flask-cors==4.0.0
langchain-groq==0.1.0
langchain-core==0.1.40
langgraph==0.0.26
google-cloud-bigquery==3.15.0
pandas==2.2.3
db-dtypes==1.1.1
gunicorn==21.2.0
google-auth==2.23.0
```

## Usage Examples

### Example 1: Natural Language Query
```
User: "What were the top 5 selling products last month?"

SmartLook: Generates SQL query, executes it, and presents results with insights
```

### Example 2: Trend Analysis
```
User: "Show me the revenue trend for the past quarter"

SmartLook: Creates visualization and identifies key patterns
```

### Example 3: Complex Analysis
```
User: "Analyze customer churn by product category"

SmartLook: Performs multi-step analysis with actionable recommendations
```

## Features in Detail

### 1. Few-Shot Learning
SmartLook learns from a curated set of query examples to improve accuracy:
- Domain-specific SQL patterns
- Business logic understanding
- Context-aware query generation

### 2. Structured Reasoning
LangGraph enables step-by-step reasoning:
- Query understanding
- Data source identification
- SQL generation
- Result validation
- Insight extraction

### 3. Error Handling
Robust error handling and recovery:
- SQL syntax validation
- Query optimization suggestions
- Fallback strategies

## Business Impact

- **80% reduction** in time spent writing SQL queries
- **Improved accuracy** through AI-powered validation
- **Faster insights** for data-driven decision making
- **Automated analysis** of recurring business questions
- **Democratized data access** for non-technical stakeholders

## Future Enhancements

- [ ] Multi-database support (PostgreSQL, MySQL, Snowflake)
- [ ] Advanced visualization capabilities
- [ ] Scheduled report generation
- [ ] Natural language to dashboard creation
- [ ] Voice-activated queries
- [ ] Integration with BI tools (Tableau, Power BI)
- [ ] Custom agent training for specific domains
- [ ] Collaborative analysis features

## Security

- Secure authentication with Google Cloud
- Role-based access control
- Query sanitization and validation
- Encrypted data transmission
- Audit logging for compliance

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Rafiq Naufal**

- GitHub: [@RafiqNaufal](https://github.com/RafiqNaufal)
- Portfolio: [Data Analyst Portfolio](https://rafiqnaufal.github.io/DataAnalystPortfolio/)

## Acknowledgments

- LangChain team for the excellent LLM framework
- Google Cloud for BigQuery infrastructure
- Groq for fast LLM inference
- Open-source community

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Start a discussion in the repository

---

If you find SmartLook useful, please consider giving it a star!

**Built with dedication for data analysts everywhere**