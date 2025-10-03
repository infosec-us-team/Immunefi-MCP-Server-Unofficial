# Immunefi Bug Bounty MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.ai/) server that allows AI assistants and other MCP-compatible clients to search, filter, and retrieve data from the Immunefi bug bounty platform.

## 🚀 Features

- **🔍 Program Search**: Search and filter bug bounty programs by name, ID, slug, or tags
- **💰 Bounty Filters**: Filter programs by bounty range
- **🐙 GitHub Discovery**: Extract GitHub repositories referenced in the program details and in-scope assets for activity analysis
- **📅 Recency Filters**: Find programs updated in the last X days/months or after a specific date
- **📦 Assets**: Retrieve in-scope assets for specific programs
- **🏷️ Tag Access**: Access categorized information (productType, ecosystem, programType, language)
- **📅 Date Information**: Retrieve launch and updated dates for programs
- **📋 KYC Status**: Check if KYC is required for specific programs
- **🏆 Rewards & Impacts**: Access detailed reward structures and impact categories
- **📋 Introspection**: List all available program IDs and fields from the API

## 🏗️ How It Works

The Immunefi MCP server acts as a bridge between MCP-compatible clients (like AI assistants) and the Immunefi bug bounty platform. It fetches data from the official Immunefi API, caches it for 6 hours to reduce API calls, and exposes a structured interface for querying bug bounty program information.

### 💡 Example: Complex scenario that requires multiple tools

>
> "List **GitHub repositories mentioned in the program details** of protocols that **updated their terms in the past 7 days**, reward **between $20,000 and $200,000**, and have **solidity** in the tags."

The example above, **gpt-oss:20b**, easily solves it by using the tools `search_updated_recently`, `get_max_bounty`, `get_tags`, and `search_github_repos`.

### Architecture

- Built with `mcp.server.fastmcp.FastMCP`
- Communicates via STDIO transport

## 🛠️ Tools Available

The server provides 18 specialized tools for interacting with Immunefi data:

### Search & Discovery

- `search_program(query: str)` - Search programs by name, ID, slug, or tags
- `get_all_project_ids()` - Retrieve all available project IDs
- `get_available_fields()` - List all available data fields

### Program Data Access

- `get_program_assets(project_ids: List[str])` - Retrieve in-scope assets
- `get_max_bounty(project_ids: List[str])` - Get maximum bounty amounts
- `get_launch_date(project_ids: List[str])` - Get program launch dates
- `get_updated_date(project_ids: List[str])` - Get last updated dates
- `is_kyc_required(project_ids: List[str])` - Check KYC requirements
- `get_rewards(project_ids: List[str])` - Access reward structures
- `get_impacts(project_ids: List[str])` - Get impact categories and descriptions
- `get_tags(project_ids: List[str])` - Access program tags (productType, ecosystem, etc.)

### Advanced Filtering

- `filter_by_bounty(min_bounty: int = 0, max_bounty: Optional[int] = None, project_ids: Optional[List[str]] = None)` - Filter by bounty range
- `filter_by_language(project_ids: List[str], language: str)` - Filter programs by language tag
- `filter_by_ecosystem(project_ids: List[str], ecosystem: str)` - Filter programs by ecosystem tag
- `search_updated_recently(days: Optional[int] = None, months: Optional[int] = None, project_ids: Optional[List[str]] = None)` - Find recently updated programs
- `search_updated_after_date(date: str, project_ids: Optional[List[str]] = None)` - Find programs updated after a specific date
- `get_field_values(project_ids: List[str], field_name: str)` - Get specific field values

### Code Discovery

- `search_github_repos(project_ids: List[str])` - Extract GitHub repositories from program data

## 📊 Data Sources

The server fetches data from the official Immunefi public API:

- All bounties: `https://immunefi.com/public-api/bounties.json`

## 📋 Prerequisites

- Python 3.8+
- `mcp` library (Model Context Protocol)

## 🚀 Installation

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Server

### Direct Execution

Run the server script directly (uses STDIO transport, intended to be launched by an MCP client):

```bash
python3 immunefi.py
```

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements or feature requests:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🐛 Issues & Support

If you encounter any issues or have questions:

- Open an issue in the repository

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Immunefi for providing the public API that makes this tool possible
