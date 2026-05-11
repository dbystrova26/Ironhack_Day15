# Ironhack Day 15 MCP Lab

## How to Run

1. Open a terminal in this folder:
   `C:\Users\dbyst\OneDrive\Desktop\Ironhack_labs\Ironhack_Day15\Ironhack_Day15`
2. Make sure `.env` contains `OPENAI_API_KEY`.
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Run the main script:
   ```bash
   python mcp_langchain.py
   ```
5. Optional: start only the local filesystem MCP server:
   ```bash
   python mcp_langchain.py --serve-filesystem-mcp
   ```

## File Map

- `mcp_langchain.py` - main MCP and LangChain example, including filesystem server, tool loading, resource loading, agent setup, and direct API comparison
- `requirements.txt` - Python dependencies used by the script
- `mcp_introduction.ipynb` - notebook version of the MCP and LangChain walkthrough
- `OLD/` - previous notebook and archived working files
- `.env` - local environment variables, including `OPENAI_API_KEY`
- `.gitignore` - ignored files and folders for the lab
- `README.md` - run instructions and repository map
- `lab_summary.md` - short narrative summary of the lab
