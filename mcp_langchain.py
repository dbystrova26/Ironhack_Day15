from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP
import mcp


WORKSPACE_ROOT = Path(__file__).resolve().parent


def load_environment() -> None:
    """Load environment variables from the local .env file."""
    env_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=env_path, override=False)


def verify_openai_api_key() -> str:
    """Return a masked OpenAI key after verifying it is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    masked = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
    print(f"OPENAI_API_KEY loaded: {masked}")
    return api_key


def print_tradeoffs() -> None:
    """Document when MCP is preferable versus direct API calls."""
    print("MCP vs direct API trade-offs:")
    print("- MCP: better when you want reusable tools, standardized server boundaries, and multiple integrations.")
    print("- MCP: adds setup overhead and an extra protocol layer.")
    print("- Direct API: better when you only need one custom flow and want the shortest path to the model.")
    print("- Direct API: puts more integration logic in your code and is harder to reuse across frameworks.")


def collect_workspace_snapshot(root: Path, max_files: int = 50) -> dict[str, object]:
    """Collect local filesystem context without using MCP."""
    files = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    )
    script_path = "mcp_langchain.py" if (root / "mcp_langchain.py").exists() else None
    return {
        "root": str(root),
        "file_count": len(files),
        "files": files[:max_files],
        "script_path": script_path,
    }


async def run_direct_api_example(api_key: str, workspace_snapshot: dict[str, object]) -> None:
    """Show the same task with a direct OpenAI API call."""
    client = AsyncOpenAI(api_key=api_key)
    prompt = (
        "You are given a local workspace snapshot. "
        "Summarize the workspace and identify the script file that implements the MCP example.\n\n"
        f"Workspace snapshot:\n{json.dumps(workspace_snapshot, indent=2)}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You analyze workspace metadata provided by the caller."},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:
        print(f"Direct API demo skipped because the model call failed: {type(exc).__name__}: {exc}")
        return

    message = response.choices[0].message.content or ""
    print("Direct API response:")
    print(message)


def build_filesystem_server(root: Path) -> FastMCP:
    """Create a small filesystem MCP server backed by a local directory."""
    server = FastMCP("local-filesystem")

    @server.tool(description="List files and folders under a local root directory.")
    def list_workspace_files(relative_path: str = ".") -> list[str]:
        target = (root / relative_path).resolve()
        if root not in target.parents and target != root:
            raise ValueError("Path escapes the configured workspace root.")

        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {relative_path}")

        if target.is_file():
            return [str(target.relative_to(root))]

        return sorted(
            str(path.relative_to(root))
            for path in target.rglob("*")
            if path.is_file()
        )

    @server.tool(description="Read a text file from the local workspace.")
    def read_workspace_file(relative_path: str) -> str:
        target = (root / relative_path).resolve()
        if root not in target.parents and target != root:
            raise ValueError("Path escapes the configured workspace root.")
        return target.read_text(encoding="utf-8")

    @server.resource("filesystem://workspace-root", description="Workspace root summary.")
    def workspace_root_resource() -> str:
        files = list_workspace_files(".")
        return json.dumps(
            {"root": str(root), "file_count": len(files), "files": files[:50]},
            indent=2,
        )

    @server.resource("filesystem://{relative_path}", description="Read a file as a resource.")
    def workspace_file_resource(relative_path: str) -> str:
        return read_workspace_file(relative_path)

    return server


async def run_filesystem_server(root: Path) -> None:
    """Run the local filesystem MCP server over stdio."""
    server = build_filesystem_server(root)
    await server.run_stdio_async()


async def connect_to_filesystem_server() -> None:
    """Launch the local filesystem MCP server and probe its tools/resources."""
    load_environment()
    api_key = verify_openai_api_key()
    print_tradeoffs()
    workspace_snapshot = collect_workspace_snapshot(WORKSPACE_ROOT)

    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", "unknown")
    print(f"LangChain model initialized: {model_name}", flush=True)

    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(Path(__file__).resolve()), "--serve-filesystem-mcp", str(WORKSPACE_ROOT)],
                "cwd": str(WORKSPACE_ROOT),
            }
        }
    )

    print("Connecting to filesystem MCP server...", flush=True)
    tools = await client.get_tools(server_name="filesystem")
    langchain_resources = await client.get_resources(server_name="filesystem")

    async with client.session("filesystem") as session:
        resource_result = await session.read_resource("filesystem://workspace-root")

    print("Connected MCP server: filesystem", flush=True)
    print("Available LangChain tools:")
    for tool in tools:
        tool_type = type(tool).__name__
        print(f"- {tool.name} ({tool_type})")

    if tools:
        sample_tool = tools[0]
        print(f"Invoking sample tool: {sample_tool.name}")
        if sample_tool.name == "list_workspace_files":
            result = await sample_tool.ainvoke({"relative_path": "."})
        elif sample_tool.name == "read_workspace_file":
            result = await sample_tool.ainvoke({"relative_path": "mcp_langchain.py"})
        else:
            result = await sample_tool.ainvoke({})
        print(f"Sample tool result: {str(result)[:160].replace(chr(10), ' ')}")

    print("Available LangChain resources:")
    for resource in langchain_resources:
        source = getattr(resource, "source", "unknown")
        preview = resource.as_string() if hasattr(resource, "as_string") else str(resource)
        print(f"- {source}: {preview[:120].replace(chr(10), ' ')}")

    print(f"LangChain tool count: {len(tools)}")
    print(f"LangChain resource count: {len(langchain_resources)}")
    resource_context = []
    for content in getattr(resource_result, "contents", []):
        if hasattr(content, "text"):
            resource_context.append(content.text)
        elif hasattr(content, "blob"):
            resource_context.append(content.blob)
    resource_context_text = "\n\n".join(resource_context) if resource_context else "No resource content available."

    system_prompt = (
        "You are connected to a local filesystem MCP server through LangChain. "
        "Use the available tools to inspect the workspace when needed. "
        "The tools include listing workspace files and reading workspace files. "
        "Prefer the tools for any question about files in this project.\n\n"
        "MCP resource context:\n"
        f"{resource_context_text}"
    )
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    print("LangChain agent created with MCP tools.")

    try:
        agent_result = await agent.ainvoke(
            {
                "messages": [
                    (
                        "user",
                        "Use the filesystem tools to list the first 5 files in this workspace "
                        "and tell me which file contains the MCP LangChain script.",
                    )
                ]
            }
        )
    except Exception as exc:
        print(f"Agent test skipped because the model call failed: {type(exc).__name__}: {exc}")
    else:
        final_message = agent_result["messages"][-1]
        print("Agent response:")
        print(getattr(final_message, "content", str(final_message)))

    await run_direct_api_example(api_key, workspace_snapshot)
    print(f"mcp package version loaded from: {mcp.__file__}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangChain + MCP filesystem example")
    parser.add_argument(
        "--serve-filesystem-mcp",
        nargs="?",
        const=str(WORKSPACE_ROOT),
        help="Run the local filesystem MCP server instead of the client demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.serve_filesystem_mcp is not None:
        asyncio.run(run_filesystem_server(Path(args.serve_filesystem_mcp)))
        return

    asyncio.run(connect_to_filesystem_server())


if __name__ == "__main__":
    main()
