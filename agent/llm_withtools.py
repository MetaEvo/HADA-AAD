import re
import json
import sys

# Try to import global config, fallback to local imports if not available
try:
    sys.path.insert(0, '/hada')
    from config import MODEL_NAME, set_global_seed
    # Set global seed
    set_global_seed()
    DEFAULT_MODEL = MODEL_NAME
except ImportError:
    from agent.llm import QWEN_MODEL
    DEFAULT_MODEL = QWEN_MODEL

from agent.llm import get_response_from_llm, QWEN_MODEL, OPENAI_MINI_MODEL, DOUBAO_MODEL
from agent.tools import load_tools

def get_tooluse_prompt(tool_infos=[]):
    """
    Get the prompt for using the available tools.
    """
    # If no tools are available, return an empty string
    if not tool_infos or len(tool_infos) == 0:
        return ""
    # Create the prompt
    tools_available = [str(tool_info) for tool_info in tool_infos]
    tools_available = '\n\n'.join(tools_available) if tools_available else 'None'
    tooluse_prompt = """Here are the available tools:
```
{tools_available}
```

You can use tools in this format:
<json>
{{
    "tool_name": ...,
    "tool_input": ...
}}
</json>

You can use MULTIPLE tools in a single response by including multiple JSON blocks.
STRICTLY FOLLOW THE FORMAT OF TOOL_NAME AND TOOL_INPUT ABOVE.
DO NOT HALLUCINATE OR MAKE UP ANYTHING.
""".format(tools_available=tools_available)
    return tooluse_prompt.strip()

def should_retry_tool_use(response, tool_uses=None):
    """
    Check if the response attempts to use a tool,
    but ran out of output context.
    """
    # If there are tool uses, we don't need to check for retry
    if tool_uses is not None and len(tool_uses) > 0:
        return False

    # Find positions of the markers
    json_pos = response.find("<json>")
    tool_name_pos = response.find("tool_name")
    tool_input_pos = response.find("tool_input")

    # Check ordering and length condition
    if (
        json_pos != -1
        and tool_name_pos != -1
        and tool_input_pos != -1
        and json_pos < tool_name_pos < tool_input_pos
        and len(response) >= 2000
    ):
        return True

    # No retry
    return False

def check_for_tool_uses(response):
    """
    Checks if the response contains one or more tool calls in json code blocks.
    Returns a list of tool use dictionaries.
    """
    pattern = r'<json>\s*(\{.*?\})\s*</json>'
    matches = re.findall(pattern, response, re.DOTALL)
    tool_uses = []

    for match in matches:
        try:
            tool_use = json.loads(match)
            if 'tool_name' not in tool_use or 'tool_input' not in tool_use:
                continue  # Skip invalid tool use
            tool_uses.append(tool_use)
        except json.JSONDecodeError:
            continue  # Skip malformed JSON blocks

    return tool_uses if tool_uses else None

def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"

def chat_with_agent(
    msg,
    model=DEFAULT_MODEL,
    msg_history=None,
    logging=print,
    tools_available='all',  # Empty list means no tools, 'all' means all tools
    multiple_tool_calls=False,  # Whether to allow multiple tool calls in a single response
    max_tool_calls=80,  # Maximum number of tool calls allowed in a single response, -1 for unlimited
    require_tool=None,  # Tool name that must be called at least once (e.g., 'str_replace' via editor)
):
    if tools_available is None:
        tools_available = []
    get_response_fn = get_response_from_llm
    # Construct message
    if msg_history is None:
        msg_history = []
    new_msg_history = msg_history

    try:
        # Load all tools
        all_tools = load_tools(logging=logging, names=tools_available)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        system_msg = f"{get_tooluse_prompt([tool['info'] for tool in all_tools])}\n\n"
        num_tool_calls = 0
        tools_called = set()

        # Call API
        logging(f"Input: {repr(msg)}")
        response, new_msg_history, info = get_response_fn(
            msg=system_msg + msg,
            model=model,
            msg_history=new_msg_history,
        )
        logging(f"Output: {repr(response)}")
        # logging(f"Info: {repr(info)}")

        # Tool use
        tool_uses = check_for_tool_uses(response)
        retry_tool_use = should_retry_tool_use(response, tool_uses)
        while tool_uses or retry_tool_use:
            # Check for max tool calls
            if max_tool_calls > 0 and num_tool_calls >= max_tool_calls:
                logging("Error: Maximum number of tool calls reached.")
                break

            tool_msgs = []

            # Process tool uses
            if tool_uses:
                tool_uses = tool_uses if multiple_tool_calls else tool_uses[:1]
                for tool_use in tool_uses:
                    tool_name = tool_use['tool_name']
                    tool_input = tool_use['tool_input']
                    tool_output = process_tool_call(tools_dict, tool_name, tool_input)
                    num_tool_calls += 1
                    tools_called.add(tool_name)
                    # Track str_replace calls via editor tool
                    if tool_name == 'editor' and isinstance(tool_input, dict) and tool_input.get('command') == 'str_replace':
                        tools_called.add('str_replace')
                    tool_msg = f'''<json>
    {{
        "tool_name": "{tool_name}",
        "tool_input": {tool_input},
        "tool_output": "{tool_output}"
    }}
    </json>'''.strip()
                    logging(f"Tool output: {repr(tool_msg)}")
                    tool_msgs.append(tool_msg)

            # Check for retry
            if retry_tool_use:
                logging("Error: Output context exceeded. Please try again.")
                tool_msgs.append("Error: Output context exceeded. Please try again.")

            # Get tool response
            response, new_msg_history, info = get_response_fn(
                msg=system_msg + '\n\n'.join(tool_msgs),
                model=model,
                msg_history=new_msg_history,
            )
            logging(f"Output: {repr(response)}")
            # logging(f"Info: {repr(info)}")

            # Check for next tool use
            tool_uses = check_for_tool_uses(response)
            retry_tool_use = should_retry_tool_use(response, tool_uses)

            # Check if required tool was called - if not, force continuation
            if require_tool and require_tool not in tools_called and not tool_uses:
                logging(f"Warning: Required tool '{require_tool}' not called yet. Forcing continuation.")
                tool_msgs_str = f"\n\n⚠️ You have NOT used the '{require_tool}' tool yet. You MUST use it before responding with final JSON."
                tool_msgs.append(tool_msgs_str)
                tool_uses = []  # Force another iteration
                retry_tool_use = True  # Force loop to continue
                logging(f"Forcing retry with tool_msgs: {repr(tool_msgs)}")

        # Final check: if required tool was never called, add warning to message history
        if require_tool and require_tool not in tools_called:
            logging(f"WARNING: Required tool '{require_tool}' was never called!")
            warning_msg = f"\n\n⚠️ CRITICAL: You did NOT use the '{require_tool}' tool as required. This will result in ZERO score."
            new_msg_history.append({"role": "user", "content": warning_msg})

    except Exception as e:
        logging(f"Error: {str(e)}")
        raise e

    return new_msg_history

if __name__ == "__main__":
    msg = """hello"""
    new_msg_history = chat_with_agent(msg)