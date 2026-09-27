from pathlib import Path
import subprocess

def tool_info():
    return {
        "name": "editor",
        "description": """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
\nNotes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`""",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`."
                },
                "file_text": {
                    "description": "Required parameter of `create` command, with the content of the file to be created.",
                    "type": "string"
                },
                "insert_line": {
                    "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                    "type": "integer"
                },
                "new_str": {
                    "description": "Required parameter of `str_replace` command containing the new string. Required parameter of `insert` command containing the string to insert.",
                    "type": "string"
                },
                "old_str": {
                    "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                    "type": "string"
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                    "type": "string"
                },
                "view_range": {
                    "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                    "items": {
                        "type": "integer"
                    },
                    "type": "array"
                }
            },
            "required": ["command", "path"]
        }
    }

class FileHistory:
    def __init__(self):
        self._history = {}
    
    def add(self, path, content):
        if path not in self._history:
            self._history[path] = []
        self._history[path].append(content)
    
    def undo(self, path):
        if path in self._history and self._history[path]:
            return self._history[path].pop()
        return None

# Global history instance
file_history = FileHistory()

def maybe_truncate(content: str, max_length: int = 10000) -> str:
    """Truncate long content and add marker."""
    if len(content) > max_length:
        return content[:max_length // 2] + "\n<response clipped>\n" + content[-max_length // 2:]
    return content

def validate_path(path: str, command: str):
    """Validate file path and command combination."""
    path_obj = Path(path)
    
    # Check if it's an absolute path
    if not path_obj.is_absolute():
        suggested_path = Path("") / path
        raise ValueError(f"The path {path} is not an absolute path, it should start with '/'. Maybe you meant {suggested_path}?")
    
    # Check existence for non-create commands
    if not path_obj.exists() and command != "create":
        raise ValueError(f"The path {path} does not exist. Please provide a valid path.")
    
    # Check if file already exists for create command
    if command == "create" and path_obj.exists():
        raise ValueError(f"File already exists at: {path}. Cannot overwrite files using command `create`.")
    
    # Check if it's a directory
    if path_obj.is_dir() and command != "view":
        raise ValueError(f"The path {path} is a directory and only the `view` command can be used on directories")
    
    return path_obj

def format_output(content: str, path: str, init_line: int = 1) -> str:
    """Format output with line numbers."""
    content = maybe_truncate(content)
    content = content.expandtabs()
    numbered_lines = [
        f"{i + init_line:6}\t{line}"
        for i, line in enumerate(content.split("\n"))
    ]
    return f"Here's the result of running `cat -n` on {path}:\n" + "\n".join(numbered_lines) + "\n"

def tool_function(command, path, file_text=None, view_range=None, old_str=None, new_str=None, insert_line=None):
    """Main tool function that handles different file operations."""
    try:
        path_obj = validate_path(path, command)
        
        if command == "view":
            return view_file(path_obj, view_range)
            
        elif command == "create":
            if not file_text:
                raise ValueError("Parameter `file_text` is required for command: create")
            write_file(path_obj, file_text)
            file_history.add(str(path_obj), file_text)
            return f"File created successfully at: {path}"
            
        elif command == "str_replace":
            if not old_str:
                raise ValueError("Parameter `old_str` is required for str_replace command")
            return replace_text(path_obj, old_str, new_str)
            
        elif command == "insert":
            if insert_line is None:
                raise ValueError("Parameter `insert_line` is required for insert command")
            if new_str is None:
                raise ValueError("Parameter `new_str` is required for insert command")
            return insert_text(path_obj, insert_line, new_str)
            
        elif command == "undo_edit":
            return undo_last_edit(path_obj)
            
        else:
            raise ValueError(f"Unknown command: {command}")
            
    except Exception as e:
        return f"Error: {str(e)}"

def read_file(path: Path) -> str:
    """Read and return file contents."""
    try:
        return path.read_text()
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

def write_file(path: Path, content: str):
    """Write content to file."""
    try:
        path.write_text(content)
    except Exception as e:
        raise ValueError(f"Failed to write file: {e}")

def view_file(path: Path, view_range=None) -> str:
    """View file or directory contents."""
    if path.is_dir():
        if view_range:
            raise ValueError("The `view_range` parameter is not allowed when `path` points to a directory.")
        
        try:
            # Use find command to list files and directories up to 2 levels deep
            result = subprocess.run(
                ['find', str(path), '-maxdepth', '2', '-not', '-path', '*/\\.*'],
                capture_output=True,
                text=True
            )
            if result.stderr:
                return f"Error listing directory: {result.stderr}"
            return f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{maybe_truncate(result.stdout, max_length=5000)}"
        except Exception as e:
            raise ValueError(f"Failed to list directory: {e}")
    
    content = read_file(path)
    
    if view_range:
        if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
            raise ValueError("Invalid `view_range`. It should be a list of two integers.")
            
        lines = content.split("\n")
        n_lines = len(lines)
        start, end = view_range
        
        if start < 1 or start > n_lines:
            raise ValueError(f"Invalid `view_range`: {view_range}. First element should be within range [1, {n_lines}]")
            
        if end != -1:
            if end > n_lines:
                raise ValueError(f"Invalid `view_range`: {view_range}. Second element should not exceed {n_lines}")
            if end < start:
                raise ValueError(f"Invalid `view_range`: {view_range}. Second element should be larger or equal to first")
                
        content = "\n".join(lines[start-1:end if end != -1 else None])
        return format_output(content, str(path), start)
    
    return format_output(content, str(path))

def replace_text(path: Path, old_str: str, new_str: str) -> str:
    """Replace text in file."""
    content = read_file(path).expandtabs()
    old_str = old_str.expandtabs()
    new_str = new_str.expandtabs() if new_str is not None else ""
    
    # Check for exact match and uniqueness
    occurrences = content.count(old_str)
    if occurrences == 0:
        raise ValueError(f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}")
    if occurrences > 1:
        lines = [idx + 1 for idx, line in enumerate(content.split("\n")) if old_str in line]
        raise ValueError(f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique")
    
    # Save to history and perform replacement
    file_history.add(str(path), content)
    new_content = content.replace(old_str, new_str)
    write_file(path, new_content)
    
    # Create snippet of edited section
    replacement_line = content.split(old_str)[0].count("\n")
    start_line = max(0, replacement_line - 4)
    end_line = replacement_line + 4 + new_str.count("\n")
    snippet = "\n".join(new_content.split("\n")[start_line:end_line + 1])
    
    return (f"The file {path} has been edited. " + 
            format_output(snippet, f"a snippet of {path}", start_line + 1) +
            "Review the changes and make sure they are as expected. Edit the file again if necessary.")

def insert_text(path: Path, insert_line: int, new_str: str) -> str:
    """Insert text after specific line number."""
    content = read_file(path).expandtabs()
    new_str = new_str.expandtabs()
    lines = content.split("\n")
    n_lines = len(lines)
    
    if insert_line < 0 or insert_line > n_lines:
        raise ValueError(f"Invalid `insert_line` parameter: {insert_line}. Should be within range [0, {n_lines}]")
    
    # Save to history
    file_history.add(str(path), content)
    
    # Insert the new text
    new_str_lines = new_str.split("\n")
    new_lines = lines[:insert_line] + new_str_lines + lines[insert_line:]
    
    # Create snippet
    snippet_lines = (
        lines[max(0, insert_line - 4):insert_line] +
        new_str_lines +
        lines[insert_line:insert_line + 4]
    )

    # Write new content
    write_file(path, "\n".join(new_lines))

    return (f"The file {path} has been edited. " +
            format_output("\n".join(snippet_lines), "a snippet of the edited file", max(1, insert_line - 4 + 1)) +
            "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary.")

def undo_last_edit(path: Path) -> str:
    """Undo last edit operation."""
    previous_content = file_history.undo(str(path))
    if previous_content is None:
        return f"No edit history found for {path}."

    write_file(path, previous_content)
    return f"Last edit to {path} undone successfully. {format_output(previous_content, str(path))}"

if __name__ == "__main__":
    # Example usage
    import os
    filepath = os.path.join(os.getcwd(), "README.md")
    result = tool_function("view", filepath, view_range=[1, -1])
    print(result)
