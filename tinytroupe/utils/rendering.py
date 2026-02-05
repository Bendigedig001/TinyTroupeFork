import json
import textwrap
from datetime import datetime
from typing import Union
import inspect

from tinytroupe.utils import logger


################################################################################
# Rendering and markup 
################################################################################
def inject_html_css_style_prefix(html, style_prefix_attributes):
    """
    Injects a style prefix to all style attributes in the given HTML string.

    For example, if you want to add a style prefix to all style attributes in the HTML string
    ``<div style="color: red;">Hello</div>``, you can use this function as follows:
    inject_html_css_style_prefix('<div style="color: red;">Hello</div>', 'font-size: 20px;')
    """
    return html.replace('style="', f'style="{style_prefix_attributes};')

def break_text_at_length(text: Union[str, dict, list], max_length: int=None) -> str:
    """
    Breaks the text (or JSON) at the specified length, inserting a "(...)" string at the break point.
    If the maximum length is `None`, the content is returned as is.
    """
    if isinstance(text, list):
        text_chunks = []
        image_count = 0
        other_count = 0
        for part in text:
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type in {"text", "input_text"}:
                    text_chunks.append(str(part.get("text", "")))
                elif part_type in {"image_url", "input_image"}:
                    image_count += 1
                else:
                    other_count += 1
            else:
                other_count += 1
        summary = "\n".join([t for t in text_chunks if t.strip()])
        if image_count:
            suffix = f"[Attached images: {image_count}]"
            summary = f"{summary}\n{suffix}" if summary else suffix
        if other_count:
            suffix = f"[Other parts: {other_count}]"
            summary = f"{summary}\n{suffix}" if summary else suffix
        text = summary
    elif isinstance(text, dict):
        text = json.dumps(text, indent=4)

    if max_length is None or len(text) <= max_length:
        return text
    else:
        return text[:max_length] + " (...)"

def pretty_datetime(dt: datetime) -> str:
    """
    Returns a pretty string representation of the specified datetime object.
    """
    return dt.strftime("%Y-%m-%d %H:%M")

def dedent(text: str) -> str:
    """
    Dedents the specified text, removing any leading whitespace and identation.
    """
    return textwrap.dedent(text).strip()

def wrap_text(text: str, width: int=100) -> str:
    """
    Wraps the text at the specified width.
    """
    return textwrap.fill(text, width=width)


def indent_at_current_level(text: str) -> str:
    """
    Indents the specified text at the current indentation level, determined dynamically.
    """
    frame = inspect.currentframe().f_back
    line = frame.f_lineno
    filename = frame.f_code.co_filename
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    current_line = lines[line - 1]
    
    indent= len(current_line) - len(current_line.lstrip())

    # first dedent the text to remove any leading whitespace
    text = dedent(text)

    # then indent it to the specified level
    return textwrap.indent(text, ' ' * indent)


class RichTextStyle:
    
    # Consult color options here: https://rich.readthedocs.io/en/stable/appendix/colors.html

    STIMULUS_CONVERSATION_STYLE = "bold italic cyan1"
    STIMULUS_THOUGHT_STYLE = "dim italic cyan1"
    STIMULUS_DEFAULT_STYLE = "italic"
    
    ACTION_DONE_STYLE = "grey82"
    ACTION_TALK_STYLE = "bold green3"
    ACTION_THINK_STYLE = "green"
    ACTION_DEFAULT_STYLE = "purple"

    INTERVENTION_DEFAULT_STYLE = "bright_magenta"

    @classmethod
    def get_style_for(cls, kind:str, event_type:str=None):
        if kind == "stimulus" or kind=="stimuli":
            if event_type == "CONVERSATION":
                return cls.STIMULUS_CONVERSATION_STYLE
            elif event_type == "THOUGHT":
                return cls.STIMULUS_THOUGHT_STYLE
            else:
                return cls.STIMULUS_DEFAULT_STYLE
            
        elif kind == "action":
            if event_type == "DONE":
                return cls.ACTION_DONE_STYLE
            elif event_type == "TALK":
                return cls.ACTION_TALK_STYLE
            elif event_type == "THINK":
                return cls.ACTION_THINK_STYLE
            else:
                return cls.ACTION_DEFAULT_STYLE
        
        elif kind == "intervention":
            return cls.INTERVENTION_DEFAULT_STYLE
