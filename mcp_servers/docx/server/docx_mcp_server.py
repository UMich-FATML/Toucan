"""
DOCX Processing MCP Server

Provides tools for Word document manipulation, including:
- Converting DOCX to markdown
- Unpacking/packing DOCX files for XML access
- Finding XML elements by tag, text, or line number
- Validating documents against XSD and redlining rules
- Adding comments and replies
- Suggesting tracked changes (deletions, insertions)
- XML node manipulation (replace, insert)
"""

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

# Add parent directories to path for imports
DOCX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DOCX_ROOT))

mcp = FastMCP("docx-processing")


# ==================== Read-Only Tools ====================


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,  # Requires pandoc to be installed
    }
)
def docx_to_markdown(docx_path: str) -> dict[str, Any]:
    """
    Convert a DOCX file to markdown format using pandoc.

    This tool converts Word documents to readable markdown text, making it
    easy to view and analyze document content without specialized software.

    Args:
        docx_path: Path to the DOCX file to convert

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - markdown: The converted markdown content (if successful)
        - message: Human-readable result description
    """
    try:
        result = subprocess.run(
            ["pandoc", "-f", "docx", "-t", "markdown", docx_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return {
                "success": False,
                "markdown": "",
                "message": f"Pandoc conversion failed: {result.stderr}",
            }

        return {
            "success": True,
            "markdown": result.stdout,
            "message": "Successfully converted DOCX to markdown",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "markdown": "",
            "message": "Pandoc not found. Please install pandoc to use this tool.",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "markdown": "",
            "message": "Conversion timed out after 60 seconds",
        }
    except Exception as e:
        return {
            "success": False,
            "markdown": "",
            "message": f"Error converting DOCX: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def docx_unpack(docx_path: str, output_dir: str) -> dict[str, Any]:
    """
    Unpack a DOCX file to a directory for direct XML access and editing.

    Extracts the DOCX archive (which is a ZIP file) and pretty-prints all XML
    files for easier reading and manipulation. This is the first step in the
    DOCX editing workflow.

    Workflow:
    1. docx_unpack() - Extract DOCX for editing
    2. docx_get_node() - Find elements to modify
    3. Edit tools (add_comment, suggest_deletion, etc.)
    4. docx_validate() - Check changes are valid
    5. docx_pack() - Repack to DOCX

    Args:
        docx_path: Path to the DOCX file to unpack
        output_dir: Directory where unpacked contents will be saved

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_dir: Path to the unpacked directory
        - suggested_rsid: Suggested RSID for tracked changes session
        - message: Human-readable result description
    """
    import random

    import defusedxml.minidom

    try:
        input_file = Path(docx_path)
        output_path = Path(output_dir)

        if not input_file.exists():
            return {
                "success": False,
                "output_dir": output_dir,
                "suggested_rsid": "",
                "message": f"DOCX file not found: {docx_path}",
            }

        # Extract the DOCX
        output_path.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(input_file).extractall(output_path)

        # Pretty print all XML files
        xml_files = list(output_path.rglob("*.xml")) + list(output_path.rglob("*.rels"))
        for xml_file in xml_files:
            content = xml_file.read_text(encoding="utf-8")
            dom = defusedxml.minidom.parseString(content)
            xml_file.write_bytes(dom.toprettyxml(indent="  ", encoding="ascii"))

        # Generate suggested RSID for edit session
        suggested_rsid = "".join(random.choices("0123456789ABCDEF", k=8))

        return {
            "success": True,
            "output_dir": output_dir,
            "suggested_rsid": suggested_rsid,
            "message": f"Successfully unpacked DOCX to {output_dir}. Use RSID {suggested_rsid} for tracked changes.",
        }
    except zipfile.BadZipFile:
        return {
            "success": False,
            "output_dir": output_dir,
            "suggested_rsid": "",
            "message": f"Invalid DOCX file (not a valid ZIP archive): {docx_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "output_dir": output_dir,
            "suggested_rsid": "",
            "message": f"Error unpacking DOCX: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def docx_get_node(
    unpacked_dir: str,
    xml_file: str,
    tag: str,
    attrs: Optional[dict[str, str]] = None,
    line_number: Optional[int] = None,
    line_range_start: Optional[int] = None,
    line_range_end: Optional[int] = None,
    contains: Optional[str] = None,
) -> dict[str, Any]:
    """
    Find an XML element in an unpacked DOCX by tag and optional filters.

    Searches for elements in the specified XML file using various criteria.
    Exactly one match must be found. This tool is used to locate elements
    before performing operations like adding comments or tracked changes.

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        xml_file: Relative path to XML file (e.g., "word/document.xml")
        tag: XML tag name to search for (e.g., "w:p", "w:r", "w:del", "w:ins")
        attrs: Optional dictionary of attribute name-value pairs to match
               (e.g., {"w:id": "1"} to find tracked change with id=1)
        line_number: Optional specific line number to match (1-indexed)
        line_range_start: Optional start of line range to search within
        line_range_end: Optional end of line range (exclusive) to search within
        contains: Optional text that must appear within the element

    Returns:
        A dictionary with:
        - success: Whether exactly one match was found
        - node_info: Information about the found node (tag, line, attributes, text preview)
        - message: Human-readable result description or error details
    """
    try:
        from scripts.utilities import XMLEditor

        xml_path = Path(unpacked_dir) / xml_file

        if not xml_path.exists():
            return {
                "success": False,
                "node_info": {},
                "message": f"XML file not found: {xml_file}",
            }

        editor = XMLEditor(xml_path)

        # Build line_number parameter
        search_line = None
        if line_number is not None:
            search_line = line_number
        elif line_range_start is not None and line_range_end is not None:
            search_line = range(line_range_start, line_range_end)

        # Find the node
        elem = editor.get_node(
            tag=tag,
            attrs=attrs,
            line_number=search_line,
            contains=contains,
        )

        # Extract node information
        parse_pos = getattr(elem, "parse_position", (None, None))
        elem_line, elem_col = parse_pos

        # Get text content preview
        text_content = editor._get_element_text(elem)
        text_preview = text_content[:100] + "..." if len(text_content) > 100 else text_content

        # Get attributes
        elem_attrs = {}
        if elem.attributes:
            for i in range(elem.attributes.length):
                attr = elem.attributes.item(i)
                elem_attrs[attr.name] = attr.value

        node_info = {
            "tag": elem.tagName,
            "line": elem_line,
            "column": elem_col,
            "attributes": elem_attrs,
            "text_preview": text_preview,
        }

        return {
            "success": True,
            "node_info": node_info,
            "message": f"Found <{tag}> at line {elem_line}",
        }

    except ValueError as e:
        return {
            "success": False,
            "node_info": {},
            "message": str(e),
        }
    except Exception as e:
        return {
            "success": False,
            "node_info": {},
            "message": f"Error finding node: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def docx_validate(
    unpacked_dir: str,
    original_docx_path: str,
    check_schema: bool = True,
    check_redlining: bool = True,
) -> dict[str, Any]:
    """
    Validate an unpacked DOCX against XSD schemas and redlining rules.

    Runs validation checks to ensure the modified document is well-formed
    and that tracked changes are properly structured. Should be called
    before repacking to catch errors early.

    Validation checks include:
    - XML well-formedness
    - Namespace declarations
    - Unique IDs
    - File references and relationships
    - Content type declarations
    - XSD schema compliance
    - Whitespace preservation
    - Tracked change structure (deletions/insertions)
    - Redlining validation (text matches after removing tracked changes)

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        original_docx_path: Path to original DOCX file (for comparison)
        check_schema: Run XSD schema validation (default: True)
        check_redlining: Run redlining validation (default: True)

    Returns:
        A dictionary with:
        - success: Whether all validation checks passed
        - schema_valid: Whether XSD schema validation passed (if run)
        - redlining_valid: Whether redlining validation passed (if run)
        - message: Human-readable result description
    """
    try:
        results = {
            "success": True,
            "schema_valid": None,
            "redlining_valid": None,
            "message": "",
        }

        messages = []

        if check_schema:
            from ooxml.scripts.validation.docx import DOCXSchemaValidator

            validator = DOCXSchemaValidator(
                unpacked_dir, original_docx_path, verbose=False
            )
            schema_valid = validator.validate()
            results["schema_valid"] = schema_valid
            if schema_valid:
                messages.append("Schema validation passed")
            else:
                messages.append("Schema validation failed")
                results["success"] = False

        if check_redlining:
            from ooxml.scripts.validation.redlining import RedliningValidator

            validator = RedliningValidator(
                unpacked_dir, original_docx_path, verbose=False
            )
            redlining_valid = validator.validate()
            results["redlining_valid"] = redlining_valid
            if redlining_valid:
                messages.append("Redlining validation passed")
            else:
                messages.append("Redlining validation failed")
                results["success"] = False

        results["message"] = "; ".join(messages) if messages else "No validation run"
        return results

    except Exception as e:
        return {
            "success": False,
            "schema_valid": False,
            "redlining_valid": False,
            "message": f"Validation error: {str(e)}",
        }


# ==================== Comment Tools ====================


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def docx_add_comment(
    unpacked_dir: str,
    start_tag: str,
    end_tag: str,
    comment_text: str,
    start_attrs: Optional[dict[str, str]] = None,
    end_attrs: Optional[dict[str, str]] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    start_contains: Optional[str] = None,
    end_contains: Optional[str] = None,
    author: str = "Claude",
    initials: str = "C",
) -> dict[str, Any]:
    """
    Add a comment spanning from one element to another in the document.

    Creates a new comment that spans the content between the start and end
    elements. The comment will appear in Word's review pane and can be
    replied to.

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        start_tag: Tag name of the starting element (e.g., "w:p", "w:r")
        end_tag: Tag name of the ending element
        comment_text: The text content of the comment
        start_attrs: Optional attributes to identify start element
        end_attrs: Optional attributes to identify end element
        start_line: Optional line number for start element
        end_line: Optional line number for end element
        start_contains: Optional text to search for in start element
        end_contains: Optional text to search for in end element
        author: Comment author name (default: "Claude")
        initials: Author initials (default: "C")

    Returns:
        A dictionary with:
        - success: Whether the comment was added successfully
        - comment_id: The ID of the created comment
        - message: Human-readable result description
    """
    try:
        from scripts.document import Document

        doc = Document(unpacked_dir, author=author, initials=initials)

        # Find start element
        start_elem = doc["word/document.xml"].get_node(
            tag=start_tag,
            attrs=start_attrs,
            line_number=start_line,
            contains=start_contains,
        )

        # Find end element
        end_elem = doc["word/document.xml"].get_node(
            tag=end_tag,
            attrs=end_attrs,
            line_number=end_line,
            contains=end_contains,
        )

        # Add comment
        comment_id = doc.add_comment(start=start_elem, end=end_elem, text=comment_text)

        # Save changes
        doc.save(validate=False)

        return {
            "success": True,
            "comment_id": comment_id,
            "message": f"Successfully added comment {comment_id} by {author}",
        }

    except ValueError as e:
        return {
            "success": False,
            "comment_id": -1,
            "message": f"Error finding element: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "comment_id": -1,
            "message": f"Error adding comment: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def docx_reply_to_comment(
    unpacked_dir: str,
    parent_comment_id: int,
    reply_text: str,
    author: str = "Claude",
    initials: str = "C",
) -> dict[str, Any]:
    """
    Add a reply to an existing comment in the document.

    Creates a reply that will appear nested under the parent comment
    in Word's review pane.

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        parent_comment_id: The w:id of the parent comment to reply to
        reply_text: The text content of the reply
        author: Reply author name (default: "Claude")
        initials: Author initials (default: "C")

    Returns:
        A dictionary with:
        - success: Whether the reply was added successfully
        - comment_id: The ID of the created reply comment
        - message: Human-readable result description
    """
    try:
        from scripts.document import Document

        doc = Document(unpacked_dir, author=author, initials=initials)

        # Add reply
        comment_id = doc.reply_to_comment(
            parent_comment_id=parent_comment_id, text=reply_text
        )

        # Save changes
        doc.save(validate=False)

        return {
            "success": True,
            "comment_id": comment_id,
            "message": f"Successfully added reply {comment_id} to comment {parent_comment_id}",
        }

    except ValueError as e:
        return {
            "success": False,
            "comment_id": -1,
            "message": f"Error: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "comment_id": -1,
            "message": f"Error adding reply: {str(e)}",
        }


# ==================== Tracked Changes Tools ====================


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def docx_suggest_deletion(
    unpacked_dir: str,
    tag: str,
    attrs: Optional[dict[str, str]] = None,
    line_number: Optional[int] = None,
    contains: Optional[str] = None,
    author: str = "Claude",
) -> dict[str, Any]:
    """
    Mark content as deleted using tracked changes.

    Wraps the specified element (w:r or w:p) in a deletion marker that will
    appear as strikethrough text in Word. The original content is preserved
    but marked for potential removal.

    For w:r (run): Wraps in <w:del>, converts <w:t> to <w:delText>
    For w:p (paragraph): Wraps content in <w:del>, handles numbered lists

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        tag: Tag of element to delete ("w:r" or "w:p")
        attrs: Optional attributes to identify the element
        line_number: Optional line number to locate the element
        contains: Optional text to search for within the element
        author: Author name for the tracked change (default: "Claude")

    Returns:
        A dictionary with:
        - success: Whether the deletion was marked successfully
        - message: Human-readable result description
    """
    try:
        from scripts.document import Document

        if tag not in ("w:r", "w:p"):
            return {
                "success": False,
                "message": "suggest_deletion only supports w:r or w:p elements",
            }

        doc = Document(unpacked_dir, author=author)

        # Find element
        elem = doc["word/document.xml"].get_node(
            tag=tag,
            attrs=attrs,
            line_number=line_number,
            contains=contains,
        )

        # Mark as deleted
        doc["word/document.xml"].suggest_deletion(elem)

        # Save changes
        doc.save(validate=False)

        return {
            "success": True,
            "message": f"Successfully marked <{tag}> as deleted",
        }

    except ValueError as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error suggesting deletion: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def docx_revert_insertion(
    unpacked_dir: str,
    tag: str,
    attrs: Optional[dict[str, str]] = None,
    line_number: Optional[int] = None,
    contains: Optional[str] = None,
    author: str = "Claude",
) -> dict[str, Any]:
    """
    Reject an insertion by wrapping its content in a deletion.

    For pre-redlined documents, this tool rejects another author's insertion
    by nesting <w:del> inside their <w:ins>. The insertion content becomes
    strikethrough text.

    Can process a single w:ins element or a container element (w:p, w:body)
    with multiple w:ins children.

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        tag: Tag of element containing insertion(s) ("w:ins", "w:p", "w:body")
        attrs: Optional attributes to identify the element
               (e.g., {"w:id": "5"} for specific w:ins)
        line_number: Optional line number to locate the element
        contains: Optional text to search for within the element
        author: Author name for the rejection (default: "Claude")

    Returns:
        A dictionary with:
        - success: Whether the insertion was rejected successfully
        - message: Human-readable result description
    """
    try:
        from scripts.document import Document

        doc = Document(unpacked_dir, author=author)

        # Find element
        elem = doc["word/document.xml"].get_node(
            tag=tag,
            attrs=attrs,
            line_number=line_number,
            contains=contains,
        )

        # Revert insertion
        doc["word/document.xml"].revert_insertion(elem)

        # Save changes
        doc.save(validate=False)

        return {
            "success": True,
            "message": f"Successfully rejected insertion in <{tag}>",
        }

    except ValueError as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error reverting insertion: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def docx_revert_deletion(
    unpacked_dir: str,
    tag: str,
    attrs: Optional[dict[str, str]] = None,
    line_number: Optional[int] = None,
    contains: Optional[str] = None,
    author: str = "Claude",
) -> dict[str, Any]:
    """
    Reject a deletion by re-inserting the deleted content.

    For pre-redlined documents, this tool restores another author's deletion
    by creating a new <w:ins> element after the <w:del>. The deleted content
    is copied and marked as a new insertion.

    Can process a single w:del element or a container element (w:p, w:body)
    with multiple w:del children.

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        tag: Tag of element containing deletion(s) ("w:del", "w:p", "w:body")
        attrs: Optional attributes to identify the element
               (e.g., {"w:id": "3"} for specific w:del)
        line_number: Optional line number to locate the element
        contains: Optional text to search for within the element
        author: Author name for the restoration (default: "Claude")

    Returns:
        A dictionary with:
        - success: Whether the deletion was restored successfully
        - message: Human-readable result description
    """
    try:
        from scripts.document import Document

        doc = Document(unpacked_dir, author=author)

        # Find element
        elem = doc["word/document.xml"].get_node(
            tag=tag,
            attrs=attrs,
            line_number=line_number,
            contains=contains,
        )

        # Revert deletion
        doc["word/document.xml"].revert_deletion(elem)

        # Save changes
        doc.save(validate=False)

        return {
            "success": True,
            "message": f"Successfully restored deleted content in <{tag}>",
        }

    except ValueError as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error reverting deletion: {str(e)}",
        }


# ==================== XML Manipulation Tools ====================


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def docx_replace_node(
    unpacked_dir: str,
    tag: str,
    new_xml: str,
    attrs: Optional[dict[str, str]] = None,
    line_number: Optional[int] = None,
    contains: Optional[str] = None,
    author: str = "Claude",
) -> dict[str, Any]:
    """
    Replace an XML node with new content.

    Finds the specified node and replaces it with the provided XML content.
    Automatically injects RSID, author, and date attributes where applicable.

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        tag: Tag of element to replace
        new_xml: XML string to replace the node with
        attrs: Optional attributes to identify the element
        line_number: Optional line number to locate the element
        contains: Optional text to search for within the element
        author: Author name for tracked change attributes (default: "Claude")

    Returns:
        A dictionary with:
        - success: Whether the replacement was successful
        - nodes_inserted: Number of nodes inserted
        - message: Human-readable result description
    """
    try:
        from scripts.document import Document

        doc = Document(unpacked_dir, author=author)

        # Find element
        elem = doc["word/document.xml"].get_node(
            tag=tag,
            attrs=attrs,
            line_number=line_number,
            contains=contains,
        )

        # Replace node
        nodes = doc["word/document.xml"].replace_node(elem, new_xml)

        # Save changes
        doc.save(validate=False)

        return {
            "success": True,
            "nodes_inserted": len(nodes),
            "message": f"Successfully replaced <{tag}> with {len(nodes)} node(s)",
        }

    except ValueError as e:
        return {
            "success": False,
            "nodes_inserted": 0,
            "message": f"Error: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "nodes_inserted": 0,
            "message": f"Error replacing node: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
def docx_insert_content(
    unpacked_dir: str,
    tag: str,
    xml_content: str,
    position: str,
    attrs: Optional[dict[str, str]] = None,
    line_number: Optional[int] = None,
    contains: Optional[str] = None,
    author: str = "Claude",
) -> dict[str, Any]:
    """
    Insert XML content before or after an element.

    Finds the specified element and inserts new XML content relative to it.
    Automatically injects RSID, author, and date attributes where applicable.

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        tag: Tag of reference element
        xml_content: XML string to insert
        position: Where to insert: "before", "after", or "append" (as child)
        attrs: Optional attributes to identify the element
        line_number: Optional line number to locate the element
        contains: Optional text to search for within the element
        author: Author name for tracked change attributes (default: "Claude")

    Returns:
        A dictionary with:
        - success: Whether the insertion was successful
        - nodes_inserted: Number of nodes inserted
        - message: Human-readable result description
    """
    try:
        from scripts.document import Document

        if position not in ("before", "after", "append"):
            return {
                "success": False,
                "nodes_inserted": 0,
                "message": "Position must be 'before', 'after', or 'append'",
            }

        doc = Document(unpacked_dir, author=author)

        # Find element
        elem = doc["word/document.xml"].get_node(
            tag=tag,
            attrs=attrs,
            line_number=line_number,
            contains=contains,
        )

        # Insert content
        editor = doc["word/document.xml"]
        if position == "before":
            nodes = editor.insert_before(elem, xml_content)
        elif position == "after":
            nodes = editor.insert_after(elem, xml_content)
        else:  # append
            nodes = editor.append_to(elem, xml_content)

        # Save changes
        doc.save(validate=False)

        return {
            "success": True,
            "nodes_inserted": len(nodes),
            "message": f"Successfully inserted {len(nodes)} node(s) {position} <{tag}>",
        }

    except ValueError as e:
        return {
            "success": False,
            "nodes_inserted": 0,
            "message": f"Error: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "nodes_inserted": 0,
            "message": f"Error inserting content: {str(e)}",
        }


# ==================== Pack Tool ====================


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def docx_pack(
    unpacked_dir: str,
    output_docx_path: str,
    validate: bool = False,
) -> dict[str, Any]:
    """
    Pack a directory back into a DOCX file.

    Reassembles the unpacked directory into a valid DOCX file. This is the
    final step in the editing workflow after making changes to the XML.

    The tool removes pretty-printing whitespace from XML files before packing
    to ensure Word can open the file correctly.

    Args:
        unpacked_dir: Path to unpacked DOCX directory
        output_docx_path: Path where the DOCX file will be saved
        validate: If True, validates with soffice before saving (default: False)

    Returns:
        A dictionary with:
        - success: Whether the packing was successful
        - output_path: Path to the created DOCX file
        - message: Human-readable result description
    """
    try:
        from ooxml.scripts.pack import pack_document

        success = pack_document(unpacked_dir, output_docx_path, validate=validate)

        if success:
            return {
                "success": True,
                "output_path": output_docx_path,
                "message": f"Successfully packed DOCX to {output_docx_path}",
            }
        else:
            return {
                "success": False,
                "output_path": output_docx_path,
                "message": "Packing failed - validation detected corruption. Use validate=False to force.",
            }

    except Exception as e:
        return {
            "success": False,
            "output_path": output_docx_path,
            "message": f"Error packing DOCX: {str(e)}",
        }


if __name__ == "__main__":
    mcp.run()
