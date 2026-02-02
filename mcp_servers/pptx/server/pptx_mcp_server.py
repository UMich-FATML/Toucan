"""
PPTX Processing MCP Server

Provides tools for PowerPoint presentation processing, including:
- Extracting structured text inventory with formatting
- Applying text replacements preserving formatting
- Rearranging, duplicating, or deleting slides
- Creating visual thumbnail grids
- Unpacking/packing Office XML for direct editing
- Validating Office XML structure
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
OOXML_SCRIPTS_DIR = Path(__file__).parent.parent / "ooxml" / "scripts"
# Add docx validation module path (shared validation code)
DOCX_OOXML_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "docx" / "ooxml" / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(OOXML_SCRIPTS_DIR))
sys.path.insert(0, str(DOCX_OOXML_SCRIPTS_DIR))

mcp = FastMCP("pptx-processing")


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pptx_extract_inventory(
    pptx_path: str,
    output_json_path: str,
    issues_only: bool = False
) -> dict[str, Any]:
    """
    Extract structured text inventory from a PowerPoint presentation.

    Extracts all text content from PowerPoint shapes with formatting information,
    positions, overflow detection, and overlap detection. The inventory is
    organized by slide and shape, with shapes sorted by visual position.

    Args:
        pptx_path: Path to the input PowerPoint file (.pptx)
        output_json_path: Path where the inventory JSON will be saved
        issues_only: If True, only include shapes with overflow or overlap issues

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_path: Path to the created JSON file
        - slide_count: Number of slides with text content
        - shape_count: Total number of text shapes extracted
        - message: Human-readable result description
    """
    try:
        from inventory import extract_text_inventory, save_inventory

        input_path = Path(pptx_path)
        if not input_path.exists():
            return {
                "success": False,
                "output_path": output_json_path,
                "slide_count": 0,
                "shape_count": 0,
                "message": f"Input file not found: {pptx_path}",
            }

        if not input_path.suffix.lower() == ".pptx":
            return {
                "success": False,
                "output_path": output_json_path,
                "slide_count": 0,
                "shape_count": 0,
                "message": "Input must be a PowerPoint file (.pptx)",
            }

        inventory = extract_text_inventory(input_path, issues_only=issues_only)

        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_inventory(inventory, output_path)

        slide_count = len(inventory)
        shape_count = sum(len(shapes) for shapes in inventory.values())

        if issues_only:
            if shape_count > 0:
                message = f"Found {shape_count} text elements with issues in {slide_count} slides"
            else:
                message = "No issues discovered"
        else:
            message = f"Extracted text from {slide_count} slides with {shape_count} text elements"

        return {
            "success": True,
            "output_path": output_json_path,
            "slide_count": slide_count,
            "shape_count": shape_count,
            "message": message,
        }
    except Exception as e:
        return {
            "success": False,
            "output_path": output_json_path,
            "slide_count": 0,
            "shape_count": 0,
            "message": f"Error extracting inventory: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pptx_apply_replacements(
    input_pptx_path: str,
    replacements_json_path: str,
    output_pptx_path: str
) -> dict[str, Any]:
    """
    Apply text replacements to a PowerPoint presentation.

    Takes a PowerPoint file and a replacements JSON (in the format output by
    pptx_extract_inventory), then produces a new PowerPoint with the text
    replaced while preserving formatting. ALL text shapes identified in the
    inventory will have their text cleared unless "paragraphs" is specified
    in the replacements for that shape.

    The replacements JSON should have the structure:
    {
        "slide-0": {
            "shape-0": {
                "paragraphs": [
                    {"text": "New title", "font_size": 24, "bold": true}
                ]
            }
        }
    }

    Args:
        input_pptx_path: Path to the input PowerPoint file
        replacements_json_path: Path to JSON file with replacement text
        output_pptx_path: Path where the modified PowerPoint will be saved

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_path: Path to the created PowerPoint file
        - message: Human-readable result description
    """
    try:
        from replace import apply_replacements

        input_path = Path(input_pptx_path)
        if not input_path.exists():
            return {
                "success": False,
                "output_path": output_pptx_path,
                "message": f"Input file not found: {input_pptx_path}",
            }

        replacements_path = Path(replacements_json_path)
        if not replacements_path.exists():
            return {
                "success": False,
                "output_path": output_pptx_path,
                "message": f"Replacements file not found: {replacements_json_path}",
            }

        output_path = Path(output_pptx_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        apply_replacements(str(input_path), str(replacements_path), str(output_path))

        return {
            "success": True,
            "output_path": output_pptx_path,
            "message": f"Successfully applied replacements and saved to {output_pptx_path}",
        }
    except ValueError as e:
        return {
            "success": False,
            "output_path": output_pptx_path,
            "message": f"Validation error: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "output_path": output_pptx_path,
            "message": f"Error applying replacements: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pptx_rearrange_slides(
    template_pptx_path: str,
    output_pptx_path: str,
    slide_sequence: list[int]
) -> dict[str, Any]:
    """
    Rearrange slides in a PowerPoint presentation.

    Creates a new presentation with slides from the template in the specified
    order. Slides can be reordered, duplicated (by including the same index
    multiple times), or omitted (by not including their index).

    Args:
        template_pptx_path: Path to the template PowerPoint file
        output_pptx_path: Path where the rearranged PowerPoint will be saved
        slide_sequence: List of 0-based slide indices specifying the new order
                       (e.g., [0, 2, 2, 1] reorders and duplicates slide 2)

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_path: Path to the created PowerPoint file
        - original_slide_count: Number of slides in the original presentation
        - final_slide_count: Number of slides in the output presentation
        - message: Human-readable result description
    """
    try:
        from rearrange import rearrange_presentation
        from pptx import Presentation

        template_path = Path(template_pptx_path)
        if not template_path.exists():
            return {
                "success": False,
                "output_path": output_pptx_path,
                "original_slide_count": 0,
                "final_slide_count": 0,
                "message": f"Template file not found: {template_pptx_path}",
            }

        # Get original slide count
        prs = Presentation(str(template_path))
        original_count = len(prs.slides)

        output_path = Path(output_pptx_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rearrange_presentation(template_path, output_path, slide_sequence)

        return {
            "success": True,
            "output_path": output_pptx_path,
            "original_slide_count": original_count,
            "final_slide_count": len(slide_sequence),
            "message": f"Rearranged {original_count} slides into {len(slide_sequence)} slides",
        }
    except ValueError as e:
        return {
            "success": False,
            "output_path": output_pptx_path,
            "original_slide_count": 0,
            "final_slide_count": 0,
            "message": f"Invalid slide sequence: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "output_path": output_pptx_path,
            "original_slide_count": 0,
            "final_slide_count": 0,
            "message": f"Error rearranging slides: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pptx_create_thumbnails(
    pptx_path: str,
    output_prefix: str = "thumbnails",
    cols: int = 5,
    outline_placeholders: bool = False
) -> dict[str, Any]:
    """
    Create visual thumbnail grids from PowerPoint slides.

    Generates grid images showing slide thumbnails with labels. For large
    presentations, multiple grid files are created automatically. Each grid
    contains up to cols*(cols+1) images (e.g., 30 slides for 5 columns).

    Args:
        pptx_path: Path to the input PowerPoint file
        output_prefix: Prefix for output image files (default: "thumbnails")
                      Creates prefix.jpg or prefix-N.jpg for multiple grids
        cols: Number of columns in the grid (default: 5, max: 6)
        outline_placeholders: If True, draw red outlines around text shapes

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - grid_files: List of paths to created grid images
        - slide_count: Number of slides processed
        - message: Human-readable result description
    """
    try:
        from thumbnail import convert_to_images, create_grids, get_placeholder_regions
        import tempfile

        input_path = Path(pptx_path)
        if not input_path.exists():
            return {
                "success": False,
                "grid_files": [],
                "slide_count": 0,
                "message": f"Input file not found: {pptx_path}",
            }

        if not input_path.suffix.lower() == ".pptx":
            return {
                "success": False,
                "grid_files": [],
                "slide_count": 0,
                "message": "Input must be a PowerPoint file (.pptx)",
            }

        # Limit columns to max 6
        cols = min(cols, 6)
        thumbnail_width = 300

        output_path = Path(f"{output_prefix}.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Get placeholder regions if outlining is enabled
            placeholder_regions = None
            slide_dimensions = None
            if outline_placeholders:
                placeholder_regions, slide_dimensions = get_placeholder_regions(input_path)

            # Convert slides to images
            slide_images = convert_to_images(input_path, Path(temp_dir), dpi=100)
            if not slide_images:
                return {
                    "success": False,
                    "grid_files": [],
                    "slide_count": 0,
                    "message": "No slides found in presentation",
                }

            # Create grids
            grid_files = create_grids(
                slide_images,
                cols,
                thumbnail_width,
                output_path,
                placeholder_regions,
                slide_dimensions,
            )

        return {
            "success": True,
            "grid_files": grid_files,
            "slide_count": len(slide_images),
            "message": f"Created {len(grid_files)} grid(s) from {len(slide_images)} slides",
        }
    except Exception as e:
        return {
            "success": False,
            "grid_files": [],
            "slide_count": 0,
            "message": f"Error creating thumbnails: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pptx_unpack(
    pptx_path: str,
    output_dir: str
) -> dict[str, Any]:
    """
    Unpack a PowerPoint file to its constituent XML files.

    Extracts the Office Open XML contents of a PPTX file to a directory,
    with XML files pretty-printed for easier editing. This allows direct
    manipulation of slide content, relationships, and other internal structures.

    Args:
        pptx_path: Path to the input PowerPoint file
        output_dir: Directory where extracted contents will be saved

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_dir: Path to the output directory
        - file_count: Number of files extracted
        - message: Human-readable result description
    """
    try:
        import zipfile
        import defusedxml.minidom

        input_path = Path(pptx_path)
        if not input_path.exists():
            return {
                "success": False,
                "output_dir": output_dir,
                "file_count": 0,
                "message": f"Input file not found: {pptx_path}",
            }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract zip contents
        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(output_path)

        # Pretty print all XML files
        xml_files = list(output_path.rglob("*.xml")) + list(output_path.rglob("*.rels"))
        for xml_file in xml_files:
            try:
                content = xml_file.read_text(encoding="utf-8")
                dom = defusedxml.minidom.parseString(content)
                xml_file.write_bytes(dom.toprettyxml(indent="  ", encoding="ascii"))
            except Exception:
                # Skip files that can't be parsed as XML
                pass

        file_count = sum(1 for _ in output_path.rglob("*") if _.is_file())

        return {
            "success": True,
            "output_dir": output_dir,
            "file_count": file_count,
            "message": f"Unpacked {file_count} files to {output_dir}",
        }
    except Exception as e:
        return {
            "success": False,
            "output_dir": output_dir,
            "file_count": 0,
            "message": f"Error unpacking file: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pptx_pack(
    input_dir: str,
    output_pptx_path: str,
    validate: bool = True
) -> dict[str, Any]:
    """
    Pack an unpacked directory back into a PowerPoint file.

    Repackages the XML contents of an unpacked PPTX directory back into
    a valid PowerPoint file. XML pretty-printing whitespace is automatically
    removed during packing.

    Args:
        input_dir: Path to the unpacked PowerPoint directory
        output_pptx_path: Path where the packed PowerPoint will be saved
        validate: If True, validate the output with LibreOffice (default: True)

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_path: Path to the created PowerPoint file
        - validated: Whether validation was performed and passed
        - message: Human-readable result description
    """
    try:
        from pack import pack_document

        input_path = Path(input_dir)
        if not input_path.is_dir():
            return {
                "success": False,
                "output_path": output_pptx_path,
                "validated": False,
                "message": f"Input directory not found: {input_dir}",
            }

        output_path = Path(output_pptx_path)
        if not output_path.suffix.lower() == ".pptx":
            return {
                "success": False,
                "output_path": output_pptx_path,
                "validated": False,
                "message": "Output must be a PowerPoint file (.pptx)",
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = pack_document(str(input_path), str(output_path), validate=validate)

        if not success:
            return {
                "success": False,
                "output_path": output_pptx_path,
                "validated": validate,
                "message": "Validation failed - contents would produce a corrupt file",
            }

        return {
            "success": True,
            "output_path": output_pptx_path,
            "validated": validate,
            "message": f"Successfully packed to {output_pptx_path}" + (" (validated)" if validate else " (not validated)"),
        }
    except Exception as e:
        return {
            "success": False,
            "output_path": output_pptx_path,
            "validated": False,
            "message": f"Error packing file: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pptx_validate(
    unpacked_dir: str,
    original_pptx_path: str
) -> dict[str, Any]:
    """
    Validate the XML structure of an unpacked PowerPoint directory.

    Performs comprehensive validation of Office XML structure including:
    - XML well-formedness
    - Namespace declarations
    - Unique ID validation
    - Relationship and file reference validation
    - Slide layout ID validation
    - Content type declarations
    - XSD schema validation

    Args:
        unpacked_dir: Path to the unpacked PowerPoint directory
        original_pptx_path: Path to the original PowerPoint file (for reference)

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - is_valid: True if all validations passed
        - errors: List of validation error messages (if any)
        - message: Human-readable result description
    """
    try:
        from validation import PPTXSchemaValidator

        unpacked_path = Path(unpacked_dir)
        if not unpacked_path.is_dir():
            return {
                "success": False,
                "is_valid": False,
                "errors": [f"Unpacked directory not found: {unpacked_dir}"],
                "message": f"Unpacked directory not found: {unpacked_dir}",
            }

        original_path = Path(original_pptx_path)
        if not original_path.exists():
            return {
                "success": False,
                "is_valid": False,
                "errors": [f"Original file not found: {original_pptx_path}"],
                "message": f"Original file not found: {original_pptx_path}",
            }

        # Capture validation output
        import io
        import contextlib

        errors = []
        stdout_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture):
            validator = PPTXSchemaValidator(unpacked_path, original_path, verbose=True)
            is_valid = validator.validate()

        # Parse captured output for errors
        output = stdout_capture.getvalue()
        for line in output.split("\n"):
            if line.strip() and ("FAILED" in line or "Error" in line):
                errors.append(line.strip())

        if is_valid:
            message = "All validations passed"
        else:
            message = f"Validation failed with {len(errors)} error(s)"

        return {
            "success": True,
            "is_valid": is_valid,
            "errors": errors,
            "message": message,
        }
    except ImportError as e:
        return {
            "success": False,
            "is_valid": False,
            "errors": [f"Validation module not available: {str(e)}"],
            "message": f"Validation module not available: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "is_valid": False,
            "errors": [str(e)],
            "message": f"Error during validation: {str(e)}",
        }


if __name__ == "__main__":
    mcp.run()
