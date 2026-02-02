"""
PDF Processing MCP Server

Provides tools for PDF form processing, including:
- Checking for fillable form fields
- Extracting form field metadata
- Filling fillable form fields
- Converting PDFs to images
- Creating validation images with bounding boxes
- Checking bounding box intersections
- Filling non-fillable PDFs with text annotations
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Add scripts directory to path for imports
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

mcp = FastMCP("pdf-processing")


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pdf_check_fillable_fields(pdf_path: str) -> dict[str, Any]:
    """
    Check if a PDF has native fillable form fields (AcroForms).

    This tool determines whether the PDF contains interactive form fields
    that can be programmatically filled, or if it requires text annotations
    to add data.

    Args:
        pdf_path: Path to the PDF file to check

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - has_fillable_fields: True if PDF has fillable fields, False otherwise
        - message: Human-readable description of the result
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        has_fields = bool(reader.get_fields())

        if has_fields:
            message = "This PDF has fillable form fields"
        else:
            message = "This PDF does not have fillable form fields; you will need to visually determine where to enter data"

        return {
            "success": True,
            "has_fillable_fields": has_fields,
            "message": message,
        }
    except Exception as e:
        return {
            "success": False,
            "has_fillable_fields": False,
            "message": f"Error checking PDF: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pdf_extract_form_fields(pdf_path: str, output_json_path: str) -> dict[str, Any]:
    """
    Extract form field metadata from a fillable PDF to a JSON file.

    Extracts information about all fillable form fields including their IDs,
    types (text, checkbox, radio, choice), page numbers, and bounding rectangles.

    Args:
        pdf_path: Path to the input PDF file
        output_json_path: Path where the extracted field info JSON will be saved

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_path: Path to the created JSON file
        - field_count: Number of fields extracted
        - message: Human-readable result description
    """
    try:
        from extract_form_field_info import write_field_info, get_field_info
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        field_info = get_field_info(reader)

        with open(output_json_path, "w") as f:
            json.dump(field_info, f, indent=2)

        return {
            "success": True,
            "output_path": output_json_path,
            "field_count": len(field_info),
            "message": f"Wrote {len(field_info)} fields to {output_json_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "output_path": output_json_path,
            "field_count": 0,
            "message": f"Error extracting form fields: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pdf_fill_fillable_fields(
    input_pdf_path: str,
    fields_json_path: str,
    output_pdf_path: str
) -> dict[str, Any]:
    """
    Fill native fillable form fields in a PDF.

    Takes a PDF with fillable fields and a JSON file containing field values,
    then produces a new PDF with the fields filled in.

    The fields JSON should be an array of objects with:
    - field_id: The field identifier
    - page: Page number (1-indexed)
    - value: The value to fill in

    Args:
        input_pdf_path: Path to the input PDF file with fillable fields
        fields_json_path: Path to JSON file containing field values to fill
        output_pdf_path: Path where the filled PDF will be saved

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_path: Path to the created PDF file
        - message: Human-readable result description
    """
    try:
        from fill_fillable_fields import fill_pdf_fields, monkeypatch_pydpf_method

        monkeypatch_pydpf_method()
        fill_pdf_fields(input_pdf_path, fields_json_path, output_pdf_path)

        return {
            "success": True,
            "output_path": output_pdf_path,
            "message": f"Successfully filled PDF form and saved to {output_pdf_path}",
        }
    except SystemExit as e:
        return {
            "success": False,
            "output_path": output_pdf_path,
            "message": "Validation errors in field values - check field IDs, page numbers, and values",
        }
    except Exception as e:
        return {
            "success": False,
            "output_path": output_pdf_path,
            "message": f"Error filling PDF form: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pdf_convert_to_images(
    pdf_path: str,
    output_dir: str,
    max_dimension: int = 1000
) -> dict[str, Any]:
    """
    Convert PDF pages to PNG images.

    Each page of the PDF is converted to a separate PNG image, scaled to fit
    within the specified maximum dimension while maintaining aspect ratio.

    Args:
        pdf_path: Path to the input PDF file
        output_dir: Directory where PNG images will be saved (page_1.png, page_2.png, etc.)
        max_dimension: Maximum width or height in pixels (default: 1000)

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_dir: Path to the output directory
        - page_count: Number of pages converted
        - image_paths: List of paths to created images
        - message: Human-readable result description
    """
    try:
        from pdf2image import convert_from_path

        os.makedirs(output_dir, exist_ok=True)
        images = convert_from_path(pdf_path, dpi=200)

        image_paths = []
        for i, image in enumerate(images):
            width, height = image.size
            if width > max_dimension or height > max_dimension:
                scale_factor = min(max_dimension / width, max_dimension / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height))

            image_path = os.path.join(output_dir, f"page_{i+1}.png")
            image.save(image_path)
            image_paths.append(image_path)

        return {
            "success": True,
            "output_dir": output_dir,
            "page_count": len(images),
            "image_paths": image_paths,
            "message": f"Converted {len(images)} pages to PNG images in {output_dir}",
        }
    except Exception as e:
        return {
            "success": False,
            "output_dir": output_dir,
            "page_count": 0,
            "image_paths": [],
            "message": f"Error converting PDF to images: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pdf_create_validation_image(
    page_number: int,
    fields_json_path: str,
    input_image_path: str,
    output_image_path: str
) -> dict[str, Any]:
    """
    Create a validation image with bounding boxes drawn on it.

    Takes a page image and a fields.json file, draws red rectangles for entry
    bounding boxes and blue rectangles for label bounding boxes. This is used
    to visually verify that bounding boxes are positioned correctly before
    filling a non-fillable PDF.

    Args:
        page_number: The page number (1-indexed) to create validation image for
        fields_json_path: Path to the fields.json file containing bounding box info
        input_image_path: Path to the input page image (PNG)
        output_image_path: Path where the validation image will be saved

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_path: Path to the created validation image
        - box_count: Number of bounding boxes drawn
        - message: Human-readable result description
    """
    try:
        from PIL import Image, ImageDraw

        with open(fields_json_path, 'r') as f:
            data = json.load(f)

        img = Image.open(input_image_path)
        draw = ImageDraw.Draw(img)
        num_boxes = 0

        for field in data["form_fields"]:
            if field["page_number"] == page_number:
                entry_box = field['entry_bounding_box']
                label_box = field['label_bounding_box']
                draw.rectangle(entry_box, outline='red', width=2)
                draw.rectangle(label_box, outline='blue', width=2)
                num_boxes += 2

        img.save(output_image_path)

        return {
            "success": True,
            "output_path": output_image_path,
            "box_count": num_boxes,
            "message": f"Created validation image at {output_image_path} with {num_boxes} bounding boxes",
        }
    except Exception as e:
        return {
            "success": False,
            "output_path": output_image_path,
            "box_count": 0,
            "message": f"Error creating validation image: {str(e)}",
        }


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pdf_check_bounding_boxes(fields_json_path: str) -> dict[str, Any]:
    """
    Validate that bounding boxes in a fields.json file don't intersect.

    Checks the fields.json file used for annotation-based PDF filling to ensure
    that no label or entry bounding boxes overlap, which would cause rendering
    issues in the filled PDF.

    Args:
        fields_json_path: Path to the fields.json file to validate

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - is_valid: True if all bounding boxes are valid (no intersections)
        - field_count: Number of fields checked
        - messages: List of validation messages (errors if any)
    """
    try:
        from check_bounding_boxes import get_bounding_box_messages

        with open(fields_json_path) as f:
            messages = get_bounding_box_messages(f)

        is_valid = any("SUCCESS" in msg for msg in messages)
        field_count = 0
        for msg in messages:
            if msg.startswith("Read "):
                try:
                    field_count = int(msg.split()[1])
                except (ValueError, IndexError):
                    pass
                break

        return {
            "success": True,
            "is_valid": is_valid,
            "field_count": field_count,
            "messages": messages,
        }
    except Exception as e:
        return {
            "success": False,
            "is_valid": False,
            "field_count": 0,
            "messages": [f"Error checking bounding boxes: {str(e)}"],
        }


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
def pdf_fill_with_annotations(
    input_pdf_path: str,
    fields_json_path: str,
    output_pdf_path: str
) -> dict[str, Any]:
    """
    Fill a non-fillable PDF by adding text annotations.

    For PDFs that don't have native fillable form fields, this tool adds
    FreeText annotations at specified bounding box locations. The fields.json
    file should contain page info and form fields with entry_text specifications.

    The fields.json format:
    {
        "pages": [{"page_number": 1, "image_width": 800, "image_height": 1000}],
        "form_fields": [{
            "page_number": 1,
            "description": "User's name",
            "label_bounding_box": [30, 125, 95, 142],
            "entry_bounding_box": [100, 125, 280, 142],
            "entry_text": {"text": "John Doe", "font_size": 14}
        }]
    }

    Args:
        input_pdf_path: Path to the input PDF file
        fields_json_path: Path to the fields.json file with bounding boxes and text
        output_pdf_path: Path where the filled PDF will be saved

    Returns:
        A dictionary with:
        - success: Whether the operation completed successfully
        - output_path: Path to the created PDF file
        - annotation_count: Number of text annotations added
        - message: Human-readable result description
    """
    try:
        from fill_pdf_form_with_annotations import fill_pdf_form

        # Count annotations that will be added
        with open(fields_json_path, "r") as f:
            fields_data = json.load(f)

        annotation_count = sum(
            1 for field in fields_data.get("form_fields", [])
            if "entry_text" in field and field["entry_text"].get("text")
        )

        fill_pdf_form(input_pdf_path, fields_json_path, output_pdf_path)

        return {
            "success": True,
            "output_path": output_pdf_path,
            "annotation_count": annotation_count,
            "message": f"Successfully filled PDF with {annotation_count} text annotations and saved to {output_pdf_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "output_path": output_pdf_path,
            "annotation_count": 0,
            "message": f"Error filling PDF with annotations: {str(e)}",
        }


if __name__ == "__main__":
    mcp.run()
