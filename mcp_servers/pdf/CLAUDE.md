# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a PDF processing skill for Claude Code that provides utilities for filling PDF forms, extracting form field information, and converting PDFs to images. The skill handles two types of PDFs: those with fillable form fields (AcroForms) and those without (requiring text annotations).

## Dependencies

- **pypdf**: PDF reading/writing, form field manipulation
- **pdf2image**: PDF to PNG conversion (requires poppler)
- **Pillow (PIL)**: Image processing for validation images

## Scripts

All scripts are in `scripts/` and should be run from the skill directory:

| Script | Purpose | Usage |
|--------|---------|-------|
| `check_fillable_fields.py` | Check if PDF has fillable fields | `python scripts/check_fillable_fields.py <file.pdf>` |
| `extract_form_field_info.py` | Extract fillable field metadata to JSON | `python scripts/extract_form_field_info.py <input.pdf> <output.json>` |
| `fill_fillable_fields.py` | Fill fillable form fields | `python scripts/fill_fillable_fields.py <input.pdf> <field_values.json> <output.pdf>` |
| `convert_pdf_to_images.py` | Convert PDF pages to PNGs (max 1000px) | `python scripts/convert_pdf_to_images.py <input.pdf> <output_dir>` |
| `create_validation_image.py` | Draw bounding boxes on image for validation | `python scripts/create_validation_image.py <page_num> <fields.json> <input.png> <output.png>` |
| `check_bounding_boxes.py` | Validate bounding boxes don't intersect | `python scripts/check_bounding_boxes.py <fields.json>` |
| `fill_pdf_form_with_annotations.py` | Add text annotations to non-fillable PDFs | `python scripts/fill_pdf_form_with_annotations.py <input.pdf> <fields.json> <output.pdf>` |

## Form Filling Workflow

The workflow differs based on whether the PDF has fillable fields:

### Fillable PDFs (AcroForms)
1. `check_fillable_fields.py` → confirms fillable fields exist
2. `extract_form_field_info.py` → outputs field metadata JSON
3. Create `field_values.json` with field IDs and values
4. `fill_fillable_fields.py` → produces filled PDF

### Non-Fillable PDFs (Annotation-based)
1. `check_fillable_fields.py` → confirms no fillable fields
2. `convert_pdf_to_images.py` → create page images for analysis
3. Visually analyze images, create `fields.json` with bounding boxes
4. `create_validation_image.py` → generate validation images
5. `check_bounding_boxes.py` → verify no intersections
6. Visually verify validation images (red=entry areas, blue=labels)
7. `fill_pdf_form_with_annotations.py` → produces filled PDF

## Key Data Formats

### field_values.json (fillable PDFs)
```json
[{"field_id": "name", "page": 1, "value": "John Doe"}]
```

### fields.json (non-fillable PDFs)
```json
{
  "pages": [{"page_number": 1, "image_width": 800, "image_height": 1000}],
  "form_fields": [{
    "page_number": 1,
    "description": "User's name",
    "field_label": "Name",
    "label_bounding_box": [30, 125, 95, 142],
    "entry_bounding_box": [100, 125, 280, 142],
    "entry_text": {"text": "John Doe", "font_size": 14}
  }]
}
```

Bounding boxes use image coordinates: `[left, top, right, bottom]` with origin at top-left.

## Coordinate Systems

- **Image coordinates**: Origin at top-left, Y increases downward
- **PDF coordinates**: Origin at bottom-left, Y increases upward

The `fill_pdf_form_with_annotations.py` script handles this transformation automatically.
