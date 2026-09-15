#!/usr/bin/env python
"""
Convert EnergyPulse IEEE paper from Markdown to Word (.docx) format
with proper IEEE conference formatting.
"""

import re
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Read the markdown file
with open("EnergyPulse_IEEE_Conference_Paper.md", "r", encoding="utf-8") as f:
    content = f.read()

# Create a new Word document
doc = Document()

# Set up document margins
sections = doc.sections
for section in sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

# Parse and add content
lines = content.split("\n")
i = 0

while i < len(lines):
    line = lines[i].strip()
    
    # Skip empty lines
    if not line:
        i += 1
        continue
    
    # Title (# Heading)
    if line.startswith("# "):
        title_text = line[2:].strip()
        p = doc.add_paragraph(title_text, style='Heading 1')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.size = Pt(14)
            run.font.bold = True
        i += 1
        continue
    
    # Author/Affiliation
    if line.startswith("**[") and "Institution" in line:
        p = doc.add_paragraph(line.replace("**", ""), style='Normal')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.size = Pt(11)
        i += 1
        continue
    
    # Section headings (##)
    if line.startswith("## "):
        section_text = line[3:].strip()
        p = doc.add_paragraph(section_text, style='Heading 1')
        for run in p.runs:
            run.font.size = Pt(13)
            run.font.bold = True
        i += 1
        continue
    
    # Subsection headings (###)
    if line.startswith("### "):
        subsection_text = line[4:].strip()
        p = doc.add_paragraph(subsection_text, style='Heading 2')
        for run in p.runs:
            run.font.size = Pt(12)
            run.font.bold = True
        i += 1
        continue
    
    # Subsubsection headings (####)
    if line.startswith("#### "):
        subsubsection_text = line[5:].strip()
        p = doc.add_paragraph(subsubsection_text, style='Heading 3')
        for run in p.runs:
            run.font.size = Pt(11)
            run.font.bold = True
        i += 1
        continue
    
    # Code blocks (```...```)
    if line.startswith("```"):
        code_lines = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("```"):
            code_lines.append(lines[i])
            i += 1
        if code_lines:
            code_text = "\n".join(code_lines).strip()
            p = doc.add_paragraph(code_text, style='Intense Quote')
            for run in p.runs:
                run.font.name = 'Courier New'
                run.font.size = Pt(10)
        i += 1
        continue
    
    # Bullet points
    if line.startswith("- "):
        bullet_text = line[2:].strip()
        p = doc.add_paragraph(bullet_text, style='List Bullet')
        for run in p.runs:
            run.font.size = Pt(11)
        i += 1
        continue
    
    # Numbered lists
    if re.match(r"^\d+\.\s", line):
        list_text = re.sub(r"^\d+\.\s", "", line)
        p = doc.add_paragraph(list_text, style='List Number')
        for run in p.runs:
            run.font.size = Pt(11)
        i += 1
        continue
    
    # Regular paragraphs
    if line:
        # Clean markdown formatting
        clean_text = line.replace("**", "").replace("*", "").replace("[CITATION NEEDED]", "[CITATION NEEDED]")
        p = doc.add_paragraph(clean_text, style='Normal')
        for run in p.runs:
            run.font.size = Pt(11)
    
    i += 1

# Save the document
output_file = "EnergyPulse_IEEE_Conference_Paper.docx"
doc.save(output_file)
print(f"✓ Word document created successfully: {output_file}")
