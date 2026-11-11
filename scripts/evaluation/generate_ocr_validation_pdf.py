"""Native text, a scanned page, and an image region with independent OCR-only codes."""
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from scripts.rag.multimodal_rag import ROOT


def generate():
    folder = ROOT / "data/samples/ocr_validation/financial_report_qa"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "selective_ocr_sample.pdf"
    pdf = canvas.Canvas(str(target), pagesize=(612, 792), invariant=1)
    pdf.setTitle("Selective OCR validation - synthetic data")
    pdf.setFont("Helvetica-Bold", 23)
    pdf.drawString(50, 720, "Native text control")
    pdf.setFont("Helvetica", 15)
    pdf.drawString(50, 660, "Native review code is IRIS.")
    pdf.drawString(50, 625, "This page has a complete searchable text layer.")
    pdf.showPage()

    def raster(lines, height):
        image = Image.new("RGB", (1100, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 39)
        for i, line in enumerate(lines):
            draw.text((45, 60 + i*85), line, font=font, fill="black")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return ImageReader(buffer)

    pdf.drawImage(raster(["SCANNED INVOICE", "Invoice reference: SCAN7429", "Payment due within 23 days.",
                          "SYNTHETIC VALIDATION DATA"], 1424), 0, 0, width=612, height=792)
    pdf.showPage()
    pdf.setFont("Helvetica-Bold", 23)
    pdf.drawString(50, 720, "Maintenance field guide")
    pdf.setFont("Helvetica", 13)
    pdf.drawString(50, 665, "Read the embedded equipment label before scheduling service.")
    pdf.drawString(50, 635, "The paragraph is native PDF text; the equipment label is an image.")
    pdf.drawImage(raster(["Calibration access code: IMAGE8635", "Inspection interval: 47 days."], 300),
                  50, 330, width=512, height=140)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, 40, "SYNTHETIC VALIDATION DATA")
    pdf.save()
    return target


if __name__ == "__main__":
    print(generate())
