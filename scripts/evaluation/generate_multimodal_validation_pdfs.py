"""Three independent fixtures; chart values exist only inside raster pixels."""
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from scripts.rag.multimodal_rag import ROOT


def generate(root=ROOT / "data/samples/multimodal_validation"):
    folder = Path(root) / "financial_report_qa"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, title in [("text_operations", "Operations handbook"),
                            ("table_budget", "Regional budget"), ("chart_revenue", "Revenue chart")]:
        path = folder / f"{filename}.pdf"
        pdf = canvas.Canvas(str(path), pagesize=(612, 792), invariant=1)
        pdf.setTitle(title + " - synthetic validation fixture")
        if filename == "chart_revenue":
            # The entire page is a raster image. No PDF text or numeric metadata.
            image = Image.new("RGB", (900, 1100), "white")
            draw = ImageDraw.Draw(image)
            font_path = "C:/Windows/Fonts/arial.ttf"
            font = ImageFont.truetype(font_path, 35)
            small = ImageFont.truetype(font_path, 28)
            draw.text((75, 70), "Quarterly Revenue (million USD)", fill="black", font=font)
            draw.line((100, 900, 820, 900), fill="black", width=3)
            for i, (label, value) in enumerate(zip(["Q1", "Q2", "Q3", "Q4"], [10, 20, 30, 40])):
                x = 150 + i * 170
                y = 900 - value * 15
                draw.rectangle((x, y, x+90, 900), fill="#1769C2")
                draw.text((x+22, y-55), str(value), fill="black", font=font)
                draw.text((x+15, 930), label, fill="black", font=font)
            draw.text((75, 1030), "SYNTHETIC TEST DATA", fill="black", font=small)
            stream = BytesIO()
            image.save(stream, format="PNG")
            pdf.drawImage(ImageReader(stream), 0, 0, width=612, height=792)
        else:
            pdf.setFont("Helvetica-Bold", 23)
            pdf.drawString(50, 725, title)
            pdf.setFont("Helvetica", 14)
            if filename == "text_operations":
                for i, line in enumerate(["The annual review code is ORCHID.",
                                          "Expense reports must be submitted within 14 days.",
                                          "The operations team meets every Tuesday."]):
                    pdf.drawString(50, 650-i*40, line)
            else:
                for row, cells in enumerate([("Region", "Budget (million USD)"), ("North", "73"), ("South", "46"), ("West", "28")]):
                    y = 650-row*65
                    pdf.drawString(65, y, cells[0])
                    pdf.drawString(270, y, cells[1])
                for x in [50, 250, 560]:
                    pdf.line(x, 680, x, 420)
                for y in [680, 615, 550, 485, 420]:
                    pdf.line(50, y, 560, y)
            pdf.setFont("Helvetica", 10)
            pdf.drawString(50, 40, "SYNTHETIC VALIDATION DATA")
        pdf.save()
        paths.append(path)
    return paths


if __name__ == "__main__":
    for path in generate():
        print(path)
