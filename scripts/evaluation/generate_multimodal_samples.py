"""Generate the synthetic, clearly labelled multimodal PDF fixture (financial_report_qa)."""
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from scripts.rag.multimodal_rag import DOMAINS, ROOT

SAMPLES = {
    "financial_report_qa": ("Quarterly revenue", "Revenue (million USD)", [120, 150, 180, 210], "Revenue increased from 120 to 210 million USD."),
}


def generate(root=ROOT / "data/samples/multimodal"):
    outputs = []
    for domain in DOMAINS:
        title, metric, values, trend = SAMPLES[domain]
        folder = Path(root) / domain
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{domain}_visual_sample.pdf"
        pdf = canvas.Canvas(str(path), pagesize=(612, 792), invariant=1)
        pdf.setTitle(title + " - synthetic RAG sample")

        def heading(section, page):
            pdf.setFillColor(colors.HexColor("#15354D"))
            pdf.setFont("Helvetica-Bold", 21)
            pdf.drawString(48, 735, title)
            pdf.setFont("Helvetica", 12)
            pdf.drawString(48, 705, section)
            pdf.setFillColor(colors.black)
            pdf.setFont("Helvetica", 10)
            pdf.drawString(48, 35, f"SYNTHETIC DEMO DATA - {domain} - Page {page}")

        heading("Overview and scope", 1)
        pdf.setFont("Helvetica", 12)
        for i, line in enumerate(["This fictional report tests local multimodal retrieval.",
                                  "All figures are invented; they are not operational policy or advice.",
                                  f"Metric: {metric}", trend,
                                  "Page 2 contains a table; page 3 has bar and line charts.",
                                  "Page 4 contains a raster process diagram."]):
            pdf.drawString(48, 650 - 27 * i, line)
        pdf.showPage()
        heading("Quarterly results table", 2)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(60, 645, "Quarter")
        pdf.drawString(220, 645, metric)
        for row in range(6):
            pdf.line(48, 670 - row * 45, 564, 670 - row * 45)
        for x in [48, 200, 564]:
            pdf.line(x, 670, x, 445)
        pdf.setFont("Helvetica", 12)
        for i, value in enumerate(values):
            pdf.drawString(60, 600 - i * 45, f"Q{i+1}")
            pdf.drawString(220, 600 - i * 45, str(value))
        pdf.drawString(48, 400, trend)
        pdf.showPage()
        heading("Bar chart and trend graph", 3)
        pdf.setFont("Helvetica", 11)
        pdf.drawString(48, 665, metric)
        maximum = max(values)
        pdf.setStrokeColor(colors.HexColor("#216B91"))
        for baseline, line_chart in [(430, False), (130, True)]:
            pdf.line(80, baseline, 540, baseline)
            pdf.line(80, baseline, 80, baseline + 200)
            last = None
            for i, value in enumerate(values):
                x, y = 140 + i*110, baseline + value/maximum*170
                pdf.setFillColor(colors.HexColor("#287DA3"))
                if line_chart:
                    pdf.circle(x, y, 4, fill=1)
                    if last:
                        pdf.line(*last, x, y)
                    last = (x, y)
                else:
                    pdf.rect(x-22, baseline, 44, y-baseline, fill=1, stroke=0)
                pdf.setFillColor(colors.black)
                pdf.drawCentredString(x, y+13, str(value))
                pdf.drawCentredString(x, baseline-18, f"Q{i+1}")
            pdf.drawString(52, baseline, "0")
        pdf.drawString(48, 80, trend)
        pdf.showPage()
        heading("Raster image: review process", 4)
        image = Image.new("RGB", (1000, 380), "white")
        draw = ImageDraw.Draw(image)
        for i, label in enumerate(["Collect", "Review", "Approve"]):
            x = 35 + i * 330
            draw.rounded_rectangle((x, 100, x+260, 260), radius=20, fill="#DAEAF2", outline="#15354D", width=4)
            draw.text((x+55, 160), label, fill="#15354D", font_size=30)
            if i < 2:
                draw.line((x+265, 180, x+320, 180), fill="#15354D", width=6)
                draw.polygon([(x+320,180), (x+300,168), (x+300,192)], fill="#15354D")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        pdf.drawImage(ImageReader(buffer), 48, 385, width=516, height=196)
        pdf.setFont("Helvetica", 12)
        pdf.drawString(48, 330, "Figure: collect evidence, review the results, then approve.")
        pdf.save()
        outputs.append(path)
    return outputs


if __name__ == "__main__":
    for output in generate():
        print(output)
