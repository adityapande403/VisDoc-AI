from pathlib import Path
from typing import List, Tuple, Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from visdoc.dataset.schema import RetrievalExample
from visdoc.utils.logging import logger

class SyntheticDocumentGenerator:
    """
    Generates rich, high-fidelity synthetic PDFs covering engineering datasheets,
    schematics, financial statements, multi-column papers, and stamped forms.
    Guarantees complete reproducibility and offline testing capability.
    """

    def __init__(self, output_dir: str | Path = "d:/Aditya/hf6/data/documents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()

    def generate_all(self) -> Tuple[List[Path], List[RetrievalExample]]:
        """Generates all document categories and their ground truth retrieval queries."""
        doc_paths: List[Path] = []
        all_queries: List[RetrievalExample] = []

        # 1. Engineering Datasheet & Pinout Table
        eng_pdf, eng_queries = self._create_engineering_datasheet()
        doc_paths.append(eng_pdf)
        all_queries.extend(eng_queries)

        # 2. Financial Balance Sheet & Multi-Column Table
        fin_pdf, fin_queries = self._create_financial_statement()
        doc_paths.append(fin_pdf)
        all_queries.extend(fin_queries)

        # 3. Commercial Invoice & Purchase Order
        inv_pdf, inv_queries = self._create_commercial_invoice()
        doc_paths.append(inv_pdf)
        all_queries.extend(inv_queries)

        # 4. Scientific Two-Column Research Paper
        sci_pdf, sci_queries = self._create_scientific_paper()
        doc_paths.append(sci_pdf)
        all_queries.extend(sci_queries)

        # 5. Stamped Administrative Form
        form_pdf, form_queries = self._create_stamped_form()
        doc_paths.append(form_pdf)
        all_queries.extend(form_queries)

        logger.info(f"Generated {len(doc_paths)} synthetic PDF documents with {len(all_queries)} ground truth queries.")
        return doc_paths, all_queries

    def _create_engineering_datasheet(self) -> Tuple[Path, List[RetrievalExample]]:
        filepath = self.output_dir / "doc_eng_datasheet_001.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        story = []

        # Page 1: Overview & Pinout Diagram
        title_style = ParagraphStyle("EngTitle", parent=self.styles["Heading1"], fontSize=18, textColor=colors.navy)
        story.append(Paragraph("MicroSens TX-4000: Multi-Axis Inertial Sensor Datasheet", title_style))
        story.append(Paragraph("Document Ref: DS-4000-REV3 | Revision Date: October 2026", self.styles["Normal"]))
        story.append(Spacer(1, 15))

        story.append(Paragraph("1. Pinout Configuration & Terminal Functions", self.styles["Heading2"]))
        pinout_data = [
            ["Pin #", "Pin Name", "Signal Type", "Voltage Level", "Description"],
            ["1", "VDD", "Power", "3.3 V DC", "Main positive supply input for internal logic"],
            ["2", "GND", "Power", "0.0 V", "Ground reference for digital and analog circuits"],
            ["3", "SCL", "Digital Input", "3.3 V I2C", "I2C Serial Clock bus line with internal pullup"],
            ["4", "SDA", "Digital I/O", "3.3 V I2C", "I2C Serial Data bus bidirectional line"],
            ["5", "INT1", "Digital Output", "3.3 V Push-Pull", "Interrupt 1 output for threshold alarm events"],
            ["6", "V_SENS", "Power Input", "5.0 V Analog", "Dedicated analog sensor rail for high-precision ADC"],
            ["7", "SYNC", "Digital Input", "3.3 V", "External hardware sample synchronization pin"],
            ["8", "RESET_N", "Digital Input", "3.3 V Active-Low", "Master reset line: pull low to reset core"],
        ]
        t = Table(pinout_data, colWidths=[40, 70, 80, 80, 240])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.navy),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(t)
        story.append(Spacer(1, 15))

        # Circuit schematic block diagram
        d = Drawing(400, 100)
        d.add(Rect(50, 20, 120, 60, fillColor=colors.lightblue, strokeColor=colors.blue))
        d.add(String(70, 50, "TX-4000 IC", fontSize=12, fontName="Helvetica-Bold"))
        d.add(Line(170, 50, 240, 50, strokeColor=colors.black, strokeWidth=2))
        d.add(Rect(240, 30, 80, 40, fillColor=colors.lightgrey, strokeColor=colors.black))
        d.add(String(250, 45, "MCU Host", fontSize=10))
        d.add(String(180, 55, "I2C Bus", fontSize=9, fillColor=colors.darkred))
        story.append(d)

        # Page 2: Electrical Characteristics & Thermal Limits
        story.append(PageBreak())
        story.append(Paragraph("2. Absolute Maximum Ratings & Operating Conditions", self.styles["Heading2"]))
        elec_data = [
            ["Parameter", "Symbol", "Min", "Typ", "Max", "Unit"],
            ["Supply Voltage (VDD)", "VDD_MAX", "-0.3", "3.3", "4.0", "V"],
            ["Sensor Rail Voltage", "VSENS_MAX", "-0.3", "5.0", "5.5", "V"],
            ["Operating Temperature", "TOPER", "-40", "+25", "+105", "°C"],
            ["Storage Temperature", "TSTG", "-55", "--", "+150", "°C"],
            ["I2C Clock Frequency", "FSCL", "10", "400", "1000", "kHz"],
            ["Standby Current", "IDD_STBY", "--", "2.5", "8.0", "µA"],
            ["Active Current Consumption", "IDD_ACTIVE", "--", "1.2", "2.8", "mA"],
        ]
        t2 = Table(elec_data, colWidths=[140, 70, 50, 50, 50, 60])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkslategray),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(t2)

        doc.build(story)

        queries = [
            RetrievalExample(
                query_id="q_eng_01",
                query="Which pin provides the 5.0 V analog sensor supply on the TX-4000?",
                positive_page_ids=["doc_eng_datasheet_001_p0001"],
                negative_page_ids=["doc_eng_datasheet_001_p0002"],
                query_type="pinout_table_lookup",
                source_document_id="doc_eng_datasheet_001",
                split="test",
            ),
            RetrievalExample(
                query_id="q_eng_02",
                query="What is the maximum operating temperature limit for the TX-4000 sensor?",
                positive_page_ids=["doc_eng_datasheet_001_p0002"],
                negative_page_ids=["doc_eng_datasheet_001_p0001"],
                query_type="spec_limit_lookup",
                source_document_id="doc_eng_datasheet_001",
                split="test",
            ),
            RetrievalExample(
                query_id="q_eng_03",
                query="Find the I2C serial data line pin assignment and description",
                positive_page_ids=["doc_eng_datasheet_001_p0001"],
                negative_page_ids=["doc_eng_datasheet_001_p0002"],
                query_type="pinout_table_lookup",
                source_document_id="doc_eng_datasheet_001",
                split="train",
            )
        ]
        return filepath, queries

    def _create_financial_statement(self) -> Tuple[Path, List[RetrievalExample]]:
        filepath = self.output_dir / "doc_fin_balance_sheet_002.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        story = []

        title_style = ParagraphStyle("FinTitle", parent=self.styles["Heading1"], fontSize=18, textColor=colors.darkgreen)
        story.append(Paragraph("Apex Industrial Corp: Consolidated Balance Sheet", title_style))
        story.append(Paragraph("Reporting Period Ending December 31, 2025 (in Millions of USD)", self.styles["Normal"]))
        story.append(Spacer(1, 15))

        fin_data = [
            ["Balance Sheet Line Item", "FY 2024 (USD)", "FY 2025 (USD)", "% YoY Change"],
            ["Current Assets: Cash and Cash Equivalents", "142.5", "189.2", "+32.7%"],
            ["Short-term Marketable Securities", "85.0", "94.3", "+10.9%"],
            ["Accounts Receivable, Net", "64.8", "71.5", "+10.3%"],
            ["Inventories (Finished Goods)", "112.4", "104.8", "-6.8%"],
            ["Total Current Assets", "404.7", "459.8", "+13.6%"],
            ["Property, Plant and Equipment (PPE), Net", "520.0", "565.4", "+8.7%"],
            ["Goodwill and Intangible Assets", "180.2", "175.0", "-2.9%"],
            ["Total Non-Current Assets", "700.2", "740.4", "+5.7%"],
            ["TOTAL ASSETS", "1,104.9", "1,200.2", "+8.6%"],
            ["Current Liabilities: Accounts Payable", "98.4", "102.1", "+3.8%"],
            ["Short-term Debt Obligations", "45.0", "30.0", "-33.3%"],
            ["Long-term Senior Notes", "310.0", "295.0", "-4.8%"],
            ["Total Liabilities", "453.4", "427.1", "-5.8%"],
            ["Stockholders' Equity: Retained Earnings", "651.5", "773.1", "+18.7%"],
            ["TOTAL LIABILITIES AND EQUITY", "1,104.9", "1,200.2", "+8.6%"],
        ]
        t = Table(fin_data, colWidths=[200, 100, 100, 90])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('FONTNAME', (0,5), (-1,5), 'Helvetica-Bold'),
            ('FONTNAME', (0,9), (-1,9), 'Helvetica-Bold'),
            ('FONTNAME', (0,14), (-1,14), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(t)
        doc.build(story)

        queries = [
            RetrievalExample(
                query_id="q_fin_01",
                query="What was the total current assets for Apex Industrial in FY 2025?",
                positive_page_ids=["doc_fin_balance_sheet_002_p0001"],
                negative_page_ids=["doc_eng_datasheet_001_p0001"],
                query_type="financial_table_lookup",
                source_document_id="doc_fin_balance_sheet_002",
                split="test",
            ),
            RetrievalExample(
                query_id="q_fin_02",
                query="Find the consolidated retained earnings and total stockholders equity",
                positive_page_ids=["doc_fin_balance_sheet_002_p0001"],
                negative_page_ids=["doc_eng_datasheet_001_p0002"],
                query_type="financial_table_lookup",
                source_document_id="doc_fin_balance_sheet_002",
                split="train",
            )
        ]
        return filepath, queries

    def _create_commercial_invoice(self) -> Tuple[Path, List[RetrievalExample]]:
        filepath = self.output_dir / "doc_com_invoice_003.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        story = []

        title_style = ParagraphStyle("InvTitle", parent=self.styles["Heading1"], fontSize=20, textColor=colors.darkred)
        story.append(Paragraph("INVOICE / TAX BILL", title_style))
        story.append(Spacer(1, 10))

        inv_header = [
            ["Invoice No:", "INV-2026-8891", "Date:", "15 October 2026"],
            ["Purchase Order:", "PO-90422-X", "Due Date:", "14 November 2026"],
            ["Billed To:", "Quantum Robotics LLC", "Vendor:", "OmniGear Precision Parts Corp"],
            ["Payment Terms:", "Net 30 Days", "Currency:", "USD ($)"],
        ]
        th = Table(inv_header, colWidths=[100, 150, 80, 150])
        th.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ]))
        story.append(th)
        story.append(Spacer(1, 15))

        line_items = [
            ["Item #", "Part Number", "Description", "Qty", "Unit Price ($)", "Line Total ($)"],
            ["1", "GB-702-M4", "Planetary Gearbox Ratio 25:1", "4", "450.00", "1,800.00"],
            ["2", "BLDC-48V", "Brushless DC Motor 48V 500W", "4", "220.00", "880.00"],
            ["3", "ENC-OPT-1024", "Optical Quadrature Encoder", "4", "85.00", "340.00"],
            ["4", "CBL-SHLD-5M", "Shielded Motor Power Cable 5m", "8", "35.00", "280.00"],
            ["", "", "Subtotal:", "", "", "3,300.00"],
            ["", "", "Sales Tax (8.25%):", "", "", "272.25"],
            ["", "", "Shipping & Handling:", "", "", "65.00"],
            ["", "", "TOTAL AMOUNT DUE:", "", "", "$3,637.25"],
        ]
        tl = Table(line_items, colWidths=[40, 90, 190, 40, 70, 70])
        tl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkred),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
            ('BACKGROUND', (0,-1), (-1,-1), colors.yellow),
            ('ROWBACKGROUNDS', (0,1), (-1,-5), [colors.whitesmoke, colors.white]),
        ]))
        story.append(tl)
        doc.build(story)

        queries = [
            RetrievalExample(
                query_id="q_inv_01",
                query="Retrieve the invoice showing purchase order PO-90422-X and total $3,637.25",
                positive_page_ids=["doc_com_invoice_003_p0001"],
                negative_page_ids=["doc_fin_balance_sheet_002_p0001"],
                query_type="invoice_field_lookup",
                source_document_id="doc_com_invoice_003",
                split="test",
            ),
            RetrievalExample(
                query_id="q_inv_02",
                query="What is the unit price of the planetary gearbox ratio 25:1 on invoice INV-2026-8891?",
                positive_page_ids=["doc_com_invoice_003_p0001"],
                negative_page_ids=["doc_eng_datasheet_001_p0001"],
                query_type="invoice_field_lookup",
                source_document_id="doc_com_invoice_003",
                split="train",
            )
        ]
        return filepath, queries

    def _create_scientific_paper(self) -> Tuple[Path, List[RetrievalExample]]:
        filepath = self.output_dir / "doc_sci_paper_004.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        story = []

        title_style = ParagraphStyle("SciTitle", parent=self.styles["Heading1"], fontSize=16, alignment=1)
        story.append(Paragraph("Multi-Vector Vision-Language Models for Complex Visual Document Search", title_style))
        story.append(Paragraph("Dr. E. Thorne, Dr. A. Vance — Deep Multimodal Retrieval Lab", ParagraphStyle("Author", parent=self.styles["Normal"], alignment=1)))
        story.append(Spacer(1, 10))

        abstract_style = ParagraphStyle("Abstract", parent=self.styles["Italic"], leftIndent=20, rightIndent=20)
        story.append(Paragraph("Abstract — Conventional OCR pipelines discard multi-column layout geometry, font hierarchies, and tabular alignments. In this paper, we demonstrate that visual late-interaction MaxSim embeddings preserve diagrammatic context while delivering sub-millisecond retrieval across dense industrial PDF corpora.", abstract_style))
        story.append(Spacer(1, 15))

        # Two-column layout representation via Table
        col1_text = """<b>1. Introduction</b><br/>Visual document retrieval replaces brittle textual extraction with direct patch embeddings. Documents such as circuit schematics, pinout charts, and audited financial statements communicate vital relationships through 2D spatial arrangement rather than linear token streams."""
        col2_text = """<b>2. Empirical Benchmarks</b><br/>We evaluated ColPali and ColQwen on engineering datasheets. As detailed in Table 1, visual late interaction achieved 0.892 NDCG@5, outperforming text-only BM25 (0.512) and single-vector CLIP (0.641)."""

        p1 = Paragraph(col1_text, self.styles["Normal"])
        p2 = Paragraph(col2_text, self.styles["Normal"])
        two_col_table = Table([[p1, p2]], colWidths=[240, 240])
        two_col_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
        story.append(two_col_table)
        story.append(Spacer(1, 15))

        # Benchmarks Table
        story.append(Paragraph("Table 1: Retrieval Performance on Engineering Datasheets Benchmark", self.styles["Heading3"]))
        bench_data = [
            ["Model Representation", "NDCG@5", "MRR@10", "Recall@5", "Index Size (MB)"],
            ["BM25 (Tesseract OCR)", "0.512", "0.480", "0.620", "12.4"],
            ["Dense CLIP (ViT-L/14)", "0.641", "0.615", "0.710", "48.2"],
            ["ColPali (PaliGemma-3B)", "0.865", "0.832", "0.915", "184.0"],
            ["ColQwen2.5 (3B Dynamic)", "0.892", "0.868", "0.942", "192.5"],
            ["ColQwen2.5 INT8 Quantized", "0.884", "0.860", "0.938", "96.2"],
            ["ColQwen2.5 1-Bit Binary", "0.861", "0.835", "0.910", "12.0"],
        ]
        tb = Table(bench_data, colWidths=[160, 60, 60, 60, 90])
        tb.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(tb)
        doc.build(story)

        queries = [
            RetrievalExample(
                query_id="q_sci_01",
                query="Which retrieval benchmark table compares ColQwen2.5 NDCG@5 against BM25 OCR?",
                positive_page_ids=["doc_sci_paper_004_p0001"],
                negative_page_ids=["doc_com_invoice_003_p0001"],
                query_type="scientific_chart_lookup",
                source_document_id="doc_sci_paper_004",
                split="test",
            ),
            RetrievalExample(
                query_id="q_sci_02",
                query="What was the NDCG@5 reported for 1-bit binary quantized ColQwen2.5?",
                positive_page_ids=["doc_sci_paper_004_p0001"],
                negative_page_ids=["doc_eng_datasheet_001_p0002"],
                query_type="table_value_lookup",
                source_document_id="doc_sci_paper_004",
                split="train",
            )
        ]
        return filepath, queries

    def _create_stamped_form(self) -> Tuple[Path, List[RetrievalExample]]:
        filepath = self.output_dir / "doc_form_stamped_005.pdf"
        doc = SimpleDocTemplate(str(filepath), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        story = []

        title_style = ParagraphStyle("FormTitle", parent=self.styles["Heading1"], fontSize=18, textColor=colors.darkslategray)
        story.append(Paragraph("Quality Assurance Sign-Off & Inspection Certificate", title_style))
        story.append(Spacer(1, 10))

        form_meta = [
            ["Equipment ID:", "EQ-TURBINE-X7", "Facility:", "Plant 4 - Turbomachinery Bay"],
            ["Inspector Name:", "Marcus Brody, PE", "Certificate No:", "CERT-QA-2026-9011"],
            ["Test Pressure:", "150 bar (Hydrostatic)", "Test Result:", "PASSED - ZERO LEAKS"],
        ]
        tf = Table(form_meta, colWidths=[120, 140, 100, 140])
        tf.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ]))
        story.append(tf)
        story.append(Spacer(1, 25))

        # Circular Stamp Graphic
        d = Drawing(450, 120)
        # Red circular stamp
        d.add(Circle(120, 60, 50, fillColor=colors.mistyrose, strokeColor=colors.red, strokeWidth=3))
        d.add(Circle(120, 60, 42, fillColor=None, strokeColor=colors.red, strokeWidth=1))
        d.add(String(85, 65, "OFFICIALLY", fontSize=10, fontName="Helvetica-Bold", fillColor=colors.red))
        d.add(String(82, 50, "APPROVED", fontSize=14, fontName="Helvetica-Bold", fillColor=colors.red))
        d.add(String(88, 35, "24 OCT 2026", fontSize=8, fontName="Helvetica-Bold", fillColor=colors.red))

        # Blue signature block
        d.add(Rect(240, 20, 180, 80, fillColor=colors.aliceblue, strokeColor=colors.blue))
        d.add(String(250, 80, "Chief Inspector Signature:", fontSize=10, fontName="Helvetica-Bold"))
        d.add(Line(250, 45, 400, 45, strokeColor=colors.darkblue, strokeWidth=1.5))
        d.add(String(260, 50, "M. Brody (Digitally Signed)", fontSize=11, fontName="Times-Italic", fillColor=colors.darkblue))
        d.add(String(250, 30, "Badge ID: PE-88092", fontSize=8, fillColor=colors.black))
        story.append(d)

        doc.build(story)

        queries = [
            RetrievalExample(
                query_id="q_form_01",
                query="Find the turbine inspection certificate showing the red APPROVED stamp and zero leaks result",
                positive_page_ids=["doc_form_stamped_005_p0001"],
                negative_page_ids=["doc_com_invoice_003_p0001"],
                query_type="stamp_annotation_lookup",
                source_document_id="doc_form_stamped_005",
                split="test",
            ),
            RetrievalExample(
                query_id="q_form_02",
                query="Who is the chief inspector on QA certificate CERT-QA-2026-9011?",
                positive_page_ids=["doc_form_stamped_005_p0001"],
                negative_page_ids=["doc_sci_paper_004_p0001"],
                query_type="factual_lookup",
                source_document_id="doc_form_stamped_005",
                split="train",
            )
        ]
        return filepath, queries
