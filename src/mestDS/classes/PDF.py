import tempfile
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from mestDS.classes.LossMetrics import LossMetrics


class PDF(FPDF):
    def add_header(self, text):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, text, new_x=XPos.LEFT, new_y=YPos.NEXT, align="C")
        self.ln(5)

    def add_subheader(self, text):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, text, ln=True)
        self.ln(2)

    def add_table(self, metrics: list[LossMetrics]):
        self.set_font("Helvetica", "B", 12)

        # Adjust column widths to fit landscape layout (297mm - margins)
        col_width = 30
        headers = ["Region", "MSE", "POCID", "Thiels U"]
        for i, header in enumerate(headers):
            self.cell(col_width, 10, header, border=1, align="C")
        self.ln()

        self.set_font("Helvetica", "", 12)
        for row in metrics:
            self.cell(col_width, 10, str(row.location_name), border=1)
            self.cell(col_width, 10, str(row.mse), border=1)
            self.cell(col_width, 10, str(row.pocid), border=1)
            self.cell(col_width, 10, str(row.thiels_u), border=1)
            self.ln()

    def add_plot(self, figure):
        self.ln()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            figure.savefig(tmpfile.name, format="png", bbox_inches="tight")

            # Calculate center position based on page width (landscape A4 is 297mm wide)
            page_width = self.w - 2 * self.l_margin
            image_width = 200  # You can adjust this value
            x_position = (page_width - image_width) / 2 + self.l_margin

            self.image(tmpfile.name, x=x_position, w=image_width)

    def add_subheader_table_and_plot(self, result):
        import textwrap
        import tempfile

        location_metrics = list(result.location_metrics)

        # Subheader
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, f"Simulation: {result.name}", ln=True)
        self.ln(2)

        # Description (wrapped)
        if result.description is not None:
            self.set_font("Helvetica", "", 12)
            self.multi_cell(0, 8, f"Description: {result.description}")
            self.ln(2)

        # Ensure we're at the right Y position before adding table
        self.set_font("Helvetica", "B", 12)
        self.cell(40, 10, "Location", border=1, align="C")
        for key, _ in location_metrics[0].metrics.items():
            self.cell(40, 10, str(key), border=1, align="C")
        self.set_font("Helvetica", "", 12)
        for metric in location_metrics:
            self.ln()
            self.cell(40, 10, str(metric.location), border=1)
            for _, value in metric.metrics.items():
                self.cell(40, 10, str(value), border=1)
        self.ln(10)

        # Get Y position after table and add padding
        y_after_table = self.get_y()
        padding_mm = 5  # 5 mm padding between table and plot
        self.set_y(y_after_table + padding_mm)

        # Calculate scaled figure size to fit page width
        fig_width_mm = 11.7 * 25.4  # ~297 mm
        fig_height_mm = 6.5 * 25.4  # ~165 mm

        page_width_mm = self.w - 2 * self.l_margin
        scale_factor = 0.9  # scale to 90% of page width
        image_width_mm = page_width_mm * scale_factor
        aspect_ratio = fig_height_mm / fig_width_mm
        image_height_mm = image_width_mm * aspect_ratio

        # Place plot image centered horizontally
        for plot in result.plots:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plot.savefig(tmpfile.name, format="png", bbox_inches="tight")
                plot.close()
                x_position_mm = (page_width_mm - image_width_mm) / 2 + self.l_margin
                self.image(tmpfile.name, x=x_position_mm, w=image_width_mm)

        self.add_page()

    def ensure_space(self, required_height: float):
        remaining_height = self.h - self.get_y() - self.b_margin
        if remaining_height < required_height:
            self.add_page()
