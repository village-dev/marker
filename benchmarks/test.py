import time
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
import torch
from marker.processors.code import CodeProcessor
from marker.processors.debug import DebugProcessor
from marker.processors.document_toc import DocumentTOCProcessor
from marker.processors.equation import EquationProcessor
from marker.processors.footnote import FootnoteProcessor
from marker.processors.ignoretext import IgnoreTextProcessor
from marker.processors.line_numbers import LineNumbersProcessor
from marker.processors.page_header import PageHeaderProcessor
from marker.processors.sectionheader import SectionHeaderProcessor
from marker.processors.table import TableProcessor
from marker.processors.text import TextProcessor

config = {
    "output_format": "json",
}
config_parser = ConfigParser(config)

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=create_model_dict(device="cuda:0",dtype=torch.bfloat16),
    processor_list=[
        "marker.processors.footnote.FootnoteProcessor",
        "marker.processors.page_header.PageHeaderProcessor",
        # "marker.processors.equation.EquationProcessor",
        "marker.processors.table.TableProcessor",
        "marker.processors.sectionheader.SectionHeaderProcessor",
        "marker.processors.text.TextProcessor",
        "marker.processors.code.CodeProcessor",
        "marker.processors.document_toc.DocumentTOCProcessor",
        "marker.processors.ignoretext.IgnoreTextProcessor",
        "marker.processors.line_numbers.LineNumbersProcessor",
        "marker.processors.debug.DebugProcessor",
    ],
    renderer=config_parser.get_renderer()
)

start_time = time.time()
rendered = converter("./data/test-ocr.pdf")

text, _, images = text_from_rendered(rendered)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

with open("./data/test-ocr.json", "w") as f:
    f.write(text)
