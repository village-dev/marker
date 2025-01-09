from typing import List

from tabled.formats import html_format
from tabled.schema import SpanTableCell

from marker.schema import BlockTypes
from marker.schema.blocks import Block


class Form(Block):
    block_type: str = BlockTypes.Form
    cells: List[SpanTableCell] | None = None
    html: str | None = None

    def assemble_html(self, child_blocks, parent_structure=None):
        # Some processors convert the form to html
        if self.html is not None:
            return self.html

        return str(html_format(self.cells))
