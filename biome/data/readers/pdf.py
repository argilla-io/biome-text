import io
import logging
import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pdfminer
from bs4 import BeautifulSoup, Tag
from pandas import DataFrame
from pdfminer.converter import XMLConverter, TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

__FEATURE_HTML_PARSER = 'html.parser'

__logger = logging.getLogger(__name__)
logging.getLogger(pdfminer.__name__).setLevel(logging.WARNING)  # disable bullshit pdfminer logging

__SIZE_FIELD = 'size'
__FONT_FIELD = 'font'
__BBOX_FIELD = 'bbox'
__COLOURSPACE_FIELD = 'colourspace'
__NCOLOUR_FIELD = 'ncolour'

__HIGHLIGHT_SPAN = 'highlight'


class Span(dict):
    def __init__(self, start: int, end: int, type: str, value: str):
        self.start = start
        self.end = end
        self.type = type
        self.value = value

        super().__init__(vars(self))


class DocumentBlock(dict):
    def __init__(self,
                 resource: str,
                 page: int,
                 sequence: int,
                 text: str,
                 highlights: List[Span],
                 metadata: Dict[str, Any],
                 parent, children):
        self.resource = resource
        self.page = page
        self.sequence = sequence
        self.text = text
        self.highlights = highlights
        self.metadata = metadata
        self.parent = parent
        self.children = children
        self.id = '{}_{}'.format(resource, sequence)

        super().__init__(vars(self))

    def __str__(self, level: int = 0):
        ret = '{}|{}|highlights:{}\n'.format("\t" * level, repr(self.text), self.highlights)
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<Section repr>'


class Bbox(dict):

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        super().__init__(vars(self))


def read_blocks(path: str, accepted_font_size_error=0.5, **kwargs) -> List[DocumentBlock]:
    def has_same_parent_tag(current_tag: Tag, last_tag: Tag) -> bool:
        return current_tag.parent == last_tag.parent

    def is_same_section(current_tag: Tag, last_tags: List[Tag], error: float) -> bool:
        return __is_similar_font_size(current_tag, last_tags[-1], error) == 0 \
               or has_same_parent_tag(current_tag, last_tags[-1])

    def is_inner_section(current_tag: Tag, last_tag: Tag, error: float) -> bool:
        # TODO check tags bboxes
        current_bbox = __get_tag_bbox_from_attrs(current_tag.attrs)
        last_bbox = __get_tag_bbox_from_attrs(last_tag.attrs)

        return __is_similar_font_size(current_tag, last_tag, error) < 0

    def has_text(tag: Tag) -> bool:
        # TODO inprove in order to detect real text_tag (not symbols or indentations...)
        return tag.text.isalnum()

    def get_lastest_n_tags(tags: List[Tag], n: int = 1) -> List[Tag]:
        return tags[-n:] if tags else []

    text_tags = document_as_dataframe(path, **kwargs)

    reference_font_attributes = __document_font_regular_attributes(text_tags)

    parent: DocumentBlock = None
    sections: List[DocumentBlock] = []
    first = text_tags.iloc[0]
    section_tags: List[Tag] = [first['tag']]

    for _, data in text_tags.iterrows():
        current_tag = data['tag']
        page_id = data['page']

        last_tags = get_lastest_n_tags(section_tags, n=2)
        last_tag = last_tags[-1]
        if not has_text(current_tag):
            # Normalize text_tag attributes
            current_tag.attrs = last_tag.attrs
        if is_same_section(current_tag, last_tags, accepted_font_size_error):
            section_tags.append(current_tag)
        else:
            parent = __find_parent_for_tag(parent, current_tag)

            section = __map_section_from_tags(
                resource=path,
                page=page_id,
                sequence=len(sections),
                section_tags=section_tags,
                parent_section=parent,
                regular_document_font_attributes=reference_font_attributes
            )
            sections.append(section)

            parent.children.append(section) if parent else _
            if is_inner_section(current_tag, last_tag, accepted_font_size_error):
                parent = section

            section_tags: List[Tag] = [current_tag]

    return sections


def document_as_dataframe(path: str, **kwargs) -> pd.DataFrame:
    def safety_str_2_float(value: str) -> Optional[float]:
        try:
            return float(value)
        except:
            return None

    def map_2_tag(page_id: int, html_tag: Tag) -> Dict[str, Any]:
        attrs = deepcopy(html_tag.attrs)

        size = safety_str_2_float(attrs.get(__SIZE_FIELD))
        ncolour = safety_str_2_float(attrs.get(__NCOLOUR_FIELD)) or 0
        font = attrs.get(__FONT_FIELD)
        colourspace = attrs.get(__COLOURSPACE_FIELD)
        return dict(
            page=page_id,
            tag=html_tag,
            bbox=__get_tag_bbox_from_attrs(attrs),
            size=size,
            font=font,
            ncolour=ncolour,
            colourspace=colourspace
        )

    parsed_xml = read_pdf_as_type(path, converter_type='xml', **kwargs)
    soup = BeautifulSoup(parsed_xml, __FEATURE_HTML_PARSER)
    text_tags: List[Tuple[int, Tag]] = [
        map_2_tag(page_id, text_tag)
        for page_id, page in enumerate(soup.find_all('page'), 1)
        for textbox in page.find_all('textbox')
        for text_tag in textbox.find_all('text')
    ]
    return pd.DataFrame(text_tags)


def __document_font_regular_attributes(text_tags: DataFrame) -> Dict[str, str]:
    top_font_attrs = text_tags \
        .groupby([__SIZE_FIELD, __FONT_FIELD, __NCOLOUR_FIELD, __COLOURSPACE_FIELD]) \
        .count() \
        .sort_values('tag', ascending=False) \
        .index.values[0]

    return {
        __SIZE_FIELD: top_font_attrs[0],
        __FONT_FIELD: top_font_attrs[1],
        __NCOLOUR_FIELD: top_font_attrs[2],
        __COLOURSPACE_FIELD: top_font_attrs[3]
    }


def read_text(path: str, pages: List[int] = None) -> bytes:
    return read_pdf_as_type(path, converter_type='text', pages=pages)


def read_pdf_as_type(fname,
                     converter_type: str = 'xml',
                     encoding: str = 'utf-8',
                     pages=None) -> bytes:
    def close_stream(stream) -> None:
        _ = stream.close if stream else None

    infile = converter = output = None
    try:
        if not pages:
            pagenums = set()
        else:
            pagenums = set(pages)

        manager = PDFResourceManager()
        output = io.BytesIO()
        caching = True

        converter_params = dict(
            rsrcmgr=manager,
            outfp=output,
            codec=encoding,
            laparams=pdfminer.layout.LAParams(),
            # imagewriter=ImageWriter(None),
        )

        converter = TextConverter(**converter_params) \
            if converter_type == 'text' \
            else XMLConverter(**converter_params, stripcontrol=True)

        interpreter = PDFPageInterpreter(manager, converter)
        infile = open(fname, 'rb')

        for page in PDFPage.get_pages(infile, pagenums, caching=caching, check_extractable=True):
            interpreter.process_page(page)

        parsed_pdf = output.getvalue()
        return parsed_pdf
    finally:
        close_stream(infile)
        close_stream(converter)
        close_stream(output)


def __get_tag_size_from_attrs(attrs: Dict[str, str]) -> float:
    return float(attrs.get('size', 0))


def __get_tag_font_from_attrs(attrs: Dict[str, str]) -> Optional[str]:
    return attrs.get(__FONT_FIELD)


def __get_tag_ncolour_from_attrs(attrs: Dict[str, str]) -> Optional[str]:
    return attrs.get(__NCOLOUR_FIELD)


def __get_tag_colourspace_from_attrs(attrs: Dict[str, str]) -> Optional[str]:
    return attrs.get(__COLOURSPACE_FIELD)


def __get_tag_bbox_from_attrs(attrs: Dict[str, str]) -> Optional[Bbox]:
    bbox = attrs.get(__BBOX_FIELD, [])
    return Bbox(*[float(coordinate) for coordinate in bbox.split(',')]) if bbox else None


def __is_similar_font_size(tag: Tag, other_tag: Tag, delta_error: float) -> int:
    epsilon = __get_tag_size_from_attrs(tag.attrs) - __get_tag_size_from_attrs(other_tag.attrs)
    return 0 if math.fabs(epsilon) <= delta_error \
        else -1 if epsilon < 0 \
        else 1


def __is_similar_font(tag: Tag, other_tag: Tag):
    return __get_tag_font_from_attrs(other_tag.attrs) == __get_tag_font_from_attrs(tag.attrs)


def __map_section_from_tags(resource: str,
                            page: int,
                            sequence: int,
                            section_tags: List[Tag],
                            parent_section: Optional[DocumentBlock],
                            regular_document_font_attributes: Dict[str, str]) -> DocumentBlock:
    def section_spans(tags: List[Tag], regular_document_font_attributes: Dict[str, str]):
        def similar_attributes(tag: Tag, refs: Dict[str, str]) -> bool:
            for k, v in refs.items():
                if v != tag.attrs.get(k):
                    return False
            return True

        start = end = None
        spans: List[Span] = []
        for idx, tag in enumerate(tags):
            if start is not None \
                    and similar_attributes(tag,
                                           {k: tags[start].attrs[k] for k in regular_document_font_attributes.keys()}):
                end = idx
            if start is not None \
                    and end is not None \
                    and (end != idx or len(tags) - 1 == idx):
                spans.append(Span(
                    start=start,
                    end=end,
                    type=__HIGHLIGHT_SPAN,
                    value=''.join([token.text for token in tags[start:end]])
                ))
                start = end = None
            if start is None and not similar_attributes(tag, regular_document_font_attributes):
                start = idx

        return spans

    def section_bbox(tags: List[Tag]) -> Bbox:
        char_bboxes = pd.DataFrame([__get_tag_bbox_from_attrs(tag.attrs) for tag in tags])
        return Bbox(
            char_bboxes['x1'].min(), char_bboxes['y1'].min(),
            char_bboxes['x2'].max(), char_bboxes['y2'].max()
        )

    return DocumentBlock(
        resource=resource,
        page=page,
        sequence=sequence,
        text=''.join([tag.text for tag in section_tags]),
        highlights=section_spans(section_tags, regular_document_font_attributes),
        metadata={
            **section_tags[0].attrs,
            __BBOX_FIELD: section_bbox(section_tags)
        },
        parent=parent_section,
        children=[]
    )


def __find_parent_for_tag(section: DocumentBlock, current_tag: Tag) -> Optional[DocumentBlock]:
    current_section = section

    while current_section and \
            __get_tag_size_from_attrs(current_section.metadata) <= __get_tag_size_from_attrs(current_tag.attrs):
        current_section = current_section.parent

    return current_section
