from tests.conftest import SAMPLE_PDF, SAMPLE_DOCX
from src.ingestion.extractor import extract_text_pdf, extract_text_docx


class TestExtractTextPdf:
    def test_returns_list_of_page_dicts(self):
        # Act
        pages = extract_text_pdf(SAMPLE_PDF)

        # Assert
        assert isinstance(pages, list)
        assert len(pages) >= 1
        for page in pages:
            assert "page_number" in page
            assert "text" in page
            assert isinstance(page["page_number"], int)
            assert isinstance(page["text"], str)

    def test_page_numbers_are_positive(self):
        # Act
        pages = extract_text_pdf(SAMPLE_PDF)

        # Assert
        for page in pages:
            assert page["page_number"] >= 1

    def test_no_empty_text_pages(self):
        # Act
        pages = extract_text_pdf(SAMPLE_PDF)

        # Assert
        for page in pages:
            assert page["text"].strip() != ""

    def test_extracts_expected_content(self):
        # Act
        pages = extract_text_pdf(SAMPLE_PDF)
        all_text = " ".join(p["text"] for p in pages).lower()

        # Assert
        assert "introduction" in all_text
        assert "methodology" in all_text


class TestExtractTextDocx:
    def test_returns_list_with_single_entry(self):
        # Act
        pages = extract_text_docx(SAMPLE_DOCX)

        # Assert
        assert isinstance(pages, list)
        assert len(pages) == 1

    def test_page_number_is_zero(self):
        # Act
        pages = extract_text_docx(SAMPLE_DOCX)

        # Assert
        assert pages[0]["page_number"] == 0

    def test_text_is_non_empty(self):
        # Act
        pages = extract_text_docx(SAMPLE_DOCX)

        # Assert
        assert pages[0]["text"].strip() != ""

    def test_extracts_paragraph_content(self):
        # Act
        pages = extract_text_docx(SAMPLE_DOCX)
        text = pages[0]["text"].lower()

        # Assert
        assert "introduction" in text
        assert "paragraph" in text

    def test_returns_empty_list_for_empty_docx(self, tmp_path):
        # Arrange
        from docx import Document
        empty_path = tmp_path / "empty.docx"
        Document().save(str(empty_path))

        # Act
        pages = extract_text_docx(empty_path)

        # Assert
        assert pages == []
