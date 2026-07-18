"""DocumentChunker chunk-size enforcement tests.

Guards against oversized chunks inflating Voyage embedding cost: a single
sentence-boundary-regex "sentence" with no usable punctuation (HTML-to-text
extraction of SEC filing tables/legal boilerplate routinely produces this)
must not become one giant chunk on its own.
"""

from __future__ import annotations

from firm.rag.chunker import DocumentChunker


class TestChunkSizeEnforcement:
    def test_unpunctuated_text_is_hard_split(self):
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        giant = " ".join(f"word{i}" for i in range(8000))  # no punctuation at all

        docs = chunker.chunk(giant, {"source": "test"})

        assert len(docs) > 1
        for d in docs:
            assert len(d.text) // 4 <= 500

    def test_normal_punctuated_text_chunks_near_target(self):
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        normal = " ".join(
            f"This is sentence number {i} about Apple earnings." for i in range(60)
        )

        docs = chunker.chunk(normal, {"source": "test"})

        assert len(docs) >= 2
        # Sentence-boundary chunking naturally overshoots the token target
        # slightly (it only checks the limit *before* adding the next
        # sentence) — allow headroom rather than asserting an exact cap.
        for d in docs:
            assert len(d.text) // 4 <= 600

    def test_mixed_content_caps_the_oversized_part_only(self):
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        normal_lead = "A short introductory sentence. Another short one. "
        giant_run = " ".join(f"tok{i}" for i in range(4000))
        docs = chunker.chunk(normal_lead + giant_run, {"source": "test"})

        assert len(docs) > 1
        for d in docs:
            assert len(d.text) // 4 <= 500


class TestChunkBySections:
    def test_prefix_headers_are_not_shadowed(self):
        # Regression: regex alternation tries alternatives in listed order,
        # not longest-match — "Item 1|Item 1A" matched "Item 1A ..." as
        # "Item 1" plus a stray leading "A". Headers that are a prefix of
        # another header (Item 1 / 1A / 1B / 10-15, Item 7 / 7A, Item 9 / 9A
        # / 9B) must resolve to their own, full section — not the shorter one.
        chunker = DocumentChunker(chunk_size=500)
        headers = [
            "Item 1", "Item 1A", "Item 1B", "Item 7", "Item 7A",
            "Item 9", "Item 9A", "Item 10",
        ]
        text = (
            "Item 1 Business overview text here. "
            "Item 1A Risk factors discussion here. "
            "Item 10 Directors and officers here. "
            "Item 7A Market risk stuff."
        )
        docs = chunker.chunk_by_sections(text, headers, {"source": "test"})
        sections = {d.metadata["section"]: d.text for d in docs}

        assert sections["Item 1"].strip() == "Business overview text here."
        assert sections["Item 1A"].strip() == "Risk factors discussion here."
        assert sections["Item 10"].strip() == "Directors and officers here."
        assert sections["Item 7A"].strip() == "Market risk stuff."
