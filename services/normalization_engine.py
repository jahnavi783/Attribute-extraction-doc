
"""
Normalization Engine - parse -> match -> normalize -> write
Supports:
- PDF / unstructured key-value normalization
- Excel / CSV multi-column normalization while preserving extra columns
"""

import io
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
from services.attribute_matcher import (
    AttributeMatcher,
    MatchResult,
    DEFAULT_FUZZY_THRESHOLD,
    DEFAULT_SEMANTIC_THRESHOLD,
)
from utils.file_parsers import (
    parse_pdf,
    parse_excel,
    parse_csv,
    parse_docx,
    extract_kv_from_excel_sheet,
    extract_structured_records_from_excel_sheet,
    extract_structured_records_from_csv_df,
    extract_raw_text_from_pdf,
    extract_raw_text_from_excel,
    extract_raw_text_from_csv,
    extract_raw_text_from_docx,
    parse_unstructured_text,
    should_use_llm_for_text,
)
from utils.file_writers import (
    write_pdf_keyvalue,
    write_excel_keyvalue,
    write_csv_records,
    write_docx_keyvalue,
)


@dataclass
class NormalizationReport:
    input_format: str
    doc_type: str
    total_attributes: int
    matched: int
    unmatched: int
    match_details: list[MatchResult]
    output_bytes: bytes
    output_ext: str
    normalized_records: list[dict] = field(default_factory=list)


class NormalizationEngine:
    def __init__(self, master_path: str):
        self.matcher = AttributeMatcher(master_path)

    def _norm_attr(self, attr: str, value: str, fuzzy_threshold: float, semantic_threshold: float):
        result = self.matcher.match(
            raw_attr=attr,
            raw_value=value,
            fuzzy_threshold=fuzzy_threshold,
            semantic_threshold=semantic_threshold,
        )
        return result.canonical_attr, result

    def _normalize_records(self, records: list[dict], fuzzy_threshold: float, semantic_threshold: float):
        normalized = []
        match_results = []
        cache = {}

        for rec in records:
            raw_attr = str(rec.get("attribute", "")).strip()
            raw_value = str(rec.get("value", "")).strip()
            cache_key = (raw_attr.lower(), raw_value)

            if cache_key in cache:
                canonical, result = cache[cache_key]
            else:
                canonical, result = self._norm_attr(
                    raw_attr,
                    raw_value,
                    fuzzy_threshold,
                    semantic_threshold,
                )
                cache[cache_key] = (canonical, result)

            match_results.append(result)

            new_rec = dict(rec)
            new_rec["attribute"] = canonical
            if "value" not in new_rec:
                new_rec["value"] = rec.get("value", "")
            normalized.append(new_rec)

        return normalized, match_results

    def _finalize_report(
        self,
        input_format: str,
        doc_type: str,
        normalized_records: list[dict],
        match_results: list[MatchResult],
        output_bytes: bytes,
        output_ext: str,
    ) -> NormalizationReport:
        matched = sum(1 for r in match_results if r.match_type != "unmatched")

        return NormalizationReport(
            input_format=input_format,
            doc_type=doc_type,
            total_attributes=len(match_results),
            matched=matched,
            unmatched=len(match_results) - matched,
            match_details=match_results,
            output_bytes=output_bytes,
            output_ext=output_ext,
            normalized_records=normalized_records,
        )
    

    def process_pdf(
        self,
        file: io.BytesIO,
        filename: str,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ) -> NormalizationReport:
        file.seek(0)
        structured_records, doc_type = parse_pdf(file)

        file.seek(0)
        raw_text = extract_raw_text_from_pdf(file)

        if doc_type == "unstructured" or should_use_llm_for_text(raw_text, len(structured_records)):
            records = parse_unstructured_text(raw_text)
            final_doc_type = "unstructured"
        else:
            records = structured_records
            final_doc_type = doc_type

        norm_records, all_results = self._normalize_records(
            records,
            fuzzy_threshold,
            semantic_threshold,
        )

        out = write_pdf_keyvalue(norm_records, title=filename)

        return self._finalize_report(
            input_format="pdf",
            doc_type=final_doc_type,
            normalized_records=norm_records,
            match_results=all_results,
            output_bytes=out,
            output_ext="pdf",
        )
    def process_docx(
        self,
        file: io.BytesIO,
        filename: str,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ) -> NormalizationReport:
        file.seek(0)
        structured_records, doc_type = parse_docx(file)
        file.seek(0)
        raw_text = extract_raw_text_from_docx(file)
        if doc_type == "unstructured" or should_use_llm_for_text(raw_text, len(structured_records)):
            records = parse_unstructured_text(raw_text)
            final_doc_type = "unstructured"
        else:
            records = structured_records
            final_doc_type = doc_type

        norm_records, all_results = self._normalize_records(
            records,
            fuzzy_threshold,
            semantic_threshold,
        )
        out = write_docx_keyvalue(norm_records, title=filename)
        return self._finalize_report(
            input_format="docx",
            doc_type=final_doc_type,
            normalized_records=norm_records,
            match_results=all_results,
            output_bytes=out,
            output_ext="docx",
    )
    

    def process_excel(
        self,
        file: io.BytesIO,
        filename: str,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ) -> NormalizationReport:
        file.seek(0)
        sheets, doc_type = parse_excel(file)

        structured_records = []
        has_multicolumn = False

        for _, df in sheets.items():
            if len(df.columns) > 2:
                has_multicolumn = True
                structured_records.extend(extract_structured_records_from_excel_sheet(df))
            else:
                structured_records.extend(extract_kv_from_excel_sheet(df))

        file.seek(0)
        raw_text = extract_raw_text_from_excel(file)

        if should_use_llm_for_text(raw_text, len(structured_records)):
            records = parse_unstructured_text(raw_text)
            final_doc_type = "unstructured"
        else:
            records = structured_records
            final_doc_type = "multicolumn" if has_multicolumn else doc_type

        norm_records, all_results = self._normalize_records(
            records,
            fuzzy_threshold,
            semantic_threshold,
        )

        out = write_excel_keyvalue(norm_records)

        return self._finalize_report(
            input_format="excel",
            doc_type=final_doc_type,
            normalized_records=norm_records,
            match_results=all_results,
            output_bytes=out,
            output_ext="xlsx",
        )

    def process_csv(
        self,
        file: io.BytesIO,
        filename: str,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ) -> NormalizationReport:
        file.seek(0)
        df, doc_type = parse_csv(file)

        has_multicolumn = len(df.columns) > 2
        if has_multicolumn:
            structured_records = extract_structured_records_from_csv_df(df)
        else:
            structured_records = []
            for _, row in df.iterrows():
                cells = [
                    str(v).strip()
                    for v in row.tolist()
                    if str(v).strip() and str(v).strip().lower() not in ("nan", "none")
                ]
                if len(cells) >= 2 and cells[0].lower() not in ("field label", "attribute", "name"):
                    structured_records.append({"attribute": cells[0], "value": cells[1]})

        file.seek(0)
        raw_text = extract_raw_text_from_csv(file)

        if should_use_llm_for_text(raw_text, len(structured_records)):
            records = parse_unstructured_text(raw_text)
            final_doc_type = "unstructured"
        else:
            records = structured_records
            final_doc_type = "multicolumn" if has_multicolumn else doc_type

        norm_records, all_results = self._normalize_records(
            records,
            fuzzy_threshold,
            semantic_threshold,
        )

        out = write_csv_records(norm_records)

        return self._finalize_report(
            input_format="csv",
            doc_type=final_doc_type,
            normalized_records=norm_records,
            match_results=all_results,
            output_bytes=out,
            output_ext="csv",
        )

    def process(
        self,
        file: io.BytesIO,
        filename: str,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    ) -> NormalizationReport:
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return self.process_pdf(file, filename, fuzzy_threshold, semantic_threshold)
        elif ext in (".xlsx", ".xls"):
            return self.process_excel(file, filename, fuzzy_threshold, semantic_threshold)
        elif ext == ".csv":
            return self.process_csv(file, filename, fuzzy_threshold, semantic_threshold)
        elif ext == ".docx":
            return self.process_docx(file, filename, fuzzy_threshold, semantic_threshold)

        raise ValueError(f"Unsupported file type: {ext}")