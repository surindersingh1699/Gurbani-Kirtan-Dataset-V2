"""
STTM (SikhiToTheMax) fuzzy matcher.
Matches OCR'd Gurmukhi text against the canonical STTM database
to get corrected text and English translations.
"""

import re
import sqlite3
from functools import lru_cache

from anvaad_py import unicode as ascii_to_unicode
from rapidfuzz import fuzz, process

# Default: Dr. Sant Singh Khalsa (most complete English translation)
DEFAULT_TRANSLATION_SOURCE = 1

# Unicode Gurmukhi ranges
_GURMUKHI_CONSONANTS = set(range(0x0A15, 0x0A39 + 1))
_GURMUKHI_VOWELS = set(range(0x0A05, 0x0A14 + 1))
_GURMUKHI_SIGNS = set(range(0x0A3E, 0x0A4D + 1))

# Map Unicode Gurmukhi first-letter to STTM ASCII first-letter
_UNI_TO_ASCII_FIRST = {
    'ੴ': '<>', 'ਅ': 'A', 'ਆ': 'a', 'ਇ': 'ie', 'ਈ': 'eI',
    'ਉ': 'au', 'ਊ': 'aU', 'ਏ': 'e', 'ਐ': 'E', 'ਓ': 'o', 'ਔ': 'O',
    'ਕ': 'k', 'ਖ': 'K', 'ਗ': 'g', 'ਘ': 'G', 'ਙ': '|',
    'ਚ': 'c', 'ਛ': 'C', 'ਜ': 'j', 'ਝ': 'J', 'ਞ': '\\',
    'ਟ': 't', 'ਠ': 'T', 'ਡ': 'f', 'ਢ': 'F', 'ਣ': 'x',
    'ਤ': 'q', 'ਥ': 'Q', 'ਦ': 'd', 'ਧ': 'D', 'ਨ': 'n',
    'ਪ': 'p', 'ਫ': 'P', 'ਬ': 'b', 'ਭ': 'B', 'ਮ': 'm',
    'ਯ': 'X', 'ਰ': 'r', 'ਲ': 'l', 'ਵ': 'v', 'ੜ': 'V',
    'ਸ': 's', 'ਹ': 'h',
    # Extended (with nukta)
    'ਖ਼': '^', 'ਗ਼': 'Z', 'ਜ਼': 'z', 'ਫ਼': '&', 'ਲ਼': 'L', 'ਸ਼': 'S',
}


def _extract_first_letters_unicode(text):
    """Extract first Gurmukhi letter of each word from Unicode text."""
    letters = []
    for word in text.split():
        for ch in word:
            cp = ord(ch)
            if cp in _GURMUKHI_CONSONANTS or cp in _GURMUKHI_VOWELS:
                ascii_fl = _UNI_TO_ASCII_FIRST.get(ch, '')
                if ascii_fl:
                    letters.append(ascii_fl)
                break
    return ''.join(letters)


def _clean_gurmukhi(text):
    """Remove punctuation and extra whitespace from Gurmukhi text."""
    # Remove common punctuation: ॥, ।, digits, etc.
    text = re.sub(r'[॥।\d੦੧੨੩੪੫੬੭੮੯\[\](),.;:!?]', '', text)
    return ' '.join(text.split())


class STTMMatcher:
    """Matches OCR Gurmukhi against the STTM database for ground-truth correction."""

    def __init__(self, db_path, translation_source_id=DEFAULT_TRANSLATION_SOURCE):
        self.db_path = db_path
        self.translation_source_id = translation_source_id
        self._lines = []           # [(line_id, gurmukhi_ascii, gurmukhi_unicode, first_letters)]
        self._fl_index = {}        # first_letters -> list of indices into _lines
        self._translations = {}    # line_id -> english_translation
        self._unicode_lookup = []  # list of gurmukhi_unicode strings for rapidfuzz
        self._loaded = False

    def load(self):
        """Load all lines from the database and build indices."""
        if self._loaded:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Load all lines
        cursor.execute("SELECT id, gurmukhi, first_letters FROM lines ORDER BY order_id")
        for line_id, gurmukhi_ascii, first_letters in cursor.fetchall():
            gurmukhi_unicode = ascii_to_unicode(gurmukhi_ascii)
            idx = len(self._lines)
            self._lines.append((line_id, gurmukhi_ascii, gurmukhi_unicode, first_letters or ''))

            # Build first-letters index
            fl = (first_letters or '').strip()
            if fl:
                self._fl_index.setdefault(fl, []).append(idx)

            self._unicode_lookup.append(_clean_gurmukhi(gurmukhi_unicode))

        # Load English translations
        cursor.execute(
            "SELECT line_id, translation FROM translations WHERE translation_source_id = ?",
            (self.translation_source_id,)
        )
        for line_id, translation in cursor.fetchall():
            self._translations[line_id] = translation

        conn.close()
        self._loaded = True
        print(f"STTM matcher loaded: {len(self._lines)} lines, "
              f"{len(self._translations)} translations")

    def match(self, ocr_gurmukhi, min_score=65):
        """
        Match OCR'd Gurmukhi text against the STTM database.

        Args:
            ocr_gurmukhi: Unicode Gurmukhi text from OCR
            min_score: Minimum fuzzy match score (0-100) to accept

        Returns:
            dict with keys: matched, score, line_id, gurmukhi_canonical,
                           english_translation, gurmukhi_ascii
            or None if no match above threshold
        """
        if not self._loaded:
            self.load()

        if not ocr_gurmukhi or not ocr_gurmukhi.strip():
            return None

        cleaned = _clean_gurmukhi(ocr_gurmukhi)
        if not cleaned:
            return None

        # Tier 1: Try first-letters based lookup
        fl = _extract_first_letters_unicode(ocr_gurmukhi)
        if fl:
            candidates = self._get_fl_candidates(fl)
            if candidates:
                best = self._best_match_from_candidates(cleaned, candidates)
                if best and best[1] >= min_score:
                    return self._build_result(best[0], best[1])

        # Tier 2: Full corpus fuzzy search (slower but more robust)
        result = process.extractOne(
            cleaned,
            self._unicode_lookup,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=min_score,
        )

        if result:
            match_text, score, idx = result
            return self._build_result(idx, score)

        return None

    def match_by_english(self, english_text, min_score=60):
        """
        Match using English translation text (fallback when Gurmukhi OCR is poor).

        Args:
            english_text: English text from OCR
            min_score: Minimum fuzzy match score

        Returns:
            Same as match()
        """
        if not self._loaded:
            self.load()

        if not english_text or not english_text.strip():
            return None

        cleaned = english_text.strip().lower()
        translations = list(self._translations.items())
        trans_texts = [t.lower() for _, t in translations]

        result = process.extractOne(
            cleaned,
            trans_texts,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=min_score,
        )

        if result:
            _, score, idx = result
            line_id = translations[idx][0]
            # Find the line index by line_id
            for i, (lid, _, _, _) in enumerate(self._lines):
                if lid == line_id:
                    return self._build_result(i, score)

        return None

    def _get_fl_candidates(self, first_letters):
        """Get candidate line indices from first-letters index."""
        candidates = set()

        # Exact match
        if first_letters in self._fl_index:
            candidates.update(self._fl_index[first_letters])

        # Also try with 1-char tolerance (common OCR errors)
        for fl, indices in self._fl_index.items():
            if abs(len(fl) - len(first_letters)) <= 1:
                if fuzz.ratio(fl, first_letters) >= 80:
                    candidates.update(indices)

        return list(candidates)

    def _best_match_from_candidates(self, cleaned_text, candidate_indices):
        """Find the best matching candidate using fuzzy matching."""
        best_idx = None
        best_score = 0

        for idx in candidate_indices:
            score = fuzz.token_sort_ratio(cleaned_text, self._unicode_lookup[idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        return (best_idx, best_score) if best_idx is not None else None

    def _build_result(self, line_idx, score):
        """Build a result dict from a matched line index."""
        line_id, gurmukhi_ascii, gurmukhi_unicode, _ = self._lines[line_idx]
        return {
            "matched": True,
            "score": round(score, 1),
            "line_id": line_id,
            "gurmukhi_canonical": gurmukhi_unicode,
            "gurmukhi_ascii": gurmukhi_ascii,
            "english_translation": self._translations.get(line_id, ""),
        }


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "database.sqlite"
    matcher = STTMMatcher(db_path)
    matcher.load()

    # Test with some known lines
    test_lines = [
        "ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ",
        "ਆਦਿ ਸਚੁ ਜੁਗਾਦਿ ਸਚੁ",
        "ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖਾਲਸਾ",
    ]

    for line in test_lines:
        result = matcher.match(line)
        if result:
            print(f"\nOCR:       {line}")
            print(f"Canonical: {result['gurmukhi_canonical']}")
            print(f"English:   {result['english_translation'][:80]}")
            print(f"Score:     {result['score']}")
        else:
            print(f"\nOCR: {line} -> NO MATCH")
