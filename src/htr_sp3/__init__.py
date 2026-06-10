"""SP-3: text-based RAG correction over OCR output (scenarios M3 and M4).

The handwriting image is already turned into text by PaliGemma (SP-1/SP-2); SP-3 only repairs
spelling errors in that text against a vocabulary of valid English words from IAM-line. No
images are stored or queried here — this is a lexical corrector, not an image-retrieval system.
"""
