#!/usr/bin/env python3

"""
Supreme Smart PDF Analyzer for NEET Question Papers
Advanced AI-powered extraction with multiple fallback strategies
Enhanced pattern recognition and validation systems
"""

import os
import re
import fitz  # PyMuPDF
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from PIL import ImageEnhance, ImageFilter

class SmartPDFAnalyzer:
    def __init__(self, debug_mode=False):
        self.question_patterns = []
        self.instruction_patterns = []
        self.option_patterns = []
        self.question_keywords = set()
        self.instruction_keywords = set()
        self.model_data = {}
        self._text_cache = {}
        self.debug_mode = debug_mode
        
        # Enhanced pattern recognition system
        self.advanced_patterns = {
            'question_starters': [],
            'option_markers': [],
            'instruction_indicators': [],
            'subject_keywords': {},
            'difficulty_indicators': {}
        }
        
        # Performance optimization
        self._extraction_stats = defaultdict(int)
        self._cache_lock = threading.Lock()
        
        # Initialize advanced patterns
        self._initialize_supreme_patterns()

    def _initialize_supreme_patterns(self):
        """Initialize comprehensive pattern recognition system"""
        
        # Supreme question starter patterns
        self.advanced_patterns['question_starters'] = [
            # Standard formats
            re.compile(r'^\s*(Q?\s*\d+[\.\)]\s*)', re.I),
            re.compile(r'^\s*(Question\s+\d+[\.\)]?\s*)', re.I),
            re.compile(r'^\s*(\(\d+\)\s*)', re.I),
            re.compile(r'^\s*(\[\d+\]\s*)', re.I),
            re.compile(r'^\s*(\d+[\.\)]\s*)', re.I),
            
            # Advanced formats
            re.compile(r'^\s*(Part\s+[A-Z]\s+\d+[\.\)]?\s*)', re.I),
            re.compile(r'^\s*(Section\s+[A-Z]\s+\d+[\.\)]?\s*)', re.I),
            re.compile(r'^\s*(\d+\s*[-–—]\s*)', re.I),
            re.compile(r'^\s*(Q\d+\s*)', re.I),
            re.compile(r'^\s*(\d+\s*\.?\s*[A-Z])', re.I),
            
            # Special formats
            re.compile(r'^\s*(\d{1,3}\s*[\.:])', re.I),
            re.compile(r'^\s*([IVX]+\s*[\.\)]\s*)', re.I),  # Roman numerals
            re.compile(r'^\s*(Item\s+\d+[\.\)]?\s*)', re.I),
        ]
        
        # Supreme option marker patterns
        self.advanced_patterns['option_markers'] = [
            # Standard formats
            re.compile(r'^\s*[\(\[]?([A-Da-d])[\)\.\]]\s*(.+)', re.I),
            re.compile(r'^\s*([A-Da-d])[\.\)]\s*(.+)', re.I),
            re.compile(r'^\s*[\(\[]([A-Da-d])[\)\]]\s*(.+)', re.I),
            
            # Numeric options
            re.compile(r'^\s*[\(\[]?([1-4])[\)\.\]]\s*(.+)', re.I),
            re.compile(r'^\s*([1-4])[\.\)]\s*(.+)', re.I),
            
            # Special formats
            re.compile(r'^\s*([A-Da-d])\s*[-–—]\s*(.+)', re.I),
            re.compile(r'^\s*([A-Da-d])\s*[:]?\s*(.+)', re.I),
            re.compile(r'^\s*Option\s*([A-Da-d])[\.\):]?\s*(.+)', re.I),
        ]
        
        # Enhanced subject keywords for NEET
        self.advanced_patterns['subject_keywords'] = {
            'physics': {
                'mechanics': ['force', 'motion', 'velocity', 'acceleration', 'momentum', 'energy', 'work', 'power', 'friction', 'gravity'],
                'thermodynamics': ['heat', 'temperature', 'entropy', 'pressure', 'volume', 'gas', 'thermal', 'calorimeter'],
                'waves': ['frequency', 'wavelength', 'amplitude', 'resonance', 'interference', 'diffraction', 'sound', 'light'],
                'electricity': ['current', 'voltage', 'resistance', 'capacitance', 'inductance', 'magnetic', 'electric', 'circuit'],
                'optics': ['lens', 'mirror', 'reflection', 'refraction', 'prism', 'telescope', 'microscope', 'laser'],
                'modern': ['quantum', 'photoelectric', 'atomic', 'nuclear', 'radioactive', 'fusion', 'fission', 'electron']
            },
            'chemistry': {
                'organic': ['carbon', 'hydrocarbon', 'alkane', 'alkene', 'benzene', 'alcohol', 'acid', 'ester', 'polymer'],
                'inorganic': ['metal', 'salt', 'oxide', 'hydroxide', 'halogen', 'noble', 'transition', 'coordination'],
                'physical': ['equilibrium', 'kinetics', 'thermodynamics', 'electrochemistry', 'solution', 'colligative'],
                'analytical': ['titration', 'spectroscopy', 'chromatography', 'qualitative', 'quantitative']
            },
            'biology': {
                'cell': ['nucleus', 'mitochondria', 'chloroplast', 'ribosome', 'membrane', 'cytoplasm', 'organelle'],
                'genetics': ['dna', 'rna', 'gene', 'chromosome', 'mutation', 'inheritance', 'allele', 'protein'],
                'physiology': ['heart', 'kidney', 'liver', 'lung', 'brain', 'hormone', 'enzyme', 'neuron', 'blood'],
                'ecology': ['ecosystem', 'population', 'community', 'biodiversity', 'conservation', 'pollution'],
                'evolution': ['darwin', 'natural', 'selection', 'adaptation', 'species', 'fossil', 'phylogeny'],
                'plant': ['photosynthesis', 'transpiration', 'root', 'stem', 'leaf', 'flower', 'seed', 'fruit']
            }
        }
        
        # Difficulty indicator patterns
        self.advanced_patterns['difficulty_indicators'] = {
            'easy': ['what', 'which', 'where', 'when', 'who', 'is', 'are', 'has', 'have', 'define', 'identify', 'name'],
            'moderate': ['how', 'why', 'explain', 'describe', 'compare', 'distinguish', 'relate', 'function', 'process'],
            'tough': ['analyze', 'evaluate', 'calculate', 'derive', 'prove', 'synthesize', 'predict', 'determine', 'assess']
        }

    def _pdftotext_path(self):
        """Enhanced pdftotext detection with multiple search paths"""
        candidates = [
            # Local poppler installation
            os.path.join(os.getcwd(), 'poppler-24.08.0', 'Library', 'bin', 'pdftotext.exe'),
            os.path.join(os.getcwd(), 'poppler', 'bin', 'pdftotext.exe'),
            
            # System installations
            shutil.which('pdftotext'),
            r"C:\Program Files\poppler\bin\pdftotext.exe",
            r"C:\Program Files (x86)\poppler\bin\pdftotext.exe",
            
            # Portable installations
            'pdftotext.exe',
            'bin/pdftotext.exe',
        ]
        
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def _detect_tesseract(self):
        """Enhanced Tesseract detection and configuration"""
        try:
            import pytesseract
            
            # If already configured, return True
            if getattr(pytesseract.pytesseract, 'tesseract_cmd', None):
                return True
                
            # Search for Tesseract installation
            candidates = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
                shutil.which('tesseract')
            ]
            
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    return True
                    
            return False
            
        except ImportError:
            return False

    def extract_text_supreme(self, pdf_path):
        """Supreme text extraction with multiple advanced methods and caching"""
        try:
            # Enhanced caching with file modification time
            try:
                stat = os.stat(pdf_path)
                cache_key = (pdf_path, stat.st_size, int(stat.st_mtime))
            except Exception:
                cache_key = (pdf_path, None, None)

            with self._cache_lock:
                if cache_key in self._text_cache:
                    self._extraction_stats['cache_hits'] += 1
                    return self._text_cache[cache_key]

            start_time = time.time()
            
            # Method 1: Enhanced pdftotext with multiple configurations
            pdftotext_result = self._extract_with_pdftotext(pdf_path)
            if self._is_good_extraction(pdftotext_result):
                with self._cache_lock:
                    self._text_cache[cache_key] = pdftotext_result
                self._log_extraction_success("pdftotext", len(pdftotext_result), time.time() - start_time)
                return pdftotext_result

            # Method 2: Advanced PyMuPDF with intelligent page detection
            pymupdf_result = self._extract_with_pymupdf_advanced(pdf_path)
            if self._is_good_extraction(pymupdf_result):
                with self._cache_lock:
                    self._text_cache[cache_key] = pymupdf_result
                self._log_extraction_success("pymupdf_advanced", len(pymupdf_result), time.time() - start_time)
                return pymupdf_result

            # Method 3: OCR-enhanced extraction
            ocr_result = self._extract_with_ocr_enhanced(pdf_path)
            if self._is_good_extraction(ocr_result):
                with self._cache_lock:
                    self._text_cache[cache_key] = ocr_result
                self._log_extraction_success("ocr_enhanced", len(ocr_result), time.time() - start_time)
                return ocr_result

            # Method 4: Multi-mode PyMuPDF extraction
            multimode_result = self._extract_with_multimode(pdf_path)
            if self._is_good_extraction(multimode_result):
                with self._cache_lock:
                    self._text_cache[cache_key] = multimode_result
                self._log_extraction_success("multimode", len(multimode_result), time.time() - start_time)
                return multimode_result

            # Method 5: Aggressive fallback extraction
            fallback_result = self._extract_with_aggressive_fallback(pdf_path)
            
            with self._cache_lock:
                self._text_cache[cache_key] = fallback_result
            self._log_extraction_success("fallback", len(fallback_result or ""), time.time() - start_time)
            return fallback_result or ""

        except Exception as e:
            if self.debug_mode:
                print(f"❌ Supreme extraction failed: {str(e)}")
            return ""

    def _extract_with_pdftotext(self, pdf_path):
        """Enhanced pdftotext extraction with multiple configurations"""
        pdftotext = self._pdftotext_path()
        if not pdftotext:
            return None
            
        configurations = [
            ['-enc', 'UTF-8', '-layout'],
            ['-enc', 'UTF-8', '-raw'],
            ['-enc', 'UTF-8', '-table'],
            ['-enc', 'UTF-8'],
        ]
        
        best_result = None
        best_score = 0
        
        for config in configurations:
            try:
                cmd = [pdftotext] + config + [pdf_path, '-']
                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True, creationflags=0x08000000, timeout=30
                )
                text = result.stdout.decode('utf-8', errors='ignore')
                
                # Score the extraction quality
                score = self._score_text_quality(text)
                if score > best_score:
                    best_score = score
                    best_result = text
                    
            except Exception:
                continue
                
        return best_result

    def _extract_with_pymupdf_advanced(self, pdf_path):
        """Advanced PyMuPDF extraction with intelligent page selection"""
        try:
            doc = fitz.open(pdf_path)
            
            # Enhanced question detection patterns
            question_indicators = [
                re.compile(r'(?:^|\n)\s*\d{1,3}[\.\)]\s*[A-Z]', re.MULTILINE),
                re.compile(r'(?:^|\n)\s*Q\s*\d+', re.MULTILINE | re.IGNORECASE),
                re.compile(r'(?:^|\n)\s*Question\s+\d+', re.MULTILINE | re.IGNORECASE),
                re.compile(r'(?:^|\n)\s*\(\d+\)', re.MULTILINE),
            ]
            
            # Phase 1: Intelligent start detection
            start_page = self._find_question_start_page(doc, question_indicators)
            
            # Phase 2: Content-aware extraction
            texts = []
            question_streak = 0
            no_question_streak = 0
            
            for page_idx in range(start_page, len(doc)):
                page = doc[page_idx]
                
                # Try multiple extraction methods per page
                page_texts = []
                
                # Method 1: Standard text extraction
                try:
                    std_text = page.get_text("text")
                    if std_text.strip():
                        page_texts.append(('standard', std_text))
                except:
                    pass
                
                # Method 2: Dictionary-based extraction
                try:
                    dict_text = self._extract_from_dict_mode(page)
                    if dict_text.strip():
                        page_texts.append(('dict', dict_text))
                except:
                    pass
                
                # Method 3: Word-based extraction
                try:
                    word_text = self._extract_from_words_mode(page)
                    if word_text.strip():
                        page_texts.append(('words', word_text))
                except:
                    pass
                
                # Select best extraction for this page
                best_page_text = self._select_best_page_text(page_texts)
                
                if best_page_text:
                    texts.append(best_page_text)
                    
                    # Check if page contains questions
                    if any(pattern.search(best_page_text) for pattern in question_indicators):
                        question_streak += 1
                        no_question_streak = 0
                    else:
                        no_question_streak += 1
                        
                    # Stop if we've gone too far past questions
                    if no_question_streak >= 8 and question_streak >= 10:
                        break
                else:
                    no_question_streak += 1
            
            doc.close()
            
            combined_text = "\n".join(texts)
            return self._post_process_text(combined_text)
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Advanced PyMuPDF extraction failed: {str(e)}")
            return None

    def _find_question_start_page(self, doc, question_indicators):
        """Find the page where questions likely start"""
        max_search_pages = min(15, len(doc))
        
        for page_idx in range(max_search_pages):
            try:
                page_text = doc[page_idx].get_text("text")
                question_count = sum(len(pattern.findall(page_text)) for pattern in question_indicators)
                
                # If we find significant question patterns, this is likely the start
                if question_count >= 5:
                    return page_idx
                    
                # Also check for NEET-specific indicators
                if any(indicator in page_text.lower() for indicator in 
                       ['biology', 'physics', 'chemistry', 'botany', 'zoology', 'neet', 'question paper']):
                    if question_count >= 2:
                        return page_idx
                        
            except:
                continue
                
        return 0  # Default fallback

    def _extract_from_dict_mode(self, page):
        """Extract text using dictionary mode with enhanced processing"""
        try:
            page_dict = page.get_text("dict")
            text_parts = []
            
            if 'blocks' in page_dict:
                for block in page_dict['blocks']:
                    if 'lines' in block:
                        block_lines = []
                        for line in block['lines']:
                            if 'spans' in line:
                                line_text = ""
                                for span in line['spans']:
                                    if 'text' in span and span['text'].strip():
                                        line_text += span['text']
                                if line_text.strip():
                                    block_lines.append(line_text.strip())
                        
                        if block_lines:
                            text_parts.append('\n'.join(block_lines))
            
            return '\n\n'.join(text_parts)
            
        except Exception:
            return ""

    def _extract_from_words_mode(self, page):
        """Extract text using words mode with intelligent reconstruction"""
        try:
            words = page.get_text("words")
            if not words:
                return ""
            
            # Group words by approximate line position
            lines = defaultdict(list)
            for word in words:
                if len(word) >= 5:  # Ensure word has coordinates and text
                    x0, y0, x1, y1, text = word[0], word[1], word[2], word[3], word[4]
                    if text.strip() and len(text.strip()) > 1:
                        line_key = round(y0 / 5) * 5  # Group by approximate y-coordinate
                        lines[line_key].append((x0, text.strip()))
            
            # Reconstruct lines
            reconstructed_lines = []
            for line_y in sorted(lines.keys()):
                line_words = sorted(lines[line_y], key=lambda x: x[0])  # Sort by x-coordinate
                line_text = ' '.join(word[1] for word in line_words)
                if len(line_text.strip()) > 3:
                    reconstructed_lines.append(line_text)
            
            return '\n'.join(reconstructed_lines)
            
        except Exception:
            return ""

    def _select_best_page_text(self, page_texts):
        """Select the best text extraction from multiple methods"""
        if not page_texts:
            return ""
            
        # Score each extraction
        scored_texts = []
        for method, text in page_texts:
            score = self._score_text_quality(text)
            scored_texts.append((score, text, method))
        
        # Return the highest scoring text
        if scored_texts:
            best_score, best_text, best_method = max(scored_texts)
            return best_text
            
        return ""

    def _extract_with_ocr_enhanced(self, pdf_path):
        """Enhanced OCR extraction with multiple configurations and preprocessing"""
        if not self._detect_tesseract():
            return None
            
        try:
            import pytesseract
            from PIL import Image, ImageEnhance, ImageFilter
        except ImportError:
            return None
        
        try:
            doc = fitz.open(pdf_path)
            ocr_texts = []
            
            # Process first few pages with OCR
            max_ocr_pages = min(5, len(doc))
            
            for page_idx in range(max_ocr_pages):
                try:
                    page = doc[page_idx]
                    
                    # Multiple resolution attempts
                    for dpi in [200, 250, 300]:
                        try:
                            zoom = dpi / 72.0
                            mat = fitz.Matrix(zoom, zoom)
                            pix = page.get_pixmap(matrix=mat, alpha=False)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            
                            # Image preprocessing
                            enhanced_img = self._preprocess_image_for_ocr(img)
                            
                            # Multiple OCR configurations
                            ocr_configs = [
                                '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()[].,;:?! ',
                                '--psm 3 --oem 3',
                                '--psm 4 --oem 3',
                                '--psm 1 --oem 3'
                            ]
                            
                            best_ocr_result = ""
                            best_ocr_score = 0
                            
                            for config in ocr_configs:
                                try:
                                    ocr_text = pytesseract.image_to_string(enhanced_img, config=config)
                                    if ocr_text.strip():
                                        score = self._score_text_quality(ocr_text)
                                        if score > best_ocr_score:
                                            best_ocr_score = score
                                            best_ocr_result = ocr_text
                                except:
                                    continue
                            
                            if best_ocr_result and best_ocr_score > 10:  # Minimum quality threshold
                                ocr_texts.append(f"--- OCR Page {page_idx + 1} (DPI: {dpi}) ---\n{best_ocr_result}")
                                break  # Found good result, no need to try higher DPI
                                
                        except:
                            continue
                            
                except:
                    continue
            
            doc.close()
            
            if ocr_texts:
                return '\n\n'.join(ocr_texts)
                
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Enhanced OCR extraction failed: {str(e)}")
        
        return None

    def _preprocess_image_for_ocr(self, img):
        """Preprocess image for better OCR results"""
        try:
            # Convert to grayscale for better OCR
            img = img.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            img = img.filter(ImageFilter.MedianFilter(size=3))
            
            return img
            
        except:
            return img

    def _extract_with_multimode(self, pdf_path):
        """Extract using multiple PyMuPDF modes and combine results"""
        try:
            doc = fitz.open(pdf_path)
            
            all_texts = []
            max_pages = min(10, len(doc))
            
            for page_idx in range(max_pages):
                try:
                    page = doc[page_idx]
                    page_results = []
                    
                    # Try different extraction modes
                    modes = ["text", "html", "xhtml", "xml"]
                    
                    for mode in modes:
                        try:
                            text = page.get_text(mode)
                            if text and text.strip():
                                if mode in ["html", "xhtml", "xml"]:
                                    # Clean HTML/XML tags
                                    text = self._clean_markup(text)
                                
                                quality_score = self._score_text_quality(text)
                                page_results.append((quality_score, text, mode))
                                
                        except:
                            continue
                    
                    # Select best result for this page
                    if page_results:
                        best_score, best_text, best_mode = max(page_results)
                        if best_score > 5:  # Minimum quality threshold
                            all_texts.append(best_text)
                            
                except:
                    continue
            
            doc.close()
            
            if all_texts:
                combined = '\n\n'.join(all_texts)
                return self._post_process_text(combined)
                
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Multi-mode extraction failed: {str(e)}")
        
        return None

    def _extract_with_aggressive_fallback(self, pdf_path):
        """Aggressive fallback extraction using all available methods"""
        try:
            doc = fitz.open(pdf_path)
            fallback_texts = []
            
            for page_idx in range(min(8, len(doc))):
                try:
                    page = doc[page_idx]
                    page_text_parts = []
                    
                    # Method 1: Raw text extraction
                    try:
                        raw_text = page.get_text("rawdict")
                        if raw_text and isinstance(raw_text, dict) and 'text' in raw_text:
                            if raw_text['text'].strip():
                                page_text_parts.append(raw_text['text'])
                    except:
                        pass
                    
                    # Method 2: Character-level extraction
                    try:
                        chars = page.get_text("dict")
                        char_text = self._extract_from_chars(chars)
                        if char_text.strip():
                            page_text_parts.append(char_text)
                    except:
                        pass
                    
                    # Method 3: Block-based extraction
                    try:
                        blocks = page.get_text_blocks()
                        if blocks:
                            block_texts = [block[4] for block in blocks if len(block) > 4 and block[4].strip()]
                            if block_texts:
                                page_text_parts.append('\n'.join(block_texts))
                    except:
                        pass
                    
                    # Combine page results
                    if page_text_parts:
                        # Select the longest/most complete text
                        best_part = max(page_text_parts, key=len)
                        if len(best_part.strip()) > 20:
                            fallback_texts.append(best_part)
                            
                except:
                    continue
            
            doc.close()
            
            if fallback_texts:
                combined = '\n\n'.join(fallback_texts)
                return self._post_process_text(combined)
                
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Aggressive fallback extraction failed: {str(e)}")
        
        return ""

    def _extract_from_chars(self, chars_dict):
        """Extract text from character-level dictionary"""
        try:
            text_parts = []
            
            if 'blocks' in chars_dict:
                for block in chars_dict['blocks']:
                    if 'lines' in block:
                        block_lines = []
                        for line in block['lines']:
                            if 'spans' in line:
                                line_chars = []
                                for span in line['spans']:
                                    if 'chars' in span:
                                        for char in span['chars']:
                                            if isinstance(char, dict) and 'c' in char:
                                                line_chars.append(char['c'])
                                    elif 'text' in span:
                                        line_chars.append(span['text'])
                                
                                if line_chars:
                                    line_text = ''.join(line_chars).strip()
                                    if line_text:
                                        block_lines.append(line_text)
                        
                        if block_lines:
                            text_parts.append('\n'.join(block_lines))
            
            return '\n\n'.join(text_parts)
            
        except Exception:
            return ""

    def _clean_markup(self, text):
        """Clean HTML/XML markup from text"""
        try:
            # Remove HTML/XML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove common markup artifacts
            text = re.sub(r'&[a-zA-Z0-9]+;', '', text)  # HTML entities
            text = re.sub(r'data:image[^;]+;base64[^"\']+', '', text)  # Base64 images
            
            # Clean excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            
            return text.strip()
            
        except Exception:
            return text

    def _post_process_text(self, text):
        """Post-process extracted text for better quality"""
        if not text:
            return text
            
        try:
            # Remove page headers/footers
            lines = text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                
                # Skip likely header/footer lines
                if self._is_likely_header_footer(line):
                    continue
                    
                # Skip very short lines that are likely artifacts
                if len(line) < 3:
                    continue
                    
                # Clean common OCR artifacts
                line = self._clean_ocr_artifacts(line)
                
                if line:
                    cleaned_lines.append(line)
            
            # Rejoin and normalize spacing
            cleaned_text = '\n'.join(cleaned_lines)
            cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
            
            return cleaned_text
            
        except Exception:
            return text

    def _is_likely_header_footer(self, line):
        """Identify likely header/footer lines"""
        line_lower = line.lower()
        
        # Common header/footer patterns
        header_footer_indicators = [
            'page', 'copyright', '©', 'all rights reserved',
            'neet', 'jee', 'entrance', 'examination',
            'time:', 'duration:', 'marks:', 'total:',
            'instructions', 'note:', 'important:'
        ]
        
        # Very short lines with only numbers or symbols
        if len(line.strip()) <= 3 and re.match(r'^[\d\-\s]+$', line.strip()):
            return True
            
        # Lines that are mostly symbols or repeated characters
        if len(set(line.strip())) <= 2 and len(line.strip()) > 10:
            return True
            
        # Check for common header/footer content
        if any(indicator in line_lower for indicator in header_footer_indicators):
            if len(line.split()) <= 5:  # Short lines with these keywords are likely headers
                return True
                
        return False

    def _clean_ocr_artifacts(self, line):
        """Clean common OCR artifacts from text"""
        try:
            # Fix common OCR character substitutions
            substitutions = {
                '|': 'I', '!': 'l', '0': 'O', '5': 'S', '1': 'l',
                '8': 'B', '6': 'G', '9': 'g', '2': 'Z'
            }
            
            # Only apply substitutions in contexts where they make sense
            # This is a simplified approach - could be made more sophisticated
            
            # Remove isolated special characters
            line = re.sub(r'\s[^\w\s]\s', ' ', line)
            
            # Fix spacing issues
            line = re.sub(r'([a-z])([A-Z])', r'\1 \2', line)  # Add space between lowercase and uppercase
            line = re.sub(r'(\d)([A-Za-z])', r'\1 \2', line)  # Add space between digit and letter
            
            # Clean excessive punctuation
            line = re.sub(r'([.!?]){3,}', r'\1', line)
            
            return line.strip()
            
        except Exception:
            return line

    def _is_good_extraction(self, text):
        """Determine if text extraction is of good quality"""
        if not text or not text.strip():
            return False
            
        # Check meaningful content ratio
        meaningful_chars = len(re.findall(r'[A-Za-z0-9]', text))
        if meaningful_chars < 100:
            return False
            
        # Check for question patterns
        question_indicators = [
            r'\b\d+[\.\)]\s*[A-Z]',  # Question numbers followed by text
            r'\bwhich\b|\bwhat\b|\bhow\b|\bwhy\b',  # Question words
            r'\b[a-d][\.\)]\s*',  # Option markers
        ]
        
        question_matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                             for pattern in question_indicators)
        
        return question_matches >= 10  # Minimum threshold for good extraction

    def _score_text_quality(self, text):
        """Score text quality for comparison"""
        if not text:
            return 0
            
        score = 0
        
        # Basic length score
        score += min(len(text) / 1000, 10)
        
        # Meaningful character ratio
        meaningful_chars = len(re.findall(r'[A-Za-z0-9]', text))
        total_chars = len(text)
        if total_chars > 0:
            ratio = meaningful_chars / total_chars
            score += ratio * 20
        
        # Question pattern bonus
        question_patterns = [
            r'\b\d+[\.\)]\s*[A-Z]',
            r'\bwhich\b|\bwhat\b|\bhow\b|\bwhy\b',
            r'\b[a-d][\.\)]\s*',
            r'\bfollowing\b|\bcorrect\b|\btrue\b|\bfalse\b'
        ]
        
        for pattern in question_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.5
        
        # NEET subject keywords bonus
        subject_keywords = ['cell', 'atom', 'molecule', 'reaction', 'force', 'energy', 'gene', 'protein']
        for keyword in subject_keywords:
            if keyword in text.lower():
                score += 1
        
        return score

    def _log_extraction_success(self, method, text_length, extraction_time):
        """Log successful extraction for performance monitoring"""
        self._extraction_stats['successful_extractions'] += 1
        self._extraction_stats[f'{method}_successes'] += 1
        self._extraction_stats['total_extraction_time'] += extraction_time
        
        if self.debug_mode:
            print(f"✅ {method} extraction: {text_length} chars in {extraction_time:.2f}s")

    def parse_questions_supreme(self, text):
        """Supreme question parsing with advanced AI validation and multiple strategies"""
        if not text or len(text.strip()) < 50:
            return []

        start_time = time.time()
        
        # Phase 1: Multi-strategy parsing
        parsing_results = []
        
        # Strategy 1: Advanced pattern-based parsing
        pattern_questions = self._parse_with_advanced_patterns(text)
        if pattern_questions:
            parsing_results.append(('advanced_patterns', pattern_questions))
        
        # Strategy 2: Block-based parsing
        block_questions = self._parse_with_block_analysis(text)
        if block_questions:
            parsing_results.append(('block_analysis', block_questions))
        
        # Strategy 3: Machine learning enhanced parsing
        ml_questions = self._parse_with_ml_enhancement(text)
        if ml_questions:
            parsing_results.append(('ml_enhanced', ml_questions))
        
        # Strategy 4: Subject-aware parsing
        subject_questions = self._parse_with_subject_awareness(text)
        if subject_questions:
            parsing_results.append(('subject_aware', subject_questions))
        
        # Phase 2: Select best parsing result
        best_questions = self._select_best_parsing_result(parsing_results)
        
        # Phase 3: Advanced validation and enhancement
        validated_questions = self._validate_and_enhance_questions(best_questions)
        
        # Phase 4: Final quality control
        final_questions = self._final_quality_control(validated_questions)
        
        parsing_time = time.time() - start_time
        
        if self.debug_mode:
            print(f"🚀 Supreme parsing completed in {parsing_time:.2f}s")
            print(f"📊 Strategies tried: {len(parsing_results)}")
            print(f"📝 Final questions: {len(final_questions)}")
        
        return final_questions

    def _parse_with_advanced_patterns(self, text):
        """Parse using advanced pattern recognition"""
        questions = []
        lines = text.split('\n')
        current_question = None
        
        # Find optimal starting position
        start_idx = self._find_optimal_start_position(lines)
        
        i = start_idx
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Check for question start patterns
            question_match = self._match_question_patterns(line)
            if question_match:
                # Save previous question
                if current_question and self._is_valid_question_advanced(current_question):
                    questions.append(current_question)
                
                # Start new question
                q_num, q_text = question_match
                current_question = {
                    'text': q_text,
                    'options': [],
                    'line_number': i,
                    'question_number': q_num,
                    'confidence': 0.7,
                    'parsing_method': 'advanced_patterns'
                }
                
            elif current_question:
                # Check for option patterns
                option_match = self._match_option_patterns(line)
                if option_match:
                    opt_letter, opt_text = option_match
                    current_question['options'].append(opt_text)
                elif len(line) > 10 and len(current_question['options']) == 0:
                    # Continuation of question text
                    current_question['text'] += ' ' + line
            
            i += 1
        
        # Add final question
        if current_question and self._is_valid_question_advanced(current_question):
            questions.append(current_question)
        
        return questions

    def _find_optimal_start_position(self, lines):
        """Find optimal starting position using multiple heuristics"""
        question_density_scores = []
        
        # Analyze chunks of text for question density
        chunk_size = 20
        for start in range(0, min(200, len(lines)), chunk_size):
            chunk = lines[start:start + chunk_size]
            score = self._calculate_question_density(chunk)
            question_density_scores.append((start, score))
        
        # Find the position with highest question density
        if question_density_scores:
            best_start, best_score = max(question_density_scores, key=lambda x: x[1])
            if best_score > 0.3:  # Threshold for good question density
                return best_start
        
        # Fallback to looking for first clear question pattern
        for i, line in enumerate(lines[:150]):
            if self._match_question_patterns(line.strip()):
                return i
        
        return min(30, len(lines) // 10)  # Default fallback

    def _calculate_question_density(self, lines):
        """Calculate density of question-like patterns in text chunk"""
        if not lines:
            return 0
        
        question_indicators = 0
        total_lines = len(lines)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for question number patterns
            if re.match(r'^\d+[\.\)]\s*[A-Za-z]', line):
                question_indicators += 2
            
            # Check for question words
            if any(word in line.lower() for word in 
                   ['what', 'which', 'how', 'why', 'when', 'where']):
                question_indicators += 1
            
            # Check for option patterns
            if re.match(r'^\s*[a-d][\.\)]\s*', line, re.IGNORECASE):
                question_indicators += 0.5
        
        return question_indicators / max(1, total_lines)

    def _match_question_patterns(self, line):
        """Match line against question patterns"""
        for pattern in self.advanced_patterns['question_starters']:
            match = pattern.match(line)
            if match:
                # Extract question number and text
                matched_part = match.group(1)
                remaining_text = line[len(matched_part):].strip()
                
                if len(remaining_text) > 8:  # Ensure substantial question text
                    # Try to extract question number
                    q_num_match = re.search(r'(\d+)', matched_part)
                    q_num = int(q_num_match.group(1)) if q_num_match else len(line)
                    
                    if 1 <= q_num <= 300:  # Reasonable question number range
                        return q_num, remaining_text
        
        return None

    def _match_option_patterns(self, line):
        """Match line against option patterns"""
        for pattern in self.advanced_patterns['option_markers']:
            match = pattern.match(line)
            if match and len(match.groups()) >= 2:
                opt_letter = match.group(1)
                opt_text = match.group(2).strip()
                
                if len(opt_text) > 2:  # Ensure meaningful option text
                    return opt_letter, opt_text
        
        return None

    def _is_valid_question_advanced(self, question):
        """Advanced question validation"""
        if not question or not question.get('text'):
            return False
        
        text = question['text'].strip()
        options = question.get('options', [])
        
        # Basic length check
        if len(text.split()) < 4:
            return False
        
        # Content quality check
        confidence = self._calculate_question_confidence_advanced(text, options)
        question['confidence'] = confidence
        
        return confidence >= 0.5

    def _calculate_question_confidence_advanced(self, text, options):
        """Calculate advanced confidence score for question"""
        confidence = 0.0
        text_lower = text.lower()
        
        # Question word indicators
        question_words = ['what', 'which', 'how', 'why', 'when', 'where', 
                         'explain', 'describe', 'calculate', 'find', 'determine']
        if any(word in text_lower for word in question_words):
            confidence += 0.3
        
        # Question mark
        if '?' in text:
            confidence += 0.2
        
        # Length appropriateness
        word_count = len(text.split())
        if 5 <= word_count <= 50:
            confidence += 0.2
        elif word_count > 50:
            confidence += 0.1
        
        # Options quality
        if len(options) >= 2:
            confidence += 0.15
            if len(options) >= 4:
                confidence += 0.1
        
        # Subject matter relevance
        subject_relevance = self._calculate_subject_relevance(text)
        confidence += subject_relevance * 0.15
        
        # Penalize instruction-like content
        instruction_words = ['mark', 'hour', 'duration', 'booklet', 'sheet', 'candidate']
        if any(word in text_lower for word in instruction_words):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))

    def _calculate_subject_relevance(self, text):
        """Calculate relevance to NEET subjects"""
        text_lower = text.lower()
        relevance_score = 0.0
        total_subjects = 0
        
        for subject, categories in self.advanced_patterns['subject_keywords'].items():
            subject_score = 0.0
            for category, keywords in categories.items():
                keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
                subject_score += keyword_matches
            
            if subject_score > 0:
                relevance_score += min(subject_score / 5, 1.0)  # Normalize
                total_subjects += 1
        
        return relevance_score / max(1, total_subjects)

    def _parse_with_block_analysis(self, text):
        """Parse using intelligent block analysis"""
        # Split text into logical blocks
        blocks = re.split(r'\n\s*\n\s*\n+', text)
        questions = []
        
        for block_idx, block in enumerate(blocks):
            block_lines = [line.strip() for line in block.split('\n') if line.strip()]
            
            if len(block_lines) < 2:
                continue
            
            # Analyze block for question structure
            block_questions = self._analyze_block_for_questions(block_lines, block_idx)
            questions.extend(block_questions)
        
        return questions

    def _analyze_block_for_questions(self, lines, block_idx):
        """Analyze a block of lines for question structure"""
        questions = []
        
        # Look for question start in first few lines
        for i, line in enumerate(lines[:3]):
            question_match = self._match_question_patterns(line)
            if question_match:
                q_num, q_text = question_match
                
                # Collect remaining lines as options or question continuation
                options = []
                question_continuation = []
                
                for remaining_line in lines[i+1:]:
                    option_match = self._match_option_patterns(remaining_line)
                    if option_match:
                        opt_letter, opt_text = option_match
                        options.append(opt_text)
                    elif len(options) == 0 and len(remaining_line) > 10:
                        question_continuation.append(remaining_line)
                
                # Combine question text
                full_question_text = q_text
                if question_continuation:
                    full_question_text += ' ' + ' '.join(question_continuation)
                
                # Create question object
                question = {
                    'text': full_question_text,
                    'options': options,
                    'line_number': block_idx * 10 + i,
                    'question_number': q_num,
                    'confidence': 0.8,
                    'parsing_method': 'block_analysis'
                }
                
                if self._is_valid_question_advanced(question):
                    questions.append(question)
                
                break  # Only one question per block
        
        return questions

    def _parse_with_ml_enhancement(self, text):
        """Parse using machine learning enhancement"""
        # This is a placeholder for more advanced ML-based parsing
        # For now, use enhanced pattern matching with ML-like scoring
        
        lines = text.split('\n')
        questions = []
        
        # Use TF-IDF to identify important lines
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) < 10:
            return questions
        
        try:
            # Simple TF-IDF to identify question-like lines
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            line_vectors = vectorizer.fit_transform(non_empty_lines)
            
            # Score lines based on TF-IDF and question patterns
            question_scores = []
            for i, line in enumerate(non_empty_lines):
                base_score = line_vectors[i].sum()
                pattern_score = self._score_line_as_question(line)
                total_score = base_score * 0.3 + pattern_score * 0.7
                question_scores.append((i, line, total_score))
            
            # Select top scoring lines as potential questions
            question_scores.sort(key=lambda x: x[2], reverse=True)
            
            for i, line, score in question_scores[:50]:  # Top 50 candidates
                if score > 0.5:  # Threshold
                    question_match = self._match_question_patterns(line)
                    if question_match:
                        q_num, q_text = question_match
                        
                        # Look for options in nearby lines
                        options = self._find_options_near_line(non_empty_lines, i)
                        
                        question = {
                            'text': q_text,
                            'options': options,
                            'line_number': i,
                            'question_number': q_num,
                            'confidence': min(score, 0.9),
                            'parsing_method': 'ml_enhanced'
                        }
                        
                        if self._is_valid_question_advanced(question):
                            questions.append(question)
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ ML enhancement failed: {str(e)}")
        
        return questions

    def _score_line_as_question(self, line):
        """Score a line's likelihood of being a question"""
        score = 0.0
        line_lower = line.lower()
        
        # Question patterns
        if re.match(r'^\d+[\.\)]\s*', line):
            score += 0.4
        
        # Question words
        question_words = ['what', 'which', 'how', 'why', 'when', 'where']
        if any(word in line_lower for word in question_words):
            score += 0.3
        
        # Question mark
        if '?' in line:
            score += 0.2
        
        # Length appropriateness
        word_count = len(line.split())
        if 8 <= word_count <= 30:
            score += 0.2
        
        # Subject relevance
        score += self._calculate_subject_relevance(line) * 0.1
        
        return score

    def _find_options_near_line(self, lines, line_idx):
        """Find option lines near a question line"""
        options = []
        
        # Look in next 10 lines for options
        for i in range(line_idx + 1, min(len(lines), line_idx + 11)):
            option_match = self._match_option_patterns(lines[i])
            if option_match:
                opt_letter, opt_text = option_match
                options.append(opt_text)
            elif len(options) > 0:
                # Stop looking once we've found options and hit a non-option
                break
        
        return options

    def _parse_with_subject_awareness(self, text):
        """Parse with awareness of NEET subject patterns"""
        questions = []
        
        # Identify subject sections
        subject_sections = self._identify_subject_sections(text)
        
        for subject, section_text in subject_sections.items():
            if not section_text:
                continue
                
            # Parse each subject section with subject-specific patterns
            subject_questions = self._parse_subject_section(section_text, subject)
            questions.extend(subject_questions)
        
        return questions

    def _identify_subject_sections(self, text):
        """Identify different subject sections in the text"""
        sections = {'physics': '', 'chemistry': '', 'biology': ''}
        
        # Look for subject headers
        subject_patterns = {
            'physics': re.compile(r'(?:physics|mechanical|thermal|wave|electric|magnetic|optic)', re.IGNORECASE),
            'chemistry': re.compile(r'(?:chemistry|chemical|organic|inorganic|physical chemistry|reaction)', re.IGNORECASE),
            'biology': re.compile(r'(?:biology|biological|botany|zoology|cell|genetic|plant|animal)', re.IGNORECASE)
        }
        
        lines = text.split('\n')
        current_subject = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for subject indicators
            for subject, pattern in subject_patterns.items():
                if pattern.search(line):
                    current_subject = subject
                    break
            
            # Add line to current subject section
            if current_subject:
                sections[current_subject] += line + '\n'
        
        # If no clear subject divisions, analyze content
        if not any(sections.values()):
            subject_scores = {'physics': 0, 'chemistry': 0, 'biology': 0}
            
            for subject, categories in self.advanced_patterns['subject_keywords'].items():
                for category, keywords in categories.items():
                    for keyword in keywords:
                        subject_scores[subject] += text.lower().count(keyword)
            
            # Assign entire text to dominant subject
            dominant_subject = max(subject_scores, key=subject_scores.get)
            sections[dominant_subject] = text
        
        return sections

    def _parse_subject_section(self, section_text, subject):
        """Parse a subject-specific section"""
        # Use standard parsing with subject-specific enhancements
        questions = self._parse_with_advanced_patterns(section_text)
        
        # Enhance with subject-specific confidence adjustments
        for question in questions:
            subject_relevance = self._calculate_subject_specific_relevance(
                question['text'], subject)
            question['confidence'] *= (0.8 + subject_relevance * 0.4)
            question['subject'] = subject
        
        return questions

    def _calculate_subject_specific_relevance(self, text, subject):
        """Calculate relevance to specific subject"""
        if subject not in self.advanced_patterns['subject_keywords']:
            return 0.5
        
        text_lower = text.lower()
        relevance_score = 0.0
        total_categories = 0
        
        for category, keywords in self.advanced_patterns['subject_keywords'][subject].items():
            category_matches = sum(1 for keyword in keywords if keyword in text_lower)
            if category_matches > 0:
                relevance_score += min(category_matches / len(keywords), 1.0)
                total_categories += 1
        
        return relevance_score / max(1, total_categories)

    def _select_best_parsing_result(self, parsing_results):
        """Select the best parsing result from multiple strategies"""
        if not parsing_results:
            return []
        
        # Score each result
        scored_results = []
        for method, questions in parsing_results:
            if not questions:
                continue
                
            # Calculate overall quality score
            total_confidence = sum(q.get('confidence', 0.5) for q in questions)
            avg_confidence = total_confidence / len(questions)
            
            # Bonus for having appropriate number of questions
            count_score = 1.0
            if 150 <= len(questions) <= 200:
                count_score = 1.2
            elif 100 <= len(questions) <= 149:
                count_score = 1.1
            elif len(questions) > 200:
                count_score = 0.9
            elif len(questions) < 50:
                count_score = 0.7
            
            overall_score = avg_confidence * count_score * len(questions) / 100
            scored_results.append((overall_score, method, questions))
        
        if scored_results:
            # Return the highest scoring result
            best_score, best_method, best_questions = max(scored_results)
            
            if self.debug_mode:
                print(f"🏆 Best parsing method: {best_method} (score: {best_score:.2f})")
            
            return best_questions
        
        return []

    def _validate_and_enhance_questions(self, questions):
        """Validate and enhance parsed questions"""
        if not questions:
            return questions
        
        enhanced_questions = []
        
        for question in questions:
            # Deep validation
            if self._deep_validate_question(question):
                # Enhancement
                enhanced_question = self._enhance_question(question)
                enhanced_questions.append(enhanced_question)
        
        return enhanced_questions

    def _deep_validate_question(self, question):
        """Perform deep validation of question quality"""
        if not question or not question.get('text'):
            return False
        
        text = question['text'].strip()
        options = question.get('options', [])
        
        # Length validation
        if len(text.split()) < 3 or len(text) > 1000:
            return False
        
        # Content validation
        if not self._has_meaningful_content(text):
            return False
        
        # Option validation
        if len(options) > 6:  # Too many options
            return False
        
        # Instruction filtering
        if self._looks_like_instruction(text):
            return False
        
        return True

    def _has_meaningful_content(self, text):
        """Check if text has meaningful question content"""
        text_lower = text.lower()
        
        # Must have some alphabetic content
        if not re.search(r'[a-zA-Z]{3,}', text):
            return False
        
        # Should not be mostly symbols or numbers
        alphanumeric = len(re.findall(r'[a-zA-Z0-9]', text))
        total_chars = len(text)
        if total_chars > 0 and alphanumeric / total_chars < 0.7:
            return False
        
        return True

    def _looks_like_instruction(self, text):
        """Check if text looks like an instruction rather than a question"""
        text_lower = text.lower()
        
        instruction_indicators = [
            'mark each question', 'time allowed', 'duration',
            'read the following', 'instructions', 'note:',
            'answer sheet', 'omr', 'candidate'
        ]
        
        return any(indicator in text_lower for indicator in instruction_indicators)

    def _enhance_question(self, question):
        """Enhance question with additional metadata"""
        enhanced = question.copy()
        
        # Add difficulty estimation
        enhanced['difficulty'] = self._estimate_difficulty(question['text'])
        
        # Add subject classification
        if 'subject' not in enhanced:
            enhanced['subject'] = self._classify_subject(question['text'])
        
        # Add quality score
        enhanced['quality_score'] = self._calculate_quality_score(question)
        
        # Normalize confidence
        enhanced['confidence'] = max(0.3, min(0.95, enhanced.get('confidence', 0.7)))
        
        return enhanced

    def _estimate_difficulty(self, text):
        """Estimate question difficulty"""
        text_lower = text.lower()
        
        # Count indicators for each difficulty level
        easy_indicators = self.advanced_patterns['difficulty_indicators']['easy']
        moderate_indicators = self.advanced_patterns['difficulty_indicators']['moderate']
        tough_indicators = self.advanced_patterns['difficulty_indicators']['tough']
        
        easy_score = sum(1 for indicator in easy_indicators if indicator in text_lower)
        moderate_score = sum(1 for indicator in moderate_indicators if indicator in text_lower)
        tough_score = sum(1 for indicator in tough_indicators if indicator in text_lower)
        
        # Additional heuristics
        if len(text.split()) < 10:
            easy_score += 1
        elif len(text.split()) > 25:
            tough_score += 1
        
        if '?' in text:
            easy_score += 0.5
        
        # Determine difficulty
        if tough_score > moderate_score and tough_score > easy_score:
            return 'tough'
        elif moderate_score > easy_score:
            return 'moderate'
        else:
            return 'easy'

    def _classify_subject(self, text):
        """Classify question subject"""
        text_lower = text.lower()
        subject_scores = {}
        
        for subject, categories in self.advanced_patterns['subject_keywords'].items():
            score = 0
            for category, keywords in categories.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1
            subject_scores[subject] = score
        
        if subject_scores:
            return max(subject_scores, key=subject_scores.get)
        
        return 'unknown'

    def _calculate_quality_score(self, question):
        """Calculate overall quality score for question"""
        score = 0.0
        
        # Base confidence
        score += question.get('confidence', 0.5) * 0.4
        
        # Text quality
        text_length = len(question['text'].split())
        if 8 <= text_length <= 30:
            score += 0.2
        elif 5 <= text_length <= 50:
            score += 0.1
        
        # Options quality
        options_count = len(question.get('options', []))
        if options_count >= 4:
            score += 0.2
        elif options_count >= 2:
            score += 0.1
        
        # Subject relevance
        if question.get('subject') != 'unknown':
            score += 0.1
        
        # Difficulty appropriateness
        if question.get('difficulty') in ['easy', 'moderate', 'tough']:
            score += 0.1
        
        return min(1.0, score)

    def _final_quality_control(self, questions):
        """Final quality control and optimization"""
        if not questions:
            return questions
        
        # Sort by quality and confidence
        questions.sort(key=lambda q: (q.get('quality_score', 0.5), q.get('confidence', 0.5)), reverse=True)
        
        # Remove duplicates
        unique_questions = []
        seen_texts = set()
        
        for question in questions:
            text_key = question['text'][:100].lower().strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_questions.append(question)
        
        # Limit to reasonable number
        final_questions = unique_questions[:200]  # Maximum reasonable for NEET
        
        # Final validation pass
        validated_final = [q for q in final_questions if q.get('confidence', 0) >= 0.3]
        
        return validated_final

    # Compatibility methods for existing interface
    def extract_text_simple(self, pdf_path):
        """Simple extraction method for backward compatibility"""
        return self.extract_text_supreme(pdf_path)

    def parse_questions_smart(self, text):
        """Smart parsing method for backward compatibility"""
        return self.parse_questions_supreme(text)

    def find_patterns_in_text(self, text):
        """Find patterns method for model training"""
        questions = []
        instructions = []
        options = []
        
        parsed_questions = self.parse_questions_supreme(text)
        
        for q in parsed_questions:
            questions.append(q['text'])
            options.extend(q.get('options', []))
        
        # Look for instruction patterns
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if self._looks_like_instruction(line):
                instructions.append(line)
        
        return questions, instructions, options

    def learn_patterns(self, questions, instructions, options):
        """Learn patterns from samples - enhanced version"""
        # Extract enhanced keywords
        self.question_keywords = self._extract_enhanced_keywords(questions)
        self.instruction_keywords = self._extract_enhanced_keywords(instructions)
        
        # Build enhanced pattern recognition rules
        self.question_patterns = self._build_enhanced_question_patterns(questions)
        self.instruction_patterns = self._build_enhanced_instruction_patterns(instructions)
        self.option_patterns = self._build_enhanced_option_patterns(options)
        
        # Store enhanced statistics
        self.model_data = {
            'question_keywords': list(self.question_keywords),
            'instruction_keywords': list(self.instruction_keywords),
            'total_samples': len(questions) + len(instructions) + len(options),
            'question_samples': len(questions),
            'instruction_samples': len(instructions),
            'option_samples': len(options),
            'advanced_patterns': self.advanced_patterns,
            'extraction_stats': dict(self._extraction_stats)
        }

    def _extract_enhanced_keywords(self, samples):
        """Extract enhanced keywords with TF-IDF scoring"""
        if not samples:
            return set()
        
        try:
            # Use TF-IDF for better keyword extraction
            vectorizer = TfidfVectorizer(
                max_features=100, 
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            tfidf_matrix = vectorizer.fit_transform(samples)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return set(keyword for keyword, score in keyword_scores[:50])
            
        except Exception:
            # Fallback to simple frequency-based extraction
            all_text = ' '.join(samples).lower()
            words = re.findall(r'\b[a-z]{4,}\b', all_text)
            word_freq = Counter(words)
            
            # Filter common words
            stop_words = {'this', 'that', 'with', 'from', 'have', 'will', 'they', 'their', 'were', 'been'}
            filtered_words = {word for word, freq in word_freq.items() 
                            if word not in stop_words and freq >= 2}
            
            return filtered_words

    def _build_enhanced_question_patterns(self, questions):
        """Build enhanced patterns for question identification"""
        patterns = []
        
        if not questions:
            return patterns
        
        # Analyze actual question samples
        question_starters = Counter()
        question_structures = Counter()
        
        for question in questions[:100]:  # Analyze first 100 samples
            words = question.lower().split()
            if words:
                question_starters[words[0]] += 1
            
            # Analyze structure
            if '?' in question:
                question_structures['interrogative'] += 1
            if any(word in question.lower() for word in ['which', 'what', 'how']):
                question_structures['wh_question'] += 1
        
        # Build patterns based on analysis
        common_starters = [starter for starter, count in question_starters.most_common(20) 
                          if count >= 2]
        
        for starter in common_starters:
            patterns.append({
                'type': 'starter',
                'pattern': starter,
                'confidence': 0.8
            })
        
        # Add structure-based patterns
        for structure, count in question_structures.items():
            if count >= 5:
                patterns.append({
                    'type': 'structure',
                    'pattern': structure,
                    'confidence': 0.7
                })
        
        return patterns

    def _build_enhanced_instruction_patterns(self, instructions):
        """Build enhanced instruction patterns"""
        patterns = []
        
        if not instructions:
            return patterns
        
        # Extract common instruction keywords
        instruction_keywords = Counter()
        for instruction in instructions:
            words = instruction.lower().split()
            instruction_keywords.update(words)
        
        # Build patterns from most common keywords
        common_keywords = [word for word, count in instruction_keywords.most_common(30) 
                          if count >= 2]
        
        for keyword in common_keywords:
            patterns.append({
                'type': 'keyword',
                'pattern': keyword,
                'confidence': 0.9
            })
        
        return patterns

    def _build_enhanced_option_patterns(self, options):
        """Build enhanced option patterns"""
        patterns = []
        
        if not options:
            return patterns
        
        # Analyze option markers
        option_markers = Counter()
        for option in options:
            # Look for option markers at the beginning
            match = re.match(r'^[\(\[]?[a-dA-D1-4][\)\.\]]\s*', option)
            if match:
                option_markers[match.group(0).strip()] += 1
        
        # Build patterns from common markers
        for marker, count in option_markers.most_common(10):
            if count >= 2:
                patterns.append({
                    'type': 'marker',
                    'pattern': marker,
                    'confidence': 0.95
                })
        
        return patterns

    def save_model(self, filename='smart_pdf_model_supreme.pkl'):
        """Save the enhanced model"""
        model_data = {
            'question_patterns': self.question_patterns,
            'instruction_patterns': self.instruction_patterns,
            'option_patterns': self.option_patterns,
            'question_keywords': self.question_keywords,
            'instruction_keywords': self.instruction_keywords,
            'model_data': self.model_data,
            'advanced_patterns': self.advanced_patterns,
            'extraction_stats': dict(self._extraction_stats),
            'version': '2.0_supreme'
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.debug_mode:
            print(f"💾 Supreme model saved to {filename}")

    def load_model(self, filename='smart_pdf_model_supreme.pkl'):
        """Load enhanced model with backward compatibility"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load basic patterns
            self.question_patterns = model_data.get('question_patterns', [])
            self.instruction_patterns = model_data.get('instruction_patterns', [])
            self.option_patterns = model_data.get('option_patterns', [])
            self.question_keywords = model_data.get('question_keywords', set())
            self.instruction_keywords = model_data.get('instruction_keywords', set())
            self.model_data = model_data.get('model_data', {})
            
            # Load advanced patterns if available
            if 'advanced_patterns' in model_data:
                loaded_patterns = model_data['advanced_patterns']
                for key, value in loaded_patterns.items():
                    if key in self.advanced_patterns:
                        self.advanced_patterns[key] = value
            
            # Load extraction stats if available
            if 'extraction_stats' in model_data:
                self._extraction_stats.update(model_data['extraction_stats'])
            
            version = model_data.get('version', '1.0')
            if self.debug_mode:
                print(f"✅ Supreme model loaded from {filename} (version: {version})")
            
            return True
            
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️ Could not load model from {filename}: {str(e)}")
            return False

    def analyze_multiple_pdfs(self, pdf_directory='QP'):
        """Analyze multiple PDFs to learn patterns - enhanced version"""
        if not os.path.exists(pdf_directory):
            if self.debug_mode:
                print(f"⚠️ PDF directory '{pdf_directory}' not found")
            return
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            if self.debug_mode:
                print(f"⚠️ No PDF files found in '{pdf_directory}'")
            return
        
        if self.debug_mode:
            print(f"🔄 Analyzing {len(pdf_files)} PDFs for pattern learning...")
        
        all_questions = []
        all_instructions = []
        all_options = []
        
        for pdf_file in pdf_files[:10]:  # Limit to first 10 PDFs for training
            pdf_path = os.path.join(pdf_directory, pdf_file)
            
            try:
                if self.debug_mode:
                    print(f"📖 Processing {pdf_file}...")
                
                text = self.extract_text_supreme(pdf_path)
                if text:
                    questions, instructions, options = self.find_patterns_in_text(text)
                    all_questions.extend(questions)
                    all_instructions.extend(instructions)
                    all_options.extend(options)
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error processing {pdf_file}: {str(e)}")
                continue
        
        if all_questions or all_instructions or all_options:
            self.learn_patterns(all_questions, all_instructions, all_options)
            self.save_model()
            
            if self.debug_mode:
                print(f"✅ Pattern learning completed!")
                print(f"   Questions learned from: {len(all_questions)}")
                print(f"   Instructions learned from: {len(all_instructions)}")
                print(f"   Options learned from: {len(all_options)}")
        else:
            if self.debug_mode:
                print("⚠️ No patterns could be learned from the PDFs")

    def extract_questions_from_pdf(self, pdf_path):
        """Extract questions with comprehensive reporting"""
        if self.debug_mode:
            print(f"🔍 Extracting questions from {os.path.basename(pdf_path)}...")
        
        start_time = time.time()
        
        # Extract text
        text = self.extract_text_supreme(pdf_path)
        if not text:
            if self.debug_mode:
                print("❌ No text could be extracted")
            return []
        
        # Parse questions
        questions = self.parse_questions_supreme(text)
        
        extraction_time = time.time() - start_time
        
        if self.debug_mode:
            print(f"✅ Extracted {len(questions)} questions in {extraction_time:.2f}s")
            
            if questions:
                avg_confidence = np.mean([q.get('confidence', 0.5) for q in questions])
                print(f"📊 Average confidence: {avg_confidence:.2f}")
                
                # Subject distribution
                subjects = Counter(q.get('subject', 'unknown') for q in questions)
                print(f"📚 Subject distribution: {dict(subjects)}")
                
                # Difficulty distribution
                difficulties = Counter(q.get('difficulty', 'unknown') for q in questions)
                print(f"🎯 Difficulty distribution: {dict(difficulties)}")
        
        return questions

def main():
    """Enhanced main function for testing"""
    analyzer = SmartPDFAnalyzer(debug_mode=True)
    
    # Load or train model
    if not analyzer.load_model():
        print("🔄 No existing model found. Training from PDFs...")
        analyzer.analyze_multiple_pdfs()
    
    # Test extraction
    test_pdfs = ['QP/20all.pdf', 'QP/21all.pdf', 'QP/22all.pdf','QP/6Dgth6XtHIp10mI7IDEutu0emP2CVaj4S0suUpOP.pdf']
    
    for test_pdf in test_pdfs:
        if os.path.exists(test_pdf):
            print(f"\n{'='*60}")
            questions = analyzer.extract_questions_from_pdf(test_pdf)
            
            if questions:
                print(f"\n📋 Sample questions from {os.path.basename(test_pdf)}:")
                
                # Show top 3 questions by confidence
                sorted_questions = sorted(questions, key=lambda x: x.get('confidence', 0), reverse=True)
                
                for i, q in enumerate(sorted_questions[:3]):
                    print(f"\n🔢 Question {i+1}:")
                    print(f"   📝 Text: {q['text'][:120]}...")
                    print(f"   📊 Confidence: {q.get('confidence', 0):.2f}")
                    print(f"   📚 Subject: {q.get('subject', 'unknown')}")
                    print(f"   🎯 Difficulty: {q.get('difficulty', 'unknown')}")
                    print(f"   🔧 Method: {q.get('parsing_method', 'unknown')}")
                    
                    if q.get('options'):
                        print(f"   📋 Options: {len(q['options'])} found")
                        for j, opt in enumerate(q['options'][:2]):
                            print(f"      {chr(65+j)}. {opt[:60]}...")
            
            print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
