"""
Enhanced HIPAA Regulations Parser for Maximum RAG Accuracy

This parser addresses specific issues identified in testing:
1. Better distinction between Privacy Rule (164.E) and Security Rule (164.C)
2. Improved semantic chunking for precise retrieval
3. Enhanced metadata for exact citation matching
4. Context-aware section splitting

Target: 100% accuracy on evaluation questions
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedSectionMetadata:
    """Enhanced metadata for precise RAG retrieval."""
    section_id: str
    section_type: str  
    section_title: str
    full_reference: str
    cfr_citation: str
    parent_section: Optional[str]
    hierarchy_level: int
    
    # Enhanced categorization
    regulation_domain: str  # 'privacy', 'security', 'transactions', 'general', 'penalties'
    content_categories: Set[str]  # 'definitions', 'requirements', 'penalties', 'exceptions'
    key_concepts: Set[str]  # Primary concepts for semantic matching
    
    # Structural information
    word_count: int
    character_count: int
    chunk_index: int
    total_chunks: int
    
    # Citation helpers
    exact_quote_ranges: List[Tuple[int, int]]  # Character ranges for exact quotes
    subsection_map: Dict[str, str]  # subsection -> description


@dataclass
class EnhancedChunk:
    """Enhanced chunk with comprehensive metadata."""
    chunk_id: int
    content: str
    metadata: EnhancedSectionMetadata
    embedding: Optional[List[float]] = None


class EnhancedHIPAAParser:
    """
    Enhanced parser targeting 100% accuracy on HIPAA QA tasks.
    
    Key improvements:
    - Precise Privacy vs Security rule distinction
    - Context-aware semantic chunking
    - Enhanced metadata for exact matching
    - Better cross-reference handling
    """
    
    def __init__(self):
        self.section_patterns = self._compile_enhanced_patterns()
        self.domain_classifiers = self._compile_domain_classifiers()
        self.concept_extractors = self._compile_concept_extractors()
        
        # Enhanced tracking
        self.parsing_stats = {
            'total_chunks': 0,
            'privacy_chunks': 0,
            'security_chunks': 0,
            'definition_chunks': 0,
            'penalty_chunks': 0,
            'requirement_chunks': 0
        }
    
    def _compile_enhanced_patterns(self) -> Dict[str, re.Pattern]:
        """Enhanced patterns for precise section identification."""
        return {
            # Core structure patterns
            'part': re.compile(r'^PART\s+(\d+)[—\-]\s*(.+?)(?:\s+\.{3,}\s*\d+)?$', re.IGNORECASE | re.MULTILINE),
            'subpart': re.compile(r'^SUBPART\s+([A-Z])[—\-]\s*(.+?)(?:\s+\.{3,}\s*\d+)?$', re.IGNORECASE | re.MULTILINE),
            'section': re.compile(r'^[§§]\s*(\d+)\.(\d+)\s+(.+?)(?:\s+\.{3,}\s*\d+)?$', re.MULTILINE),
            'subsection_header': re.compile(r'^##\s+[§§]\s*(\d+)\.(\d+)\s+(.+)$', re.MULTILINE),
            
            # Enhanced subsection patterns
            'numbered_subsection': re.compile(r'^\((\d+)\)\s+(.+)', re.MULTILINE),
            'lettered_subsection': re.compile(r'^\(([a-z])\)\s+(.+)', re.MULTILINE),
            'roman_subsection': re.compile(r'^\(([ivxlc]+)\)\s+(.+)', re.MULTILINE | re.IGNORECASE),
            
            # Privacy Rule specific patterns
            'privacy_subpart': re.compile(r'SUBPART\s+E[—\-]\s*PRIVACY\s+OF\s+INDIVIDUALLY\s+IDENTIFIABLE\s+HEALTH\s+INFORMATION', re.IGNORECASE),
            'privacy_section': re.compile(r'[§§]\s*164\.5\d+', re.IGNORECASE),
            
            # Security Rule specific patterns  
            'security_subpart': re.compile(r'SUBPART\s+C[—\-]\s*SECURITY\s+STANDARDS', re.IGNORECASE),
            'security_section': re.compile(r'[§§]\s*164\.3\d+', re.IGNORECASE),
            
            # Citation patterns
            'cfr_full': re.compile(r'45\s+CFR\s+[§§]?\s*(\d+)\.(\d+)', re.IGNORECASE),
            'section_ref': re.compile(r'[§§]\s*(\d+)\.(\d+)(?:\([a-z0-9]+\))*', re.IGNORECASE),
        }
    
    def _compile_domain_classifiers(self) -> Dict[str, Dict[str, re.Pattern]]:
        """Compile patterns to classify regulation domains."""
        return {
            'privacy': {
                'keywords': re.compile(r'\b(?:privacy|protected health information|phi|uses?\s+and\s+disclosures?|authorization|minimum\s+necessary)\b', re.IGNORECASE),
                'sections': re.compile(r'164\.5\d+'),
                'subpart': re.compile(r'subpart\s+e', re.IGNORECASE)
            },
            'security': {
                'keywords': re.compile(r'\b(?:security|safeguards|access\s+control|encryption|integrity|transmission|audit)\b', re.IGNORECASE),
                'sections': re.compile(r'164\.3\d+'),
                'subpart': re.compile(r'subpart\s+c', re.IGNORECASE)
            },
            'penalties': {
                'keywords': re.compile(r'\b(?:civil\s+money\s+penalty|violation|fine|penalty|sanctions?)\b', re.IGNORECASE),
                'sections': re.compile(r'160\.4\d+')
            },
            'general': {
                'keywords': re.compile(r'\b(?:definitions?|applicability|compliance|general\s+provisions)\b', re.IGNORECASE),
                'sections': re.compile(r'160\.1\d+')
            },
            'transactions': {
                'keywords': re.compile(r'\b(?:transactions?|standards|code\s+sets|identifiers?)\b', re.IGNORECASE),
                'sections': re.compile(r'162\.\d+')
            }
        }
    
    def _compile_concept_extractors(self) -> Dict[str, re.Pattern]:
        """Extract key concepts for semantic matching."""
        return {
            'covered_entities': re.compile(r'\b(?:covered\s+entit(?:y|ies)|health\s+plan|health\s+care\s+clearinghouse|health\s+care\s+provider)\b', re.IGNORECASE),
            'business_associates': re.compile(r'\b(?:business\s+associate|business\s+partner)\b', re.IGNORECASE),
            'phi': re.compile(r'\b(?:protected\s+health\s+information|individually\s+identifiable\s+health\s+information|phi|iihi)\b', re.IGNORECASE),
            'minimum_necessary': re.compile(r'\bminimum\s+necessary\b', re.IGNORECASE),
            'authorization': re.compile(r'\bauthorization\b', re.IGNORECASE),
            'disclosure': re.compile(r'\b(?:disclos(?:e|ure)|uses?\s+and\s+disclosures?)\b', re.IGNORECASE),
            'law_enforcement': re.compile(r'\b(?:law\s+enforcement|legal\s+proceedings?|court\s+orders?)\b', re.IGNORECASE),
            'family_members': re.compile(r'\b(?:family\s+members?|relatives?|personal\s+representatives?)\b', re.IGNORECASE),
            'encryption': re.compile(r'\b(?:encrypt(?:ion|ed)?|cryptographic)\b', re.IGNORECASE),
            'breach': re.compile(r'\b(?:breach|security\s+incident)\b', re.IGNORECASE)
        }
    
    def parse_hipaa_regulations(self, input_file: str, output_file: str = None) -> List[EnhancedChunk]:
        """
        Parse HIPAA regulations with enhanced accuracy.
        
        Args:
            input_file: Path to hipaa_regulations.txt
            output_file: Optional output path for enhanced chunks
            
        Returns:
            List of enhanced chunks optimized for RAG
        """
        logger.info(f"🚀 Starting enhanced HIPAA parsing: {input_file}")
        start_time = datetime.now()
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"📖 Read {len(content):,} characters")
        
        # Parse into structured sections
        raw_sections = self._extract_structured_sections(content)
        logger.info(f"📑 Extracted {len(raw_sections)} raw sections")
        
        # Create enhanced chunks
        enhanced_chunks = []
        chunk_id = 0
        
        for section in raw_sections:
            section_chunks = self._create_enhanced_chunks(section, chunk_id)
            enhanced_chunks.extend(section_chunks)
            chunk_id += len(section_chunks)
        
        # Post-process and validate
        enhanced_chunks = self._post_process_chunks(enhanced_chunks)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Created {len(enhanced_chunks)} enhanced chunks in {processing_time:.2f}s")
        logger.info(f"📊 Stats: {self.parsing_stats}")
        
        if output_file:
            self._save_enhanced_chunks(enhanced_chunks, output_file)
        
        return enhanced_chunks
    
    def _extract_structured_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections with enhanced structure detection."""
        sections = []
        
        # Split by major section boundaries
        section_boundaries = [
            r'^PART\s+\d+',
            r'^SUBPART\s+[A-Z]',
            r'^[§§]\s*\d+\.\d+',
            r'^##\s+[§§]\s*\d+\.\d+'
        ]
        
        pattern = '|'.join(f'({p})' for p in section_boundaries)
        boundary_regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
        
        matches = list(boundary_regex.finditer(content))
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            
            section_content = content[start_pos:end_pos].strip()
            if len(section_content) < 50:  # Skip very short sections
                continue
                
            section_info = self._analyze_section_structure(section_content)
            if section_info:
                sections.append(section_info)
        
        return sections
    
    def _analyze_section_structure(self, section_content: str) -> Optional[Dict[str, Any]]:
        """Analyze section structure with enhanced metadata."""
        lines = section_content.split('\n')
        first_line = lines[0].strip()
        
        # Identify section type and metadata
        section_type, section_id, title = self._identify_enhanced_section_type(first_line)
        if not section_type:
            return None
        
        # Classify regulation domain
        domain = self._classify_regulation_domain(section_content)
        
        # Extract key concepts
        concepts = self._extract_key_concepts(section_content)
        
        # Identify content categories
        categories = self._identify_content_categories(section_content)
        
        # Build CFR citation
        cfr_citation = self._build_cfr_citation(section_type, section_id, title)
        
        return {
            'content': section_content,
            'section_type': section_type,
            'section_id': section_id,
            'title': title,
            'domain': domain,
            'concepts': concepts,
            'categories': categories,
            'cfr_citation': cfr_citation,
            'word_count': len(section_content.split()),
            'char_count': len(section_content)
        }
    
    def _identify_enhanced_section_type(self, first_line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Enhanced section type identification."""
        
        # Check for PART
        part_match = self.section_patterns['part'].search(first_line)
        if part_match:
            return 'part', f"PART_{part_match.group(1)}", part_match.group(2).strip()
        
        # Check for SUBPART with specific handling for Privacy/Security
        subpart_match = self.section_patterns['subpart'].search(first_line)
        if subpart_match:
            subpart_id = f"SUBPART_{subpart_match.group(1)}"
            title = subpart_match.group(2).strip()
            
            # Special handling for key subparts
            if 'PRIVACY' in title.upper():
                subpart_id += "_PRIVACY"
            elif 'SECURITY' in title.upper():
                subpart_id += "_SECURITY"
                
            return 'subpart', subpart_id, title
        
        # Check for SECTION
        section_match = self.section_patterns['section'].search(first_line)
        if section_match:
            section_id = f"{section_match.group(1)}.{section_match.group(2)}"
            return 'section', section_id, section_match.group(3).strip()
        
        # Check for subsection header
        subsection_match = self.section_patterns['subsection_header'].search(first_line)
        if subsection_match:
            section_id = f"{subsection_match.group(1)}.{subsection_match.group(2)}"
            return 'section', section_id, subsection_match.group(3).strip()
        
        return None, None, None
    
    def _classify_regulation_domain(self, content: str) -> str:
        """Classify the regulatory domain with enhanced accuracy."""
        scores = {}
        
        for domain, patterns in self.domain_classifiers.items():
            score = 0
            
            # Keyword matching
            if 'keywords' in patterns:
                score += len(patterns['keywords'].findall(content)) * 2
            
            # Section number matching
            if 'sections' in patterns:
                score += len(patterns['sections'].findall(content)) * 5
            
            # Subpart matching
            if 'subpart' in patterns:
                if patterns['subpart'].search(content):
                    score += 10
            
            scores[domain] = score
        
        # Return domain with highest score, default to 'general'
        return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'general'
    
    def _extract_key_concepts(self, content: str) -> Set[str]:
        """Extract key concepts for semantic matching."""
        concepts = set()
        
        for concept, pattern in self.concept_extractors.items():
            if pattern.search(content):
                concepts.add(concept)
        
        return concepts
    
    def _identify_content_categories(self, content: str) -> Set[str]:
        """Identify content categories."""
        categories = set()
        
        # Check for definitions
        if re.search(r'\b(?:means|definition|defined as)\b', content, re.IGNORECASE):
            categories.add('definitions')
        
        # Check for requirements
        if re.search(r'\b(?:must|shall|required|implement)\b', content, re.IGNORECASE):
            categories.add('requirements')
        
        # Check for penalties
        if re.search(r'\b(?:penalty|fine|violation|civil money)\b', content, re.IGNORECASE):
            categories.add('penalties')
        
        # Check for exceptions
        if re.search(r'\b(?:exception|unless|except|does not apply)\b', content, re.IGNORECASE):
            categories.add('exceptions')
        
        return categories
    
    def _build_cfr_citation(self, section_type: str, section_id: str, title: str) -> str:
        """Build proper CFR citation."""
        if section_type == 'part':
            part_num = section_id.replace('PART_', '')
            return f"45 CFR Part {part_num}"
        elif section_type == 'subpart':
            # Extract part and subpart
            if '_' in section_id:
                parts = section_id.split('_')
                if len(parts) >= 2:
                    subpart_letter = parts[1]
                    return f"45 CFR Part 164, Subpart {subpart_letter}"
        elif section_type == 'section':
            return f"45 CFR § {section_id}"
        
        return f"45 CFR {section_id}"
    
    def _create_enhanced_chunks(self, section: Dict[str, Any], start_chunk_id: int) -> List[EnhancedChunk]:
        """Create enhanced chunks with optimal sizing."""
        content = section['content']
        
        # Determine chunk strategy based on content length and type
        if len(content) <= 800:  # Small section - keep as single chunk
            chunks = [content]
        elif section['section_type'] == 'section' and 'definitions' in section['categories']:
            # Definition sections - split by term
            chunks = self._split_definitions_section(content)
        else:
            # Large section - split semantically
            chunks = self._split_section_semantically(content)
        
        enhanced_chunks = []
        for i, chunk_content in enumerate(chunks):
            metadata = self._create_enhanced_metadata(
                section, i, len(chunks), start_chunk_id + i
            )
            
            enhanced_chunks.append(EnhancedChunk(
                chunk_id=start_chunk_id + i,
                content=chunk_content.strip(),
                metadata=metadata
            ))
            
            # Update stats
            self.parsing_stats['total_chunks'] += 1
            if section['domain'] == 'privacy':
                self.parsing_stats['privacy_chunks'] += 1
            elif section['domain'] == 'security':
                self.parsing_stats['security_chunks'] += 1
            
            if 'definitions' in section['categories']:
                self.parsing_stats['definition_chunks'] += 1
            if 'penalties' in section['categories']:
                self.parsing_stats['penalty_chunks'] += 1
            if 'requirements' in section['categories']:
                self.parsing_stats['requirement_chunks'] += 1
        
        return enhanced_chunks
    
    def _split_definitions_section(self, content: str) -> List[str]:
        """Split definitions section by individual terms."""
        chunks = []
        
        # Look for definition patterns
        definition_pattern = re.compile(r'^([A-Z][A-Za-z\s]+?)\s+means\s+(.+?)(?=\n[A-Z][A-Za-z\s]+?\s+means|\Z)', 
                                       re.MULTILINE | re.DOTALL)
        
        matches = list(definition_pattern.finditer(content))
        
        if matches:
            # Split by definitions
            last_end = 0
            for match in matches:
                # Include any intro text before first definition
                if match.start() > last_end:
                    intro = content[last_end:match.start()].strip()
                    if intro and len(intro) > 50:
                        chunks.append(intro)
                
                # Add the definition
                definition_text = match.group(0).strip()
                chunks.append(definition_text)
                last_end = match.end()
            
            # Add any remaining content
            if last_end < len(content):
                remaining = content[last_end:].strip()
                if remaining and len(remaining) > 50:
                    chunks.append(remaining)
        else:
            # No clear definitions pattern - split by paragraphs
            chunks = self._split_by_paragraphs(content)
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def _split_section_semantically(self, content: str) -> List[str]:
        """Split section content semantically."""
        
        # First try to split by subsections
        subsection_pattern = re.compile(r'\n\(([a-z]|\d+)\)\s+', re.MULTILINE)
        subsection_splits = subsection_pattern.split(content)
        
        if len(subsection_splits) > 1:
            chunks = []
            for i in range(0, len(subsection_splits), 2):
                if i + 1 < len(subsection_splits):
                    subsection_id = subsection_splits[i + 1]
                    subsection_content = subsection_splits[i + 2] if i + 2 < len(subsection_splits) else ""
                    chunk = f"({subsection_id}) {subsection_content}".strip()
                    if len(chunk) > 50:
                        chunks.append(chunk)
                elif subsection_splits[i].strip():
                    chunks.append(subsection_splits[i].strip())
            
            return chunks
        else:
            # Fall back to paragraph splitting
            return self._split_by_paragraphs(content)
    
    def _split_by_paragraphs(self, content: str, max_chunk_size: int = 800) -> List[str]:
        """Split content by paragraphs with size limits."""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_enhanced_metadata(self, section: Dict[str, Any], chunk_index: int, 
                                 total_chunks: int, chunk_id: int) -> EnhancedSectionMetadata:
        """Create enhanced metadata for precise retrieval."""
        
        return EnhancedSectionMetadata(
            section_id=section['section_id'],
            section_type=section['section_type'],
            section_title=section['title'],
            full_reference=f"45 CFR {section['section_id']} - {section['title']}",
            cfr_citation=section['cfr_citation'],
            parent_section=None,  # Could be enhanced
            hierarchy_level=self._get_hierarchy_level(section['section_type']),
            regulation_domain=section['domain'],
            content_categories=section['categories'],
            key_concepts=section['concepts'],
            word_count=len(section['content'].split()),
            character_count=len(section['content']),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            exact_quote_ranges=[],  # Could be enhanced
            subsection_map={}  # Could be enhanced
        )
    
    def _get_hierarchy_level(self, section_type: str) -> int:
        """Get hierarchy level for section type."""
        hierarchy_map = {
            'part': 1,
            'subpart': 2,
            'section': 3,
            'subsection': 4,
            'paragraph': 5
        }
        return hierarchy_map.get(section_type, 3)
    
    def _post_process_chunks(self, chunks: List[EnhancedChunk]) -> List[EnhancedChunk]:
        """Post-process chunks for optimization."""
        logger.info("🔧 Post-processing chunks for optimization...")
        
        # Remove very short chunks that don't add value
        filtered_chunks = []
        for chunk in chunks:
            if len(chunk.content.split()) >= 10:  # Minimum 10 words
                filtered_chunks.append(chunk)
            else:
                logger.debug(f"Filtered out short chunk: {chunk.chunk_id}")
        
        # Renumber chunk IDs
        for i, chunk in enumerate(filtered_chunks):
            chunk.chunk_id = i
        
        logger.info(f"📋 Kept {len(filtered_chunks)} of {len(chunks)} chunks after filtering")
        return filtered_chunks
    
    def _save_enhanced_chunks(self, chunks: List[EnhancedChunk], output_file: str):
        """Save enhanced chunks to JSON file."""
        chunk_data = []
        
        for chunk in chunks:
            chunk_dict = {
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'metadata': {
                    'section_id': chunk.metadata.section_id,
                    'section_type': chunk.metadata.section_type,
                    'section_title': chunk.metadata.section_title,
                    'full_reference': chunk.metadata.full_reference,
                    'cfr_citation': chunk.metadata.cfr_citation,
                    'parent_section': chunk.metadata.parent_section,
                    'hierarchy_level': chunk.metadata.hierarchy_level,
                    'regulation_domain': chunk.metadata.regulation_domain,
                    'content_categories': list(chunk.metadata.content_categories),
                    'key_concepts': list(chunk.metadata.key_concepts),
                    'word_count': chunk.metadata.word_count,
                    'character_count': chunk.metadata.character_count,
                    'chunk_index': chunk.metadata.chunk_index,
                    'total_chunks': chunk.metadata.total_chunks
                }
            }
            chunk_data.append(chunk_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(chunks)} enhanced chunks to {output_file}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Enhanced HIPAA Parser for RAG Systems')
    parser.add_argument('input_file', help='Input HIPAA regulations text file')
    parser.add_argument('-o', '--output', help='Output JSON file for enhanced chunks')
    
    args = parser.parse_args()
    
    enhanced_parser = EnhancedHIPAAParser()
    chunks = enhanced_parser.parse_hipaa_regulations(
        args.input_file, 
        args.output or 'enhanced_hipaa_chunks.json'
    )
    
    print(f"✅ Successfully created {len(chunks)} enhanced chunks")
    print(f"📊 Processing stats: {enhanced_parser.parsing_stats}")


if __name__ == "__main__":
    main()