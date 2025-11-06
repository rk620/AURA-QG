import re
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import numpy as np
import faiss
from collections import Counter
import math
import string
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def extract_markdown_text_blocks(markdown_text: str) -> List[str]:
    # Drop Markdown headings
    lines = markdown_text.splitlines()
    filtered_lines = [line for line in lines if not re.match(r'^\s{0,3}#{1,6}\s', line)]

    # Reconstruct the text after removing headings
    text = '\n'.join(lines)

    # Normalize list items for bullets and numbers to be on new lines
    text = re.sub(r'\n\s*-\s*', r'\n- ', text)
    text = re.sub(r'\n\s*(\(?\d+[\.\)])\s*', r'\n\1 ', text)

    blocks = []
    buffer = []
    lines = text.split('\n')

    for line in lines:
        if not line.strip():
            if buffer:
                blocks.append(' '.join(buffer).strip())
                buffer = []
            continue

        # Detect bullet or numbered list item
        is_list_item = re.match(r'^(-|\(?\d+[\.\)])\s+', line.strip())

        if is_list_item:
            if buffer:
                blocks.append(' '.join(buffer).strip())
                buffer = []
            blocks.append(line.rstrip())  # Preserve leading "-" or "1."
        else:
            buffer.append(line.strip())

    if buffer:
        blocks.append(' '.join(buffer).strip())

    # Group consecutive list items into one string block
    final_blocks = []
    temp_list = []

    for block in blocks:
        if re.match(r'^(-|\(?\d+[\.\)])\s+', block.strip()):
            temp_list.append(block.strip())
        else:
            if temp_list:
                final_blocks.append('\n' + '\n'.join(temp_list))
                temp_list = []
            final_blocks.append(block)

    if temp_list:
        final_blocks.append('\n' + '\n'.join(temp_list))

    return final_blocks






def clean_pdf_blocks_and_build_necessary_IUs(blocks: List[str]) -> List[str]:
    cleaned_blocks = []
    precleaned_blocks = []
    necessary_IUs = []
    bullet_or_number_pattern = re.compile(r"^\s*([-*•]|\(?\d+[\.\)])\s+", re.MULTILINE)

    # Better pattern: Remove markdown image blocks and following 'Source'/'Chart' lines until next newline
    image_with_source_pattern = re.compile(
        r"!\[.*?\]\(.*?\)\s*(?:\n\n(?:Source|Chart)[^\n]*)?", re.IGNORECASE
    )
    latex_superscript_pattern = re.compile(r"\$\{\s*\}\^\{.*?\}\$")
    # General table matcher (works on inline tables in block)
    table_pattern_inline = re.compile(
        r"\|(?:[^\n]*\|)+", re.MULTILINE
    )

    # Footnote pattern: [^1] style + definition lines
    footnote_pattern = re.compile(r"\[\^\d+\]:.*(?:\n[^\[\]\n]+)*", re.MULTILINE)
    standalone_footnote_pattern = re.compile(r"^\[\^\d+\]$", re.MULTILINE)
    for block in blocks:
        # Remove tables
        block = table_pattern_inline.sub("", block)

        # Remove image + source/chart references
        block = image_with_source_pattern.sub("", block)

        # Remove footnotes
        block = footnote_pattern.sub("", block)
        block = standalone_footnote_pattern.sub("", block)
        block = latex_superscript_pattern.sub("", block)
        precleaned_blocks.append(block.strip())

    # Normalize bullets and numbering
    for block in precleaned_blocks:
        block = block.replace('\n', ' ').strip()
        block = re.sub(r"\s*•\s+", r"\n• ", block)
        block = re.sub(r"\s*-\s+", r"\n- ", block)
        block = re.sub(r"(?:^|\n)\s*(\d+)\.\s+", r"\n\1. ", block)
        cleaned_blocks.append(block)

    # Remove header-style blocks
    cleaned_blocks = [b for b in cleaned_blocks if not b.strip().startswith("#")]

    # return cleaned_blocks


    

    # Step 2: Merge blocks ending with ':' with the next block
    merged_blocks = []
    i = 0
    while i < len(cleaned_blocks):
        current = cleaned_blocks[i]
        if current.endswith(":") and i + 1 < len(cleaned_blocks):
            merged = f"{current} {cleaned_blocks[i + 1]}"
            merged_blocks.append(merged.strip())
            i += 2
        else:
            merged_blocks.append(current.strip())
            i += 1

    # Step 3: Sentence tokenize and assign subtype
    for block in merged_blocks:
        if bullet_or_number_pattern.search(block.strip()):  # List-like block
            bullet_points = block.strip().split("\n")
            long_bullet_found = any(len(sent_tokenize(bp)) > 2 for bp in bullet_points if bp.strip())

            if long_bullet_found:
                # Treat as paragraph: sentence tokenize
                sentences = sent_tokenize(block)
                for sent in sentences:
                    sent = sent.strip()
                    if sent:
                        necessary_IUs.append({
                            "text": sent,
                            "subtype": "paragraph"
                        })
            else:
                # Treat as list block
                necessary_IUs.append({
                    "text": block.strip(),
                    "subtype": "list"
                })
        else:
            # Normal paragraph: sentence tokenize
            sentences = sent_tokenize(block)
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    necessary_IUs.append({
                        "text": sent,
                        "subtype": "paragraph"
                    })


    return necessary_IUs






def split_list_blocks_to_individual_items_with_parent_ids(
    necessary_IUs_metadata: List[Dict]
) -> List[Tuple[str, int]]:
    fine_grained = []
    bullet_or_number_pattern = re.compile(r"(?:^|\n)\s*(?:[-*•]|\(?\d+[\.\)])\s+")

    for meta in necessary_IUs_metadata:
        text = meta["text"]
        parent_chunk_id = meta["chunk_id"]

        if re.search(r"^\s*(?:[-*•]|\(?\d+[\.\)])\s+", text, re.MULTILINE):
            items = re.split(bullet_or_number_pattern, text)
            items = [item.strip() for item in items if item.strip()]
            fine_grained.extend([(item, parent_chunk_id) for item in items])

    return fine_grained





def extract_sentence_windows_from_paragraphs(necessary_IUs_metadata: List[Dict]) -> List[str]:
    """
    Extract 2- and 3-sentence sliding windows from paragraph-type IUs
    that are contiguous in the metadata list.
    """
    windows = []
    
    # Step 1: Filter out only paragraph-type IUs with their indices
    paragraph_chunks = [(iu["chunk_id"], iu["text"]) for iu in necessary_IUs_metadata if iu["subtype"] == "paragraph"]

    if not paragraph_chunks:
        return []

    # Step 2: Group paragraph chunks with contiguous chunk_ids
    grouped = []
    current_group = [paragraph_chunks[0][1]]
    last_id = paragraph_chunks[0][0]

    for chunk_id, text in paragraph_chunks[1:]:
        if chunk_id == last_id + 1:
            current_group.append(text)
        else:
            grouped.append(current_group)
            current_group = [text]
        last_id = chunk_id
    grouped.append(current_group)

    # Step 3: Tokenize and create sliding windows for each group
    for group in grouped:
        all_sentences = []
        for text in group:
            all_sentences.extend(sent_tokenize(text))
        # 2-sentence windows
        for i in range(len(all_sentences) - 1):
            windows.append(" ".join(all_sentences[i:i+2]))
        # 3-sentence windows
        for i in range(len(all_sentences) - 2):
            windows.append(" ".join(all_sentences[i:i+3]))

    return windows







def generate_optional_window_metadata_cosine(
    optional_windows: List[str],
    necessary_metadata: List[Dict],
    model: SentenceTransformer,
    start_index: int
) -> List[Dict]:
    """
    For each optional window IU, split it into sentences and find semantically similar sentences
    (cosine similarity ≥ 0.85) in the necessary IUs. Build metadata with those linked sentence indices.
    """
    metadata = []
    necessary_texts = [m["text"].strip() for m in necessary_metadata]
    necessary_embeddings = model.encode(necessary_texts, convert_to_tensor=True)

    for i, window_text in enumerate(optional_windows):
        # Split window into individual sentences
        window_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', window_text) if s.strip()]
        linked_ids = set()

        for sentence in window_sentences:
            sentence_emb = model.encode(sentence, convert_to_tensor=True)
            cosine_scores = util.cos_sim(sentence_emb, necessary_embeddings)[0]
            matched_indices = (cosine_scores >= 0.85).nonzero(as_tuple=True)[0].tolist()
            linked_ids.update(matched_indices)

        metadata.append({
            "text": window_text,
            "type": "optional",
            "subtype": "window",
            "chunk_id": start_index + i,
            "parent_chunk_id": sorted(list(linked_ids)),
            
        })

    return metadata










def dynamic_list_refinement(
    question: str,
    top_span: str,
    top_score: float,
    cross_encoder,
    window_size: int = 2,
    stride: int = 1,
) -> Tuple[str, float]:
    """
    Refine the top_span dynamically if it's a 'list' subtype and below the score threshold.
    """

    
    # Split the list block into bullet/numbered items
    list_items = [item.strip() for item in re.split(r"(?:^|\n)\s*(?:[-*•]|\d+\.)\s+", top_span) if item.strip()]

    # Generate sliding windows over the list items
    windows = []
    for i in range(0, len(list_items) - window_size + 1, stride):
        window = list_items[i:i + window_size]
        windows.append(" ".join(window))

    # Optionally handle residual window at the end
    if len(list_items) > window_size and (len(list_items) - window_size) % stride != 0:
        last_window = list_items[-window_size:]
        if " ".join(last_window) not in windows:
            windows.append(" ".join(last_window))

    # Encode and score each window
    cross_inputs = [[question, span] for span in windows]
    if cross_inputs:
        window_scores = cross_encoder.predict(cross_inputs)

        # Find the highest scoring window
        best_idx = int(np.argmax(window_scores))
        best_window = windows[best_idx]
        best_score = window_scores[best_idx]

        # Return refined window only if it outperforms original top span
        if best_score > top_score:
            return best_window, best_score
    return top_span, top_score









def building_faiss_index(all_IUs, metadata, questions, top_k):
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')  # For semantic embeddings
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # For reranking
    span_embeddings = bi_encoder.encode(all_IUs, convert_to_tensor=True)
    # Create FAISS index
    dimension = span_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(span_embeddings.cpu().detach().numpy())

    results = []
    covered_flags = [0 for _ in metadata]  # aligned with metadata

    for question in questions:
        # Step 1: Encode question
        question_embedding = bi_encoder.encode(question, convert_to_tensor=True)
        question_embedding_np = question_embedding.cpu().detach().numpy()

        # Step 2: Retrieve top-k from FAISS
        D, I = faiss_index.search(np.array([question_embedding_np]), top_k)
        retrieved_spans = [all_IUs[i] for i in I[0]]
        retrieved_metadata = [metadata[i] for i in I[0]]

        # Step 3: Rerank with cross-encoder
        cross_inputs = [[question, span] for span in retrieved_spans]
        rerank_scores = cross_encoder.predict(cross_inputs)

        # Step 4: Sort by score
        ranked = sorted(zip(retrieved_spans, rerank_scores, retrieved_metadata), key=lambda x: x[1], reverse=True)
        top_span, top_score, top_metadata = ranked[0]

        # Step 5: Apply dynamic list refinement conditionally
        if top_metadata["type"] == "necessary" and top_metadata["subtype"] == "list" and top_score < 0:
            top_span, top_score = dynamic_list_refinement(
                question=question,
                top_span=top_span,
                top_score=top_score,
                top_span_metadata=top_metadata,
                bi_encoder=bi_encoder,
                cross_encoder=cross_encoder
            )

        is_answerable = top_score >= 0

        results.append({
            "question": question,
            "is_answerable": is_answerable,
            "top_span": top_span,
            "top_score": top_score,
            "all_ranked_spans": [(s, sc) for s, sc, _ in ranked]
        })
        if not is_answerable:
            continue

        if top_span in all_IUs:
            top_index = all_IUs.index(top_span)
            top_meta = metadata[top_index]
        
            covered_flags[top_index] = 1
        
            if top_meta["type"] == "necessary":
                pass
        
            elif top_meta["subtype"] == "window":
                parent_ids = top_meta.get("parent_chunk_id", [])
                if isinstance(parent_ids, int):
                    parent_ids = [parent_ids]
                for idx, meta in enumerate(metadata):
                    if meta["chunk_id"] in parent_ids:
                        covered_flags[idx] = 1
        
            elif top_meta["subtype"] == "bullet":
                parent_id = top_meta.get("parent_chunk_id")
                if isinstance(parent_id, int):
                    for idx, meta in enumerate(metadata):
                        if meta["chunk_id"] == parent_id:
                            covered_flags[idx] = 1
                            break
        else:
            # Handle coverage for refined dynamic spans (optional bullets)
            for idx, meta in enumerate(metadata):
                if meta.get("type") == "optional" and meta.get("subtype") == "bullet":
                    if meta["text"] in top_span:
                        covered_flags[idx] = 1

    return results, covered_flags








def split_questions_by_answerability(results: List[Dict]) -> (List[str], List[str]):
    questions_answerable = []
    questions_unanswerable = []

    for r in results:
        if r["is_answerable"]:
            questions_answerable.append(r["question"])
        else:
            questions_unanswerable.append(r["question"])

    return questions_answerable, questions_unanswerable







def group_questions_by_exact_answer(results: List[Dict]) -> List[Dict]:
    """
    Groups all questions that share the exact same top_span (answer) into one group.
    
    Returns:
        A list of groups, each as a dict:
        {
            "answer": top_span,
            "questions": [q1, q2, ...]
        }
    """
    answer_to_questions = {}

    for entry in results:
        if not entry["is_answerable"]:
            continue  # Skip unanswerable questions

        answer = entry["top_span"].strip()
        question = entry["question"]

        if answer in answer_to_questions:
            answer_to_questions[answer].append(question)
        else:
            answer_to_questions[answer] = [question]

    # Filter: only return groups with more than 1 question (redundant)
    groups = []
    for answer, questions in answer_to_questions.items():
        if len(questions) > 1:
            groups.append({
                "answer": answer,
                "questions": questions
            })

    return groups




def get_wh_template(question: str) -> str:
    helping_verbs = {
        "is", "are", "was", "were", "am", "be", "being", "been",
        "do", "does", "did",
        "have", "has", "had",
        "can", "could", "shall", "should", "will", "would", "may", "might", "must"
    }

    wh_words = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how"
    }

    question = question.lower().strip()
    tokens = question.split()

    if not tokens:
        return "other"

    # Skip leading punctuation-only tokens
    i = 0
    while i < len(tokens) and all(c in string.punctuation for c in tokens[i]):
        i += 1

    if i >= len(tokens):
        return "other"

    # Check if first token is a helping verb
    if tokens[i] in helping_verbs:
        return "other"

    # Check for "how many"/"how much"
    for j in range(i, len(tokens) - 1):
        if tokens[j] == "how" and tokens[j + 1] in {"many", "much"}:
            return f"how {tokens[j + 1]}"

    # Check for any WH-word in the sentence
    for token in tokens[i:]:
        if token in wh_words:
            return token

    # If no WH-word found, return the first meaningful word
    return tokens[i]




def compute_structural_entropy_from_questions(questions):
    # Step 1: Extract WH-type templates
    templates = [get_wh_template(q) for q in questions]

    # Step 2: Count frequencies
    counts = Counter(templates)
    total = sum(counts.values())
    num_classes = len(counts)

    # Step 3: Compute entropy
    entropy = -sum(
        (count / total) * math.log(count / total, 2)
        for count in counts.values()
    )

    # Step 4: Normalize
    balanced_entropy = (entropy * len(questions)/num_classes) / (entropy * (len(questions)/num_classes) + num_classes)


    return balanced_entropy* 100







def find_mterics(all_IUs, necessary_IUs, metadata, results, covered_flags):
    

    #finding coverage
    covered_necessary_count = sum(
        1 for iu, meta, flag in zip(all_IUs, metadata, covered_flags)
        if meta["type"] == "necessary" and flag == 1
    )
    coverage = covered_necessary_count/len(necessary_IUs)




    ##finding answerability
    questions_answerable, questions_unanswerable = split_questions_by_answerability(results)
    total_questions = questions_answerable + questions_unanswerable
    answerability = len(questions_answerable)/len(total_questions)

    if(len(questions_answerable)==0):
        questions_answerable=["1"]


    #finding redundancy
    redundant_groups = group_questions_by_exact_answer(results)
    total_questions_in_redundant_groups = sum(len(group["questions"]) for group in redundant_groups)
    number_of_redundant_questions = total_questions_in_redundant_groups - len(redundant_groups)
    redundancy = round(number_of_redundant_questions/len(questions_answerable),3)



    #finding structural entropy
    struct_entropy = compute_structural_entropy_from_questions(questions_answerable)

    return coverage*100, answerability*100, (1-redundancy)*100, struct_entropy






def plot_spider_web_chart(set1, set2):

    # Metrics (axes labels)
    categories = ['Answerability', 'Non-Redundancy', 'Coverage', 'Structural Entropy']
    num_vars = len(categories)

    # Values for two datasets
    values_1 = set1
    values_2 = set2

    # Repeat first value to close the circle
    values_1 += values_1[:1]
    values_2 += values_2[:1]

    # Calculate angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot first set
    ax.plot(angles, values_1, label='Set 1', color='red')
    ax.fill(angles, values_1, color='red', alpha=0.25)

    # Plot second set
    ax.plot(angles, values_2, label='Set 2', color='blue')
    ax.fill(angles, values_2, color='blue', alpha=0.25)

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    # Set the range for radial axes
    ax.set_ylim(0, 100)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Show plot
    plt.title("Spider Web Chart for Two Question Sets")
    plt.show()

