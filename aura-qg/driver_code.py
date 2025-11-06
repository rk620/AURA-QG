import json
import modules as md
from sentence_transformers import SentenceTransformer


def run_comparison_pipeline(markdown_text, questions_1, questions_2):

    print("Processing for scores:")
    print("\n")
    blocks = md.extract_markdown_text_blocks(markdown_text)
    necessary_IUs = md.clean_pdf_blocks_and_build_necessary_IUs(blocks)

    necessary_metadata = []
    for i, chunk in enumerate(necessary_IUs):
        necessary_metadata.append({
            "text": chunk["text"],
            "type": "necessary",
            "subtype": chunk["subtype"],
            "chunk_id": i,
            "parent_chunk_id": i
        })
    necessary_IUs = [chunk["text"] for chunk in necessary_IUs]

    optional_items_with_parents = md.split_list_blocks_to_individual_items_with_parent_ids(necessary_metadata)

    optional_bullets_metadata = []
    for i, (chunk, parent_id) in enumerate(optional_items_with_parents):
        optional_bullets_metadata.append({
            "text": chunk,
            "type": "optional",
            "subtype": "bullet",
            "chunk_id": len(necessary_IUs) + i,
            "parent_chunk_id": parent_id
        })
    optional_bullets = [text for text, _ in optional_items_with_parents]

    optional_windows = md.extract_sentence_windows_from_paragraphs(necessary_metadata)

    optional_window_metadata = md.generate_optional_window_metadata_cosine(
        optional_windows=optional_windows,
        necessary_metadata=necessary_metadata,
        model = SentenceTransformer('all-MiniLM-L6-v2'),
        start_index=len(necessary_IUs) + len(optional_bullets)
    )

    optional_IUs = optional_bullets + optional_windows
    all_IUs = necessary_IUs + optional_IUs
    metadata = necessary_metadata + optional_bullets_metadata + optional_window_metadata

    results_1, covered_flags_1 = md.building_faiss_index(all_IUs, metadata, questions_1, top_k=5)

    coverage_1, answerability_1, redundancy_1, struct_entropy_1 = md.find_mterics(
        all_IUs, necessary_IUs, metadata, results_1, covered_flags_1
    )
    print("#"*80)
    print("\n")
    print("Metric Values for set 1")
    print("Coverage: ", coverage_1)
    print("Answerability: ", answerability_1)
    print("Non-Redundancy: ", redundancy_1)
    print("Structural Entropy: ", struct_entropy_1)
    print("\n")
    results_2, covered_flags_2 = md.building_faiss_index(all_IUs, metadata, questions_2, top_k=5)

    coverage_2, answerability_2, redundancy_2, struct_entropy_2 = md.find_mterics(
        all_IUs, necessary_IUs, metadata, results_2, covered_flags_2
    )
    print("#"*80)
    print("\n")
    print("Metric Values for set 2")
    print("Coverage: ", coverage_2)
    print("Answerability: ", answerability_2)
    print("Non-Redundancy: ", redundancy_2)
    print("Structural Entropy: ", struct_entropy_2)
    print("\n")
    print("#"*80)
    print("\n")
    set1 = [answerability_1, redundancy_1, coverage_1, struct_entropy_1]
    set2 = [answerability_2, redundancy_2, coverage_2, struct_entropy_2]

    md.plot_spider_web_chart(set1, set2)



if __name__ == "__main__":
    json_path = "sample_input.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    passage = data["passage"]
    questions1 = data["questions1"]
    questions2 = data["questions2"]
    print("\n")
    print("*"*100)
    print("\n")
    print("AURA-QG: Automated Unsupervised Replicable Assessment for Question Generation\n")
    print("\n")
    print("*"*100)
    print("\n")
    run_comparison_pipeline(passage, questions1, questions2)
